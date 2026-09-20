# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Resumable Diagnostic B CLI; all GPU-heavy stages run in separate processes."""

import argparse
import fcntl
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from .logging_utils import environment_report, single_log
from .storage import atomic_json, read_json

STAGES = (
    "preflight",
    "prepare",
    "generate",
    "verify",
    "heldout_features",
    "training_features_and_routes",
    "statistics",
)


def parser():
    """CLI parsing remains available even when scientific dependencies are missing."""
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description="Frozen-checkpoint NCR Diagnostic B (no training update)")
    result.add_argument("stage", nargs="?", choices=(*STAGES, "all"), default="all")
    result.add_argument("--config", type=Path, default=root / "configs" / "diagnostic_b_round1.yaml")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--force-stage", choices=STAGES)
    result.add_argument(
        "--dry-run", action="store_true", help="Validate config and paths only; no model/dataset loading"
    )
    result.add_argument("--cpu-smoke", action="store_true", help="Tiny synthetic fixture; no external data or GPU")
    result.add_argument("--enable-round2", action="store_true", help="Use only after explicit author approval")
    return result


def interrupted(signum, frame):
    """Unwind through stage state and single-file logging on termination."""
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def main(argv=None):
    """Return an exit status without emitting an unlogged second traceback."""
    args = parser().parse_args(argv)
    base = Path(__file__).resolve().parents[1]
    # Use filename only to open a log even if YAML or required dependencies fail to load.
    label = "round_cpu_smoke" if args.cpu_smoke else ("round2" if "round2" in args.config.name else "round1")
    output = (
        args.output_dir or Path(os.environ.get("NASH_OUTPUT_DIR", base / "artifacts" / "diagnostic_b" / label))
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logfile = output / f"diagnostic_b_{label}.log"
    command = [sys.executable, "-m", "diagnostic_b", *(sys.argv[1:] if argv is None else argv)]
    resume = [
        sys.executable,
        "-m",
        "diagnostic_b",
        "all",
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str(output),
        "--resume",
    ]
    if args.cpu_smoke:
        resume.append("--cpu-smoke")
    if args.enable_round2:
        resume.append("--enable-round2")
    old_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        with single_log(logfile, command, resume):
            log = logging.getLogger(__name__)
            environment = environment_report(output, gpu=False)
            if (output / "environment.json").exists():
                prior = read_json(output / "environment.json")
                for key in ("nvidia_smi", "pytorch_compiled_cuda", "installed_cvxpy_solvers"):
                    if key in prior:
                        environment[key] = prior[key]
            atomic_json(output / "environment.json", environment)
            log.info("Environment: %s", environment)
            if args.cpu_smoke:
                from .smoke import run_cpu_smoke

                log.info("%s", run_cpu_smoke(output, resume=args.resume))
                return 0
            import dataclasses

            import yaml

            from .config import load_config, validate_config
            from .routing import SolverConfig

            config = load_config(args.config)
            config["routing"]["solver"] = dataclasses.asdict(SolverConfig(**config["routing"]["solver"]))
            log.info("Resolved config before execution:\n%s", yaml.safe_dump(config, sort_keys=False))
            (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            validate_config(config, enable_round2=args.enable_round2)
            if args.dry_run:
                from .data import validate_wrapper

                validate_wrapper(read_json(config["prompt"]["wrapper_file"]))
                log.info("DRY_RUN_VALID: config, wrapper and paths checked; no model or dataset loaded")
                return 0
            if args.stage == "preflight":
                environment = environment_report(output, gpu=True)
                try:
                    import torch

                    environment["pytorch_compiled_cuda"] = torch.version.cuda
                except ImportError:
                    environment["pytorch_compiled_cuda"] = "PyTorch not installed"
                import cvxpy

                environment["installed_cvxpy_solvers"] = cvxpy.installed_solvers()
                atomic_json(output / "environment.json", environment)
                log.info("Server GPU/CUDA/solver environment: %s", environment)
            if args.stage == "all":
                # Releasing each process guarantees vLLM and the feature model never coexist.
                for stage in STAGES:
                    child = [
                        sys.executable,
                        "-m",
                        "diagnostic_b",
                        stage,
                        "--config",
                        str(args.config.resolve()),
                        "--output-dir",
                        str(output),
                    ]
                    if args.resume:
                        child.append("--resume")
                    if args.force_stage:
                        child.extend(["--force-stage", args.force_stage])
                    if args.enable_round2:
                        child.append("--enable-round2")
                    log.info("Starting isolated stage: %s", child)
                    env = dict(os.environ, NASH_LOG_TEE="1")
                    with subprocess.Popen(
                        child, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
                    ) as process:
                        try:
                            for line in process.stdout:
                                print(line, end="", flush=True)
                            code = process.wait()
                        except BaseException:
                            process.terminate()
                            process.wait(timeout=30)
                            raise
                    if code:
                        raise RuntimeError(f"Stage {stage} exited {code}; see its full traceback above")
                log.info("Diagnostic complete: %s", output / f"report_{config['round_id']}.md")
            else:
                from .pipeline import run_stage

                with (output / ".stage.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    run_stage(
                        args.stage,
                        config,
                        output,
                        resume=args.resume,
                        force_stage=args.force_stage,
                        enable_round2=args.enable_round2,
                    )
            if (output / "model_lock.json").exists():
                resolved = dict(config, resolved_model=read_json(output / "model_lock.json"))
                (output / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
