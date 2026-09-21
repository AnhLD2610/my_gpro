"""Isolated prompt-shard workers; execution topology never changes rollout identity.

The stage process owns the global manifest and stage lock. These hidden workers
receive a resolved config snapshot and only write their assigned request parts.
The supervisor assembles sharded split caches after every worker has succeeded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .config import ConfigurationError
from .storage import CacheConflict, atomic_json, read_json

LOG = logging.getLogger(__name__)


def _replica_count(config):
    replicas = config["generation"].get("replicas_per_split", 1)
    if type(replicas) is not int or replicas <= 0:
        raise ConfigurationError("generation.replicas_per_split must be a positive integer")
    return replicas


def _worker_specs(replicas):
    for split in ("train", "heldout"):
        for shard_index in range(replicas):
            key = split if replicas == 1 else f"{split}:{shard_index}"
            yield key, split, shard_index


def device_groups(config, output_dir):
    """Resolve physical indices and UUIDs without initializing supervisor CUDA."""
    from .preflight import detect_hardware

    devices = detect_hardware(output_dir)["gpu_devices"]
    identifiers = [device["uuid"] for device in devices]
    indices = [device["index"] for device in devices]
    if len(identifiers) != len(set(identifiers)) or len(indices) != len(set(indices)):
        raise ConfigurationError("Parallel generation requires distinct visible GPUs")
    tp = config["generation"]["tensor_parallel_size"]
    if not isinstance(tp, int) or isinstance(tp, bool) or tp <= 0:
        raise ConfigurationError("generation.tensor_parallel_size must be positive")
    replicas = _replica_count(config)
    engine_count = 2 * replicas
    if len(identifiers) < engine_count * tp:
        raise ConfigurationError(
            f"Parallel generation needs {engine_count * tp} visible GPUs ({engine_count} TP{tp} engines)"
        )
    selected = [{"index": device["index"], "uuid": device["uuid"]} for device in devices[: engine_count * tp]]
    return {
        key: selected[position * tp : (position + 1) * tp]
        for position, (key, _, _) in enumerate(_worker_specs(replicas))
    }


def _group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _cleanup_workers(processes, *, grace_seconds=10):
    """Reap owned process groups, including descendants of an exited worker."""
    for process in processes.values():
        _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        # poll() reaps exited direct children before checking their groups.
        for process in processes.values():
            process.poll()
        if not any(_group_exists(process.pid) for process in processes.values()):
            break
        time.sleep(0.05)
    for process in processes.values():
        _signal_group(process.pid, signal.SIGKILL)
    for process in processes.values():
        process.wait()


def _forward_output(stream):
    # Forward through the supervisor's TeeStream so direct Python invocation and
    # the shell launcher both keep one complete log, without pipe backpressure.
    try:
        for line in stream:
            print(line, end="", flush=True)
    finally:
        stream.close()


def run_split_workers(config, root, model_lock, contract):
    """Launch disjoint prompt shards, then finalize and reap owned processes."""
    replicas = _replica_count(config)
    groups = device_groups(config, root)
    execution = {
        "mode": "parallel_splits",
        "generation_fingerprint": contract["fingerprint"],
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device_groups": groups,
        "replicas_per_split": replicas,
        "status": "running",
        "workers": {},
    }
    execution_path = root / "generation_execution.json"
    atomic_json(execution_path, execution)
    processes = {}
    readers = []
    # The resolved snapshot avoids re-reading config/env overrides in children.
    # CUDA_VISIBLE_DEVICES is the sole per-worker hardware assignment.
    with tempfile.TemporaryDirectory(prefix=".generation_workers_", dir=root) as work_dir:
        job_path = Path(work_dir) / "job.json"
        atomic_json(
            job_path,
            {
                "config": config,
                "output_dir": str(root.resolve()),
                "model_lock": model_lock,
                "identity": contract["identity"],
            },
        )
        try:
            for key, split, shard_index in _worker_specs(replicas):
                if (root / f"{split}_rollouts.jsonl.zst").exists():
                    LOG.info("Parallel generation reuses completed split=%s", split)
                    continue
                # vLLM 0.11 parses each visible-device ID with int(). UUIDs
                # remain in execution metadata, but its CUDA mask must be numeric.
                visible = ",".join(str(device["index"]) for device in groups[key])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
                command = [sys.executable, "-m", "diagnostic_b.parallel_generation", str(job_path), split]
                if replicas > 1:
                    command.extend(["--shard-index", str(shard_index), "--shard-count", str(replicas)])
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                processes[key] = process
                execution["workers"][key] = {
                    "pid": process.pid,
                    "split": split,
                    "shard_index": shard_index,
                    "shard_count": replicas,
                    "devices": groups[key],
                    "cuda_visible_devices": visible,
                }
                atomic_json(execution_path, execution)
                reader = threading.Thread(target=_forward_output, args=(process.stdout,), daemon=True)
                readers.append(reader)
                reader.start()
                LOG.info("Started parallel generation worker=%s pid=%s devices=%s", key, process.pid, groups[key])
            pending = set(processes)
            while pending:
                for key in tuple(pending):
                    code = processes[key].poll()
                    if code is None:
                        continue
                    execution["workers"][key]["returncode"] = code
                    if code:
                        raise RuntimeError(
                            f"Parallel generation split={key} exited with status {code}; completed parts preserved"
                        )
                    pending.remove(key)
                    LOG.info("Parallel generation worker complete split=%s", key)
                if pending:
                    time.sleep(0.1)
            if replicas > 1:
                from .generation import _finalize_split

                for split in ("train", "heldout"):
                    _finalize_split(root, split, contract["requests"][split], contract["fingerprint"])
            execution["status"] = "complete"
        except BaseException as exc:
            execution["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            execution["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _cleanup_workers(processes)
            for reader in readers:
                reader.join(timeout=2)
            for split, process in processes.items():
                execution["workers"][split]["returncode"] = process.returncode
            execution["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(execution_path, execution)


def _interrupted(signum, frame):
    raise KeyboardInterrupt(f"Split worker interrupted by signal {signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Internal Diagnostic B split generation worker")
    parser.add_argument("job", type=Path)
    parser.add_argument("split", choices=("train", "heldout"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-count must be positive and shard-index must be in [0, shard-count)")
    label = args.split if args.shard_count == 1 else f"{args.split}:{args.shard_index}"
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s %(levelname)s %(name)s [{label}] %(message)s", stream=sys.stdout
    )
    signal.signal(signal.SIGTERM, _interrupted)
    job = read_json(args.job)
    from .generation import _generate_split, _generation_contract, _LazyEngine

    root = Path(job["output_dir"])
    contract = _generation_contract(job["config"], root, job["model_lock"])
    if (
        contract["identity"] != job["identity"]
        or read_json(root / "rollout_parts" / "generation_identity.json") != job["identity"]
    ):
        raise CacheConflict("Worker generation contract changed after supervisor validation")
    engine = _LazyEngine(contract["engine_kwargs"])
    try:
        _generate_split(
            job["config"],
            root,
            args.split,
            contract,
            engine,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
