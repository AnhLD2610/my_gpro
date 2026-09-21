# Copyright 2026 The Nash Credit Routing Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real CLI and shell subprocesses constrained to offline CPU-only paths."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from diagnostic_b import storage

EXP_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def subprocess_environment(tmp_path):
    guard_dir = tmp_path / "import_guard"
    guard_dir.mkdir()
    guard_log = tmp_path / "forbidden_imports.txt"
    # A startup hook makes the assertion effective inside real CLI subprocesses.
    (guard_dir / "sitecustomize.py").write_text(
        """import builtins
import os

_original_import = builtins.__import__
_forbidden = {'torch', 'transformers', 'datasets', 'vllm', 'huggingface_hub'}

def _guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in _forbidden:
        with open(os.environ['NASH_TEST_IMPORT_GUARD_LOG'], 'a') as handle:
            handle.write(name + '\\n')
        raise AssertionError('Model/dataset library imported in CPU-only CLI: ' + name)
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import

if os.environ.get('NASH_TEST_FORCE_GENERATION_FAILURE') == '1':
    import diagnostic_b.smoke as _smoke

    def _fail_generate(*args, **kwargs):
        raise RuntimeError('forced synthetic generation failure ' + os.environ['NASH_TEST_SECRET'])

    _smoke.SyntheticEngine.generate = _fail_generate
""",
        encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NASH_")}
    environment.update(
        PYTHONPATH=os.pathsep.join([str(guard_dir), str(EXP_ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        NASH_PYTHON=sys.executable,
        NASH_TEST_IMPORT_GUARD_LOG=str(guard_log),
    )
    yield environment
    assert not guard_log.exists(), "A CPU-only CLI path imported a model/dataset library"


def _run_cli(environment, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "diagnostic_b", *map(str, arguments)],
        cwd=EXP_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _approved_synthetic_config(tmp_path):
    config = yaml.safe_load((EXP_ROOT / "configs" / "diagnostic_b_round1.yaml").read_text())
    wrapper = tmp_path / "approved_wrapper.json"
    wrapper.write_text(json.dumps({"user_template": "Solve this problem: {problem}"}))
    config["prompt"]["wrapper_file"] = str(wrapper)
    config["sampling"]["max_new_tokens"] = 256
    config["routing"]["radius_coefficient"] = 0.2
    config["seed"] = 42
    config["verifier"]["backend"] = "math_verify"
    path = tmp_path / "approved_round1.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _unresolved_config(tmp_path):
    config = yaml.safe_load((EXP_ROOT / "configs" / "diagnostic_b_round1.yaml").read_text())
    for field in (
        "sampling.max_new_tokens",
        "routing.radius_coefficient",
        "seed",
        "prompt.wrapper_file",
        "verifier.backend",
        "train.id",
        "train.revision",
    ):
        target = config
        components = field.split(".")
        for component in components[:-1]:
            target = target[component]
        target[components[-1]] = None
    path = tmp_path / "unresolved_round1.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_real_cpu_smoke_cli_and_resume_preserve_generation_caches(tmp_path, subprocess_environment):
    output = tmp_path / "smoke"
    first = _run_cli(subprocess_environment, "--cpu-smoke", "--output-dir", output)
    assert first.returncode == 0, first.stdout + first.stderr
    state = storage.read_json(output / "stage_state.json")
    assert all(stage["status"] == "complete" for stage in state["stages"].values())
    assert state["last_completed_stage"] == "statistics"
    cache_names = ("train_rollouts.jsonl.zst", "heldout_rollouts.jsonl.zst")
    cache_identity = {
        name: (storage.file_hash(output / name), (output / name).stat().st_mtime_ns) for name in cache_names
    }
    assert len(list(storage.read_compressed_records(output / cache_names[0]))) == 16
    assert len(list(storage.read_compressed_records(output / cache_names[1]))) == 8
    results = storage.read_json(output / "correlations.json")
    assert results["upstream_summary"]["no_model_or_dataset_loaded"] is True
    assert "SYNTHETIC CPU FIXTURE ONLY" in results["upstream_summary"]["notice"]
    assert results["populations"]["all_eligible"]["n_directions"] == 8
    assert results["populations"]["all_eligible"]["n_prompts"] == 4
    for suffix in ("pdf", "png"):
        assert (output / f"diagnostic_b_support_vs_usefulness_round_cpu_smoke.{suffix}").is_file()
    second = _run_cli(subprocess_environment, "--cpu-smoke", "--output-dir", output, "--resume")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "stage=generate SKIP" in second.stdout
    for name, identity in cache_identity.items():
        assert (storage.file_hash(output / name), (output / name).stat().st_mtime_ns) == identity
    assert storage.read_json(output / "stage_state.json")["stages"]["generate"] == state["stages"]["generate"]
    assert (output / "report_round_cpu_smoke.md").is_file()


def test_dry_run_unresolved_author_fields_are_all_logged_before_model_or_data_imports(tmp_path, subprocess_environment):
    output = tmp_path / "unresolved"
    path = _unresolved_config(tmp_path)
    process = _run_cli(subprocess_environment, "--dry-run", "--config", path, "--output-dir", output)
    assert process.returncode == 1
    text = (output / "diagnostic_b_round1.log").read_text()
    for field in (
        "sampling.max_new_tokens",
        "routing.radius_coefficient",
        "seed",
        "prompt.wrapper_file",
        "verifier.backend",
        "train.id",
        "train.revision",
    ):
        assert f"unresolved author setting: {field}" in text
    assert "COMMAND:" in text and "RESUME:" in text and "Traceback" in text
    assert "Return this one log file:" in text
    assert (output / "config_resolved.yaml").is_file()
    assert (output / "environment.json").is_file()
    assert not (output / "stage_state.json").exists()
    assert process.stderr == ""
    assert process.stdout.count("Traceback (most recent call last):") == text.count(
        "Traceback (most recent call last):"
    )


def test_dry_run_validates_complete_locked_config_without_loading_anything(tmp_path, subprocess_environment):
    path = _approved_synthetic_config(tmp_path)
    output = tmp_path / "valid_dry_run"
    process = _run_cli(subprocess_environment, "--dry-run", "--config", path, "--output-dir", output)
    assert process.returncode == 0, process.stdout + process.stderr
    text = (output / "diagnostic_b_round1.log").read_text()
    assert "DRY_RUN_VALID" in text
    resolved = yaml.safe_load((output / "config_resolved.yaml").read_text())
    assert resolved["train"]["n_problem"] == 500
    assert resolved["train"]["selection"] == "sample"
    assert resolved["heldout"]["n_problem"] == 500
    assert resolved["heldout"]["selection"] == "all"
    assert resolved["n_rollout"] == 32
    assert not (output / "stage_state.json").exists()
    assert not (output / "train_rollouts.jsonl.zst").exists()


def test_round2_shell_launcher_refuses_before_opening_any_stage(tmp_path, subprocess_environment):
    output = tmp_path / "must_not_exist"
    environment = dict(subprocess_environment, NASH_OUTPUT_DIR=str(output), NASH_PYTHON="/must/not/run/python")
    process = subprocess.run(
        ["bash", str(EXP_ROOT / "scripts" / "run_diagnostic_b_round2_gpu.sh")],
        cwd=EXP_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert process.returncode == 2
    assert "Round 2 is disabled" in process.stderr
    assert "--enable-round2" in process.stderr
    assert not output.exists()


def test_forced_cli_exception_is_logged_with_no_second_unlogged_traceback(tmp_path, subprocess_environment):
    output = tmp_path / "forced_failure"
    secret = "synthetic-private-token-for-cli-test"
    environment = dict(subprocess_environment, NASH_TEST_FORCE_GENERATION_FAILURE="1", NASH_TEST_SECRET=secret)
    process = _run_cli(environment, "--cpu-smoke", "--output-dir", output)
    assert process.returncode == 1
    text = (output / "diagnostic_b_round_cpu_smoke.log").read_text()
    assert "forced synthetic generation failure" in text
    assert "stage=generate" in text
    assert "RESUME:" in text and "--resume" in text
    assert "<redacted>" in text
    assert secret not in text + process.stdout + process.stderr
    assert process.stderr == ""
    assert process.stdout.count("Traceback (most recent call last):") == text.count(
        "Traceback (most recent call last):"
    )
    state = storage.read_json(output / "stage_state.json")
    assert state["stages"]["generate"]["status"] == "failed"
    assert state["last_completed_stage"] == "prepare"


def test_round1_shell_launcher_logs_missing_settings_and_resume_hint(tmp_path, subprocess_environment):
    output = tmp_path / "launcher_failure"
    environment = dict(
        subprocess_environment, NASH_OUTPUT_DIR=str(output), NASH_CONFIG=str(_unresolved_config(tmp_path))
    )
    process = subprocess.run(
        ["bash", str(EXP_ROOT / "scripts" / "run_diagnostic_b_round1_gpu.sh")],
        cwd=EXP_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 1
    text = (output / "diagnostic_b_round1.log").read_text()
    assert "unresolved author setting: sampling.max_new_tokens" in text
    assert "unresolved author setting: routing.radius_coefficient" in text
    assert "Launcher failed: exit=1" in text
    assert "Resume: bash scripts/run_diagnostic_b_round1_gpu.sh" in text
    assert "Return this one log file:" in text
    assert "Command:" in text and "COMMAND:" in text
    assert not (output / "stage_state.json").exists()
