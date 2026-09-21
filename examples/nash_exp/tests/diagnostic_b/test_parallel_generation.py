"""Real CPU subprocesses exercise split isolation, resume and failure cleanup."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from diagnostic_b import parallel_generation
from diagnostic_b.config import ConfigurationError
from diagnostic_b.generation import generate_shared
from diagnostic_b.storage import (
    CacheConflict,
    atomic_json,
    file_hash,
    read_compressed_records,
    read_json,
    write_records,
)
from test_generation import fixture_config

FAKE_VLLM = """
import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

SamplingParams = SimpleNamespace

class LLM:
    def __init__(self, **kwargs):
        self.calls = 0
        self.llm_engine = self
        self.active = {}
        self.directory = pathlib.Path(os.environ["NASH_TEST_ENGINE_DIR"])
        # Match vLLM 0.11: UUID masks fail during platform initialization.
        devices = [int(identifier) for identifier in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        self.split = "heldout" if devices[0] >= 4 else "train"
        self.replicas = int(os.environ.get("NASH_TEST_REPLICAS", "1"))
        self.label = self.split if self.replicas == 1 else self.split + ":" + str(devices[0] % 4)
        self.directory.joinpath(self.label + ".ready").write_text(json.dumps({
            "devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), "kwargs": kwargs
        }))

    def _before_generation(self):
        self.calls += 1
        if os.environ.get("NASH_TEST_SERIAL_FAIL") and self.calls == 2:
            raise RuntimeError("serial interruption")
        if os.environ.get("NASH_TEST_BARRIER"):
            deadline = time.monotonic() + 15
            labels = [split if self.replicas == 1 else split + ":" + str(index)
                for split in ("train", "heldout") for index in range(self.replicas)]
            while not all(self.directory.joinpath(label + ".ready").exists() for label in labels):
                if time.monotonic() > deadline:
                    raise RuntimeError("both workers did not start concurrently")
                time.sleep(0.02)
        if os.environ.get("NASH_TEST_FAILURE"):
            if self.split == "train":
                code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
                child = subprocess.Popen([sys.executable, "-c", code])
                self.directory.joinpath("descendant.pid").write_text(str(child.pid))
                time.sleep(120)
            else:
                deadline = time.monotonic() + 15
                while not self.directory.joinpath("descendant.pid").exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError("sibling did not start")
                    time.sleep(0.02)
                raise RuntimeError("simulated heldout failure")
        if os.environ.get("NASH_TEST_HANG"):
            time.sleep(120)

    @staticmethod
    def _completion(prompt, request_id):
        return SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[7, 0], text="answer<EOS>", finish_reason="stop", stop_reason=0,
                logprobs=[{7: SimpleNamespace(logprob=-0.3)}, {0: SimpleNamespace(logprob=-0.3)}],
                cumulative_logprob=-0.6)],
            prompt_token_ids=prompt["prompt_token_ids"], finished=True, request_id=request_id
        )

    def generate(self, prompts, sampling_params, use_tqdm):
        self._before_generation()
        with self.directory.joinpath(self.label + ".seeds").open("a") as handle:
            for param in sampling_params:
                handle.write(str(param.seed) + "\\n")
        return [self._completion(prompt, "engine-id") for prompt in prompts]

    def add_request(self, request_id, prompt, params):
        assert params.output_kind == 2
        assert request_id not in self.active
        self.active[request_id] = (prompt, params)
        with self.directory.joinpath(self.label + ".seeds").open("a") as handle:
            handle.write(str(params.seed) + "\\n")

    def has_unfinished_requests(self):
        return bool(self.active)

    def step(self):
        self._before_generation()
        # Complete the most recently admitted response first; the supervisor
        # must restore canonical ordering without relying on completion order.
        request_id = next(reversed(self.active))
        prompt, params = self.active.pop(request_id)
        with self.directory.joinpath(self.label + ".completed").open("a") as handle:
            handle.write(request_id + "\\n")
        return [self._completion(prompt, request_id)]

    def abort_request(self, request_ids):
        for request_id in request_ids:
            self.active.pop(request_id, None)
        self.directory.joinpath(self.label + ".aborted").write_text(json.dumps(request_ids))
"""


@pytest.fixture
def fake_subprocess_engine(tmp_path, monkeypatch):
    modules = tmp_path / "fake_modules"
    modules.mkdir()
    package = modules / "vllm"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text(FAKE_VLLM)
    (package / "sampling_params.py").write_text(
        "from enum import IntEnum\nclass RequestOutputKind(IntEnum):\n    FINAL_ONLY = 2\n"
    )
    spec = importlib.util.spec_from_file_location("vllm", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vllm", module)
    package_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(modules), str(package_root), os.environ.get("PYTHONPATH", "")])
    )
    monkeypatch.setenv("NASH_TEST_ENGINE_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        parallel_generation,
        "device_groups",
        lambda *_: {"train": [{"index": 0, "uuid": "GPU-train"}], "heldout": [{"index": 4, "uuid": "GPU-heldout"}]},
    )
    cleanup = parallel_generation._cleanup_workers
    monkeypatch.setattr(
        parallel_generation, "_cleanup_workers", lambda processes: cleanup(processes, grace_seconds=0.2)
    )
    return tmp_path


def test_workers_overlap_and_use_disjoint_gpus_with_identical_request_contracts(fake_subprocess_engine, monkeypatch):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    config["generation"]["parallel_splits"] = True
    monkeypatch.setenv("NASH_TEST_BARRIER", "1")
    paths = generate_shared(config, root, lock)
    assert len(paths) == 3
    manifest = read_json(root / "generation_manifest.json")
    assert manifest["identity"]["engine_backend"] == "vllm"
    assert "parallel_splits" not in manifest["identity"]
    execution = read_json(root / "generation_execution.json")
    assert execution["status"] == "complete"
    assert len({worker["pid"] for worker in execution["workers"].values()}) == 2
    for split in ("train", "heldout"):
        engine = read_json(root / f"{split}.ready")
        assert engine["devices"] == ("0" if split == "train" else "4")
        assert execution["workers"][split]["cuda_visible_devices"] == engine["devices"]
        assert execution["workers"][split]["devices"][0]["uuid"] == f"GPU-{split}"
        assert engine["kwargs"] == manifest["identity"]["engine_kwargs"]
        assert manifest["splits"][split]["records"] == 3
        assert execution["workers"][split]["returncode"] == 0
    # Both files are already final: toggling back requires no further engine.
    before = {path: file_hash(path) for path in paths}
    config["generation"]["parallel_splits"] = False
    monkeypatch.setenv("NASH_TEST_SERIAL_FAIL", "1")
    assert generate_shared(config, root, lock) == paths
    assert before == {path: file_hash(path) for path in paths}


def test_partial_serial_run_resumes_in_parallel_without_changing_any_completed_part(
    fake_subprocess_engine, monkeypatch
):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    monkeypatch.setenv("NASH_TEST_SERIAL_FAIL", "1")
    with pytest.raises(RuntimeError, match="serial interruption"):
        generate_shared(config, root, lock)
    parts = list((root / "rollout_parts" / "train").glob("*.json"))
    assert len(parts) == 2
    before = {path: file_hash(path) for path in parts}
    identity_path = root / "rollout_parts" / "generation_identity.json"
    identity_hash = file_hash(identity_path)
    previous_seeds = {read_json(path)["seed"] for path in parts}
    (root / "train.ready").unlink()
    (root / "train.seeds").unlink()
    monkeypatch.delenv("NASH_TEST_SERIAL_FAIL")
    monkeypatch.setenv("NASH_TEST_BARRIER", "1")
    config["generation"]["parallel_splits"] = True
    generate_shared(config, root, lock)
    assert file_hash(identity_path) == identity_hash
    assert before == {path: file_hash(path) for path in parts}
    new_seeds = {
        int(seed) for split in ("train", "heldout") for seed in (root / f"{split}.seeds").read_text().splitlines()
    }
    assert len(new_seeds) == 4
    assert previous_seeds.isdisjoint(new_seeds)


def _alive(pid):
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split()[2] != "Z"


def test_failed_worker_terminates_sibling_and_descendant_without_global_manifest(fake_subprocess_engine, monkeypatch):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    config["generation"]["parallel_splits"] = True
    monkeypatch.setenv("NASH_TEST_BARRIER", "1")
    monkeypatch.setenv("NASH_TEST_FAILURE", "1")
    with pytest.raises(RuntimeError, match="split=heldout exited"):
        generate_shared(config, root, lock)
    assert not (root / "generation_manifest.json").exists()
    execution = read_json(root / "generation_execution.json")
    assert execution["status"] == "failed"
    assert not any(_alive(worker["pid"]) for worker in execution["workers"].values())
    descendant = int((root / "descendant.pid").read_text())
    deadline = time.monotonic() + 3
    while _alive(descendant) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(descendant)


def test_parent_sigterm_cleans_both_workers(fake_subprocess_engine, monkeypatch):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    config["generation"]["parallel_splits"] = True
    job = root / "test_job.json"
    atomic_json(job, {"config": config, "lock": lock})
    monkeypatch.setenv("NASH_TEST_BARRIER", "1")
    monkeypatch.setenv("NASH_TEST_HANG", "1")
    script = """
import signal,sys
from pathlib import Path
from diagnostic_b import parallel_generation
from diagnostic_b.__main__ import interrupted
from diagnostic_b.generation import generate_shared
from diagnostic_b.storage import read_json
signal.signal(signal.SIGTERM, interrupted)
parallel_generation.device_groups = lambda *_: {
    "train": [{"index": 0, "uuid": "GPU-train"}], "heldout": [{"index": 4, "uuid": "GPU-heldout"}]
}
job = read_json(sys.argv[1])
generate_shared(job["config"], Path(sys.argv[1]).parent, job["lock"])
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(job)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    ) as parent:
        try:
            deadline = time.monotonic() + 20
            while not all((root / f"{split}.ready").exists() for split in ("train", "heldout")):
                assert parent.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.05)
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=20)
        finally:
            if parent.poll() is None:
                parent.terminate()
                parent.wait(timeout=20)
    execution = read_json(root / "generation_execution.json")
    assert execution["status"] == "interrupted"
    assert not any(_alive(worker["pid"]) for worker in execution["workers"].values())
    assert not (root / "generation_manifest.json").exists()


def test_invalid_heldout_contract_prevents_both_engines(fake_subprocess_engine):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    config["generation"]["parallel_splits"] = True
    write_records(
        root / "heldout_prompt_manifest.parquet",
        [
            {
                "prompt_index": 0,
                "row_id": "bad",
                "prompt_hash": "bad",
                "prompt_token_ids": [3, 4],
                "prompt_token_count": 1,
            }
        ],
    )
    with pytest.raises(CacheConflict, match="token count"):
        generate_shared(config, root, lock)
    assert not (root / "train.ready").exists()
    assert not (root / "heldout.ready").exists()
    assert not (root / "generation_manifest.json").exists()


@pytest.mark.parametrize("uuids", [["GPU-0"], ["GPU-0", "GPU-0"]])
def test_parallel_device_validation_rejects_insufficient_or_duplicate_devices(tmp_path, monkeypatch, uuids):
    from diagnostic_b import preflight

    config, _ = fixture_config(tmp_path)
    monkeypatch.setattr(
        preflight,
        "detect_hardware",
        lambda *_: {"gpu_devices": [{"index": index, "uuid": uuid} for index, uuid in enumerate(uuids)]},
    )
    with pytest.raises(ConfigurationError, match="visible GPUs"):
        parallel_generation.device_groups(config, tmp_path)


def test_parallel_device_assignment_uses_visible_order_and_per_engine_tp(tmp_path, monkeypatch):
    from diagnostic_b import preflight

    config, _ = fixture_config(tmp_path)
    config["generation"]["tensor_parallel_size"] = 2
    monkeypatch.setattr(
        preflight,
        "detect_hardware",
        lambda *_: {"gpu_devices": [{"index": index, "uuid": f"GPU-{index}"} for index in (7, 4, 6, 1)]},
    )
    assert parallel_generation.device_groups(config, tmp_path) == {
        "train": [{"index": 7, "uuid": "GPU-7"}, {"index": 4, "uuid": "GPU-4"}],
        "heldout": [{"index": 6, "uuid": "GPU-6"}, {"index": 1, "uuid": "GPU-1"}],
    }


def test_four_replicas_assign_eight_distinct_single_gpu_engines(tmp_path, monkeypatch):
    from diagnostic_b import preflight

    config, _ = fixture_config(tmp_path)
    config["generation"].update(replicas_per_split=4, tensor_parallel_size=1)
    order = [7, 4, 6, 1, 3, 2, 0, 5]
    monkeypatch.setattr(
        preflight,
        "detect_hardware",
        lambda *_: {"gpu_devices": [{"index": index, "uuid": f"GPU-{index}"} for index in order]},
    )
    groups = parallel_generation.device_groups(config, tmp_path)
    assert list(groups) == [f"{split}:{index}" for split in ("train", "heldout") for index in range(4)]
    assert [group[0]["index"] for group in groups.values()] == order
    config["generation"]["tensor_parallel_size"] = 2
    with pytest.raises(ConfigurationError, match="16 visible GPUs"):
        parallel_generation.device_groups(config, tmp_path)


@pytest.mark.parametrize("replicas", [True, 0, -1, 1.5])
def test_invalid_replica_count_is_rejected_before_worker_launch(tmp_path, monkeypatch, replicas):
    from diagnostic_b import preflight

    config, _ = fixture_config(tmp_path)
    config["generation"]["replicas_per_split"] = replicas
    monkeypatch.setattr(preflight, "detect_hardware", lambda *_: {"gpu_devices": []})
    with pytest.raises(ConfigurationError, match="replicas_per_split"):
        parallel_generation.device_groups(config, tmp_path)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("dispatch_mode", ["batch", "continuous"])
def test_replicas_keep_prompt_draws_together_and_finalize_canonical_cache(
    fake_subprocess_engine, monkeypatch, resume, dispatch_mode
):
    root = fake_subprocess_engine
    config, lock = fixture_config(root)
    config["generation"].update(dispatch_mode=dispatch_mode, max_in_flight=2)
    for split, first_token in (("train", 1), ("heldout", 3)):
        config[split]["n_problem"] = 4
        write_records(
            root / f"{split}_prompt_manifest.parquet",
            [
                {
                    "prompt_index": index,
                    "row_id": f"{split}-row-{index}",
                    "prompt_hash": f"{split}-hash-{index}",
                    "prompt_token_ids": [first_token, index],
                    "prompt_token_count": 2,
                }
                for index in range(4)
            ],
        )
    preserved = {}
    if resume:
        monkeypatch.setenv("NASH_TEST_SERIAL_FAIL", "1")
        with pytest.raises(RuntimeError, match="serial interruption"):
            generate_shared(config, root, lock)
        preserved = {path: file_hash(path) for path in (root / "rollout_parts" / "train").glob("*.json")}
        assert len(preserved) == (1 if dispatch_mode == "continuous" else 2)
        if dispatch_mode == "continuous":
            assert read_json(root / "train.aborted")
        monkeypatch.delenv("NASH_TEST_SERIAL_FAIL")
    old_seeds = {read_json(path)["seed"] for path in preserved}
    config["generation"].update(parallel_splits=True, replicas_per_split=4)
    monkeypatch.setenv("NASH_TEST_REPLICAS", "4")
    monkeypatch.setenv("NASH_TEST_BARRIER", "1")
    monkeypatch.setattr(
        parallel_generation,
        "device_groups",
        lambda *_: {
            f"{split}:{index}": [{"index": offset + index, "uuid": f"GPU-{offset + index}"}]
            for split, offset in (("train", 0), ("heldout", 4))
            for index in range(4)
        },
    )
    paths = generate_shared(config, root, lock)
    execution = read_json(root / "generation_execution.json")
    assert execution["status"] == "complete"
    assert execution["replicas_per_split"] == 4
    assert len({worker["pid"] for worker in execution["workers"].values()}) == 8
    for split in ("train", "heldout"):
        records = list(read_compressed_records(root / f"{split}_rollouts.jsonl.zst"))
        assert [(row["prompt_index"], row["rollout_index"]) for row in records] == [
            (index, draw) for index in range(4) for draw in range(3)
        ]
        for index in range(4):
            worker = execution["workers"][f"{split}:{index}"]
            assert (worker["split"], worker["shard_index"], worker["shard_count"]) == (split, index, 4)
            assert worker["returncode"] == 0
            seeds = [int(seed) for seed in (root / f"{split}:{index}.seeds").read_text().splitlines()]
            expected = [row["seed"] for row in records if row["prompt_index"] == index and row["seed"] not in old_seeds]
            assert seeds == expected
            if dispatch_mode == "continuous":
                metrics = read_json(root / "generation_metrics" / f"{split}-{index:03d}.json")
                assert metrics["dispatch_mode"] == "continuous"
                assert metrics["generated_requests"] == len(expected)
                assert metrics["reused_requests"] == 3 - len(expected)
                assert metrics["output_tokens"] == 2 * len(expected)
                completed = (root / f"{split}:{index}.completed").read_text().splitlines()
                canonical_ids = [
                    row["request_id"]
                    for row in records
                    if row["prompt_index"] == index and row["seed"] not in old_seeds
                ]
                assert set(completed) == set(canonical_ids)
                assert completed != canonical_ids
    assert preserved == {path: file_hash(path) for path in preserved}
    before = {path: file_hash(path) for path in paths}
    config["generation"].update(parallel_splits=False, replicas_per_split=1)
    monkeypatch.setenv("NASH_TEST_SERIAL_FAIL", "1")
    assert generate_shared(config, root, lock) == paths
    assert before == {path: file_hash(path) for path in paths}
