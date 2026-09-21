"""CPU fake-engine tests for immutable, shared and resumable generation."""

import copy
from types import SimpleNamespace

import pytest
from diagnostic_b.config import ConfigurationError
from diagnostic_b.generation import generate_shared, request_identity
from diagnostic_b.storage import CacheConflict, file_hash, read_compressed_records, read_json, write_records


def fixture_config(tmp_path):
    config = {
        "seed": 42,
        "round_id": "round1",
        "n_rollout": 3,
        "model": {"id": "toy", "dtype": "float32", "max_model_len": 32, "local_path": None},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_new_tokens": 4},
        "generation": {
            "request_chunk_size": 2,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": True,
        },
        "train": {"n_problem": 1},
        "heldout": {"n_problem": 1},
    }
    for split, tokens in (("train", [1, 2]), ("heldout", [3, 4])):
        write_records(
            tmp_path / f"{split}_prompt_manifest.parquet",
            [
                {
                    "prompt_index": 0,
                    "row_id": f"{split}-row",
                    "prompt_hash": f"{split}-hash",
                    "prompt_token_ids": tokens,
                    "prompt_token_count": 2,
                }
            ],
        )
    lock = {"model_revision": "immutable-model", "tokenizer_revision": "immutable-tokenizer"}
    return config, lock


class FakeFactory:
    def __init__(self, *, fail_call=None, empty_seed=None):
        self.instances = 0
        self.calls = []
        self.fail_call = fail_call
        self.empty_seed = empty_seed
        self.kwargs = None

    def __call__(self, **kwargs):
        self.instances += 1
        self.kwargs = kwargs
        return self

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append([param.seed for param in sampling_params])
        if self.fail_call == len(self.calls):
            raise RuntimeError("simulated generator failure")
        outputs = []
        for prompt, param in zip(prompts, sampling_params, strict=True):
            tokens = [] if param.seed == self.empty_seed else [7, 0]
            completion = SimpleNamespace(
                token_ids=tokens,
                text="answer<EOS>",
                finish_reason="stop",
                stop_reason=0,
                logprobs=[{token: SimpleNamespace(logprob=-0.3)} for token in tokens],
                cumulative_logprob=-0.6,
            )
            outputs.append(
                SimpleNamespace(
                    outputs=[completion],
                    prompt_token_ids=prompt["prompt_token_ids"],
                    finished=True,
                    request_id="engine-id",
                )
            )
        return outputs


def run(config, tmp_path, lock, factory):
    return generate_shared(config, tmp_path, lock, engine_factory=factory, sampler_factory=SimpleNamespace)


def test_one_engine_shared_between_splits_and_all_methods_reuse(tmp_path):
    config, lock = fixture_config(tmp_path)
    factory = FakeFactory()
    paths = run(config, tmp_path, lock, factory)
    assert factory.instances == 1
    assert factory.kwargs["generation_config"] == "vllm"
    assert factory.kwargs["trust_remote_code"] is False
    assert len(paths) == 3
    all_seeds = []
    for split in ("train", "heldout"):
        records = list(read_compressed_records(tmp_path / f"{split}_rollouts.jsonl.zst"))
        assert len(records) == 3
        for record in records:
            assert record["response_token_ids"] == [7, 0]
            assert record["token_log_probs"] == [-0.3, -0.3]
            assert record["output_token_count"] == 2
            assert record["sampling"]["top_k"] == -1
            assert record["sampling"]["min_p"] == 0.0
            assert record["sampling"]["logprobs"] == 0
            all_seeds.append(record["seed"])
    assert len(set(all_seeds)) == 6
    checksums = [file_hash(path) for path in paths]
    for _method in ("GRPO", "Linear", "NCR"):
        reused = FakeFactory()
        assert run(config, tmp_path, lock, reused) == paths
        assert reused.instances == 0
    assert [file_hash(path) for path in paths] == checksums


def test_resume_after_partial_request_batch_never_regenerates_committed_requests(tmp_path):
    config, lock = fixture_config(tmp_path)
    failure = FakeFactory(fail_call=2)
    with pytest.raises(RuntimeError, match="simulated"):
        run(config, tmp_path, lock, failure)
    committed = list((tmp_path / "rollout_parts" / "train").glob("*.json"))
    assert len(committed) == 2
    committed_seeds = {read_json(path)["seed"] for path in committed}
    resumed = FakeFactory()
    run(config, tmp_path, lock, resumed)
    assert committed_seeds.isdisjoint(seed for call in resumed.calls for seed in call)
    assert sum(map(len, resumed.calls)) == 4


def test_training_cache_survives_heldout_failure(tmp_path):
    config, lock = fixture_config(tmp_path)
    failure = FakeFactory(fail_call=3)
    with pytest.raises(RuntimeError, match="simulated"):
        run(config, tmp_path, lock, failure)
    train_cache = tmp_path / "train_rollouts.jsonl.zst"
    before = file_hash(train_cache)
    resumed = FakeFactory()
    run(config, tmp_path, lock, resumed)
    assert sum(map(len, resumed.calls)) == 3
    assert file_hash(train_cache) == before


@pytest.mark.parametrize("failure_call", [None, 2, 3])
def test_serial_engine_shuts_down_on_success_and_either_split_failure(tmp_path, failure_call):
    config, lock = fixture_config(tmp_path)
    factory = FakeFactory(fail_call=failure_call)
    shutdowns = []
    factory.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: shutdowns.append(True)))
    if failure_call is None:
        run(config, tmp_path, lock, factory)
    else:
        with pytest.raises(RuntimeError, match="simulated"):
            run(config, tmp_path, lock, factory)
    assert shutdowns == [True]


@pytest.mark.parametrize("change", ["seed", "cap", "revision", "top_p"])
def test_changed_generation_contract_cannot_reuse_immutable_caches(tmp_path, change):
    config, lock = fixture_config(tmp_path)
    run(config, tmp_path, lock, FakeFactory())
    config = copy.deepcopy(config)
    if change == "seed":
        config["seed"] += 1
    elif change == "cap":
        config["sampling"]["max_new_tokens"] += 1
    elif change == "revision":
        lock["model_revision"] = "another-immutable-revision"
    else:
        config["sampling"]["top_p"] = 0.9
    with pytest.raises((CacheConflict, ValueError)):
        run(config, tmp_path, lock, FakeFactory())


def test_invalid_empty_request_does_not_commit_but_preserves_prior(tmp_path):
    config, lock = fixture_config(tmp_path)
    _, seed = request_identity(
        42, "round1", "train", "train-hash", 1, prompt_identity={"row_id": "train-row", "prompt_index": 0}
    )
    with pytest.raises(ValueError, match="EMPTY_GENERATED_RESPONSE"):
        run(config, tmp_path, lock, FakeFactory(empty_seed=seed))
    assert len(list((tmp_path / "rollout_parts" / "train").glob("*.json"))) == 1
    resumed = FakeFactory()
    run(config, tmp_path, lock, resumed)
    assert sum(map(len, resumed.calls)) == 5


def test_corrupted_final_cache_is_rejected(tmp_path):
    config, lock = fixture_config(tmp_path)
    run(config, tmp_path, lock, FakeFactory())
    cache = tmp_path / "train_rollouts.jsonl.zst"
    with cache.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(CacheConflict, match="changed"):
        run(config, tmp_path, lock, FakeFactory())


def test_seed_derivation_is_order_independent():
    expected = request_identity(5, "round1", "train", "hash", 3)
    request_identity(5, "round1", "heldout", "other", 10)
    assert request_identity(5, "round1", "train", "hash", 3) == expected
    assert expected != request_identity(5, "round1", "train", "hash", 4)


def test_duplicate_prompt_text_in_distinct_rows_keeps_independent_draws(tmp_path):
    config, lock = fixture_config(tmp_path)
    config["train"]["n_problem"] = 2
    write_records(
        tmp_path / "train_prompt_manifest.parquet",
        [
            {
                "prompt_index": index,
                "row_id": f"train-row-{index}",
                "prompt_hash": "same-text-and-tokens",
                "prompt_token_ids": [1, 2],
                "prompt_token_count": 2,
            }
            for index in range(2)
        ],
    )
    run(config, tmp_path, lock, FakeFactory())
    records = list(read_compressed_records(tmp_path / "train_rollouts.jsonl.zst"))
    assert len(records) == 6
    assert len({row["request_id"] for row in records}) == 6
    assert len({row["seed"] for row in records}) == 6
    assert {row["prompt_index"] for row in records} == {0, 1}


@pytest.mark.parametrize("split", ["train", "heldout"])
@pytest.mark.parametrize("prompt_length,context", [(1024, 4096), (1025, 8192)])
def test_generation_enforces_prompt_limit_before_constructing_engine(tmp_path, split, prompt_length, context):
    config, lock = fixture_config(tmp_path)
    config["prompt"] = {"max_tokens": 1024}
    config["model"]["max_model_len"] = context
    config["sampling"]["max_new_tokens"] = 3072
    tokens = [7] * prompt_length
    write_records(
        tmp_path / f"{split}_prompt_manifest.parquet",
        [
            {
                "prompt_index": 0,
                "row_id": f"{split}-row",
                "prompt_hash": f"{split}-hash",
                "prompt_token_ids": tokens,
                "prompt_token_count": prompt_length,
            }
        ],
    )
    factory = FakeFactory()
    if prompt_length > 1024:
        # The prompt still fits the model context; its separate cap must reject it.
        assert prompt_length + 3072 < context
        with pytest.raises(ConfigurationError, match="prompt.max_tokens"):
            run(config, tmp_path, lock, factory)
        assert factory.instances == 0
        assert factory.calls == []
        assert not (tmp_path / f"{split}_rollouts.jsonl.zst").exists()
    else:
        run(config, tmp_path, lock, factory)
        assert factory.instances == 1
        records = list(read_compressed_records(tmp_path / f"{split}_rollouts.jsonl.zst"))
        assert all(record["prompt_token_ids"] == tokens for record in records)


def test_changed_prompt_cap_cannot_reuse_generation_cache_even_when_prompts_fit(tmp_path):
    config, lock = fixture_config(tmp_path)
    config["prompt"] = {"max_tokens": 1024}
    config["model"]["max_model_len"] = 4096
    config["sampling"]["max_new_tokens"] = 3072
    run(config, tmp_path, lock, FakeFactory())
    identity = read_json(tmp_path / "generation_manifest.json")["identity"]
    assert identity["prompt_max_tokens"] == 1024
    original_hash = file_hash(tmp_path / "train_rollouts.jsonl.zst")
    config["prompt"]["max_tokens"] = 1023
    factory = FakeFactory()
    with pytest.raises(CacheConflict):
        run(config, tmp_path, lock, factory)
    assert factory.instances == 0
    assert file_hash(tmp_path / "train_rollouts.jsonl.zst") == original_hash


class StreamingFactory(FakeFactory):
    """CPU engine with a long first response and out-of-order short responses."""

    def __init__(self, *, fail_step=None, invalid_output=None, on_add=None):
        super().__init__()
        self.llm_engine = self
        self.engine_core = SimpleNamespace(shutdown=self.shutdown)
        self.fail_step = fail_step
        self.invalid_output = invalid_output
        self.on_add = on_add
        self.active = {}
        self.submitted = []
        self.finished = []
        self.step_count = 0
        self.high_watermark = 0
        self.aborted = []
        self.shutdown_count = 0
        self.events = []

    def generate(self, *args, **kwargs):
        raise AssertionError("Continuous dispatch must not fall back to blocking generate()")

    def add_request(self, request_id, prompt, params):
        assert request_id not in self.active
        assert params.n == 1
        assert getattr(params.output_kind, "value", params.output_kind) == 2
        if self.on_add:
            self.on_add(self, request_id)
        self.events.append(("add", request_id))
        self.submitted.append((request_id, params.seed, copy.deepcopy(prompt)))
        self.active[request_id] = {"prompt": prompt, "remaining": 7 if len(self.submitted) == 1 else 1}
        self.high_watermark = max(self.high_watermark, len(self.active))

    def has_unfinished_requests(self):
        return bool(self.active)

    def step(self):
        self.step_count += 1
        self.events.append(("step", tuple(self.active)))
        if self.step_count == self.fail_step:
            raise RuntimeError("simulated continuous engine failure")
        if self.step_count == 1:
            # A valid step can produce no final output while requests decode.
            return []
        finished_ids = []
        for request_id, state in self.active.items():
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                finished_ids.append(request_id)
        outputs = []
        for request_id in reversed(finished_ids):
            state = self.active.pop(request_id)
            outputs.append(
                SimpleNamespace(
                    request_id=request_id,
                    prompt_token_ids=state["prompt"]["prompt_token_ids"],
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            token_ids=[7, 0],
                            text="answer<EOS>",
                            finish_reason="stop",
                            stop_reason=0,
                            logprobs=[{7: SimpleNamespace(logprob=-0.3)}, {0: SimpleNamespace(logprob=-0.3)}],
                            cumulative_logprob=-0.6,
                        )
                    ],
                )
            )
            self.finished.append(request_id)
            self.events.append(("finish", request_id))
        if outputs and self.invalid_output:
            invalid = self.invalid_output
            self.invalid_output = None
            if invalid == "unknown":
                outputs[0].request_id = "unknown-request-id"
            elif invalid == "duplicate":
                outputs.append(outputs[0])
            elif invalid == "missing":
                self.active.clear()
                return []
        return outputs

    def abort_request(self, request_ids):
        assert isinstance(request_ids, list)
        self.aborted.extend(request_ids)
        for request_id in request_ids:
            self.active.pop(request_id, None)

    def shutdown(self):
        self.shutdown_count += 1


def continuous_fixture(tmp_path):
    config, lock = fixture_config(tmp_path)
    config["generation"].update(dispatch_mode="continuous", max_in_flight=2)
    return config, lock


def test_continuous_refills_before_long_response_finishes_and_commits_immediately(tmp_path):
    config, lock = continuous_fixture(tmp_path)
    observed_commits = []

    def check_committed_before_refill(factory, request_id):
        if len(factory.submitted) == 2:
            first, second = [entry[0] for entry in factory.submitted]
            assert first in factory.active
            assert second in factory.finished
            part = tmp_path / "rollout_parts" / "train" / f"{second}.json"
            assert read_json(part)["request_id"] == second
            observed_commits.append(second)

    factory = StreamingFactory(on_add=check_committed_before_refill)
    run(config, tmp_path, lock, factory)
    assert factory.instances == 1
    assert factory.high_watermark == 2
    assert len(observed_commits) == 1
    assert len(factory.submitted) == 6
    assert len({entry[1] for entry in factory.submitted}) == 6
    assert factory.finished[0] == factory.submitted[1][0]
    assert factory.shutdown_count == 1
    for split in ("train", "heldout"):
        records = list(read_compressed_records(tmp_path / f"{split}_rollouts.jsonl.zst"))
        assert [record["rollout_index"] for record in records] == [0, 1, 2]
        assert all(record["response_token_ids"] == [7, 0] for record in records)
        assert all(record["engine_request_id"] == record["request_id"] for record in records)
    reused = StreamingFactory()
    run(config, tmp_path, lock, reused)
    assert reused.instances == 0


def test_continuous_failure_preserves_finished_request_and_resume_skips_it(tmp_path):
    config, lock = continuous_fixture(tmp_path)
    factory = StreamingFactory(fail_step=3)
    with pytest.raises(RuntimeError, match="simulated continuous"):
        run(config, tmp_path, lock, factory)
    parts = list((tmp_path / "rollout_parts" / "train").glob("*.json"))
    assert len(parts) == 1
    committed = read_json(parts[0])
    before = file_hash(parts[0])
    assert factory.aborted
    assert committed["request_id"] not in factory.aborted
    assert factory.shutdown_count == 1
    assert not (tmp_path / "train_rollouts.jsonl.zst").exists()
    resumed = StreamingFactory()
    run(config, tmp_path, lock, resumed)
    assert len(resumed.submitted) == 5
    assert committed["request_id"] not in {entry[0] for entry in resumed.submitted}
    assert committed["seed"] not in {entry[1] for entry in resumed.submitted}
    assert file_hash(parts[0]) == before


@pytest.mark.parametrize("invalid_output", ["unknown", "duplicate", "missing"])
def test_continuous_rejects_unknown_duplicate_or_missing_final_requests(tmp_path, invalid_output):
    config, lock = continuous_fixture(tmp_path)
    factory = StreamingFactory(invalid_output=invalid_output)
    with pytest.raises((ValueError, RuntimeError, CacheConflict)):
        run(config, tmp_path, lock, factory)
    assert not (tmp_path / "train_rollouts.jsonl.zst").exists()
    assert not (tmp_path / "generation_manifest.json").exists()
    assert factory.shutdown_count == 1


def test_optional_scheduler_controls_reach_engine_and_are_part_of_cache_identity(tmp_path):
    config, lock = fixture_config(tmp_path)
    config["generation"].update(max_num_seqs=64, max_num_batched_tokens=8192, enable_chunked_prefill=True)
    factory = FakeFactory()
    run(config, tmp_path, lock, factory)
    assert factory.kwargs["max_num_seqs"] == 64
    assert factory.kwargs["max_num_batched_tokens"] == 8192
    assert factory.kwargs["enable_chunked_prefill"] is True
    config["generation"]["max_num_seqs"] = 32
    reused = FakeFactory()
    with pytest.raises(CacheConflict):
        run(config, tmp_path, lock, reused)
    assert reused.instances == 0


def test_changing_dispatch_mode_cannot_reuse_old_sampling_execution_cache(tmp_path):
    config, lock = fixture_config(tmp_path)
    run(config, tmp_path, lock, FakeFactory())
    config["generation"].update(dispatch_mode="continuous", max_in_flight=2)
    factory = StreamingFactory()
    with pytest.raises(CacheConflict):
        run(config, tmp_path, lock, factory)
    assert factory.instances == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_in_flight", 0),
        ("max_in_flight", True),
        ("dispatch_mode", "unsupported"),
        ("max_num_seqs", -1),
        ("max_num_batched_tokens", -8192),
        ("enable_chunked_prefill", "true"),
    ],
)
def test_invalid_scheduler_settings_fail_before_any_engine_or_cache_is_created(tmp_path, field, value):
    config, lock = continuous_fixture(tmp_path)
    config["generation"][field] = value
    factory = StreamingFactory()
    with pytest.raises(ConfigurationError, match=field):
        run(config, tmp_path, lock, factory)
    assert factory.instances == 0
    assert factory.submitted == []
    assert not (tmp_path / "rollout_parts").exists()


def test_prompt_shards_keep_all_draws_together_and_finalize_only_after_every_part(tmp_path):
    from diagnostic_b.generation import _finalize_split, _generate_split, _generation_contract, _LazyEngine

    config, lock = continuous_fixture(tmp_path)
    config["n_rollout"] = 32
    config["train"]["n_problem"] = 2
    write_records(
        tmp_path / "train_prompt_manifest.parquet",
        [
            {
                "prompt_index": index,
                "row_id": f"train-row-{index}",
                "prompt_hash": f"train-hash-{index}",
                "prompt_token_ids": [1, index + 2],
                "prompt_token_count": 2,
            }
            for index in range(2)
        ],
    )
    contract = _generation_contract(config, tmp_path, lock, injected=True)
    first = StreamingFactory()
    first_engine = _LazyEngine(contract["engine_kwargs"], first, SimpleNamespace)
    try:
        _generate_split(config, tmp_path, "train", contract, first_engine, shard_index=0, shard_count=2)
    finally:
        first_engine.close()
    assert len(first.submitted) == 32
    parts = list((tmp_path / "rollout_parts" / "train").glob("*.json"))
    assert {read_json(path)["prompt_index"] for path in parts} == {0}
    checksums = {path: file_hash(path) for path in parts}
    assert not (tmp_path / "train_rollouts.jsonl.zst").exists()
    with pytest.raises((ValueError, CacheConflict, FileNotFoundError)):
        _finalize_split(tmp_path, "train", contract["requests"]["train"], contract["fingerprint"])
    assert not (tmp_path / "train_rollouts.jsonl.zst").exists()
    resumed = StreamingFactory()
    resumed_engine = _LazyEngine(contract["engine_kwargs"], resumed, SimpleNamespace)
    try:
        _generate_split(config, tmp_path, "train", contract, resumed_engine, shard_index=0, shard_count=2)
        assert resumed.instances == 0
        _generate_split(config, tmp_path, "train", contract, resumed_engine, shard_index=1, shard_count=2)
    finally:
        resumed_engine.close()
    assert len(resumed.submitted) == 32
    assert set(entry[0] for entry in first.submitted).isdisjoint(entry[0] for entry in resumed.submitted)
    assert all(file_hash(path) == checksums[path] for path in parts)
    _finalize_split(tmp_path, "train", contract["requests"]["train"], contract["fingerprint"])
    records = list(read_compressed_records(tmp_path / "train_rollouts.jsonl.zst"))
    assert [(record["prompt_index"], record["rollout_index"]) for record in records] == [
        (prompt_index, rollout_index) for prompt_index in range(2) for rollout_index in range(32)
    ]
