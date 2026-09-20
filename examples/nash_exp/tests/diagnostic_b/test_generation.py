"""CPU fake-engine tests for immutable, shared and resumable generation."""

import copy
from types import SimpleNamespace

import pytest
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
