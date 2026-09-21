# Copyright 2026 Nash Credit Routing contributors
# SPDX-License-Identifier: Apache-2.0
"""Metadata and resource arithmetic only; fake files are never loaded as models."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from diagnostic_b.config import ConfigurationError
from diagnostic_b.preflight import GIB, detect_hardware, estimate_parameter_count, estimate_resources, run_preflight
from diagnostic_b.storage import CacheConflict, read_json


@pytest.fixture
def model_config():
    return {
        "model_type": "qwen3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
    }


@pytest.fixture
def config(tmp_path, model_config):
    source = Path(__file__).resolve().parents[2] / "configs" / "diagnostic_b_round1.yaml"
    config = yaml.safe_load(source.read_text())
    config["model"].update(id="Qwen/Qwen3-1.7B-Base", max_model_len=32768)
    config["prompt"].pop("max_tokens", None)
    config["generation"]["tensor_parallel_size"] = 1  # Synthetic fixture has one GPU and two heads.
    config["generation"]["parallel_splits"] = False
    config["generation"]["replicas_per_split"] = 1
    config["features"].pop("factor_cache_retention", None)  # Legacy configs retain all factors.
    config["seed"] = 123  # Synthetic fixture, not an approved experiment setting.
    config["sampling"]["max_new_tokens"] = 64
    config["routing"]["radius_coefficient"] = 0.5
    config["verifier"]["backend"] = "math_verify"
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text('{"user": "{problem}"}')
    config["prompt"]["wrapper_file"] = str(wrapper)
    local = tmp_path / "local_model"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(model_config))
    (local / "tokenizer.json").write_text('{"fake": "never loaded"}')
    (local / "model.safetensors").write_bytes(b"fake CPU test bytes, never a model")
    config["model"]["local_path"] = str(local)
    return config


@pytest.fixture
def hardware():
    return {
        "gpu_devices": [{"index": 0, "uuid": "GPU-FAKE", "total_bytes": 16 * GIB, "free_bytes": 16 * GIB}],
        "cpu_available_bytes": 64 * GIB,
        "disk_free_bytes": 1024 * GIB,
    }


@pytest.fixture
def parallel_setup(config, model_config, hardware):
    config["generation"].update(tensor_parallel_size=4, parallel_splits=True)
    model_config.update(model_type="qwen2", num_attention_heads=28, num_key_value_heads=4, hidden_size=112)
    hardware["gpu_devices"] = [
        {"index": index, "uuid": f"GPU-FAKE-{index}", "total_bytes": 16 * GIB, "free_bytes": 16 * GIB}
        for index in range(8)
    ]
    return config, model_config, hardware


def test_parameter_count_includes_full_output_path_and_tying(model_config):
    assert estimate_parameter_count(model_config) == 1848
    untied = model_config | {"tie_word_embeddings": False}
    assert estimate_parameter_count(untied) - estimate_parameter_count(model_config) == 32 * 8


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_parameter_count_matches_small_transformers_model(model_config, model_type, tie_embeddings):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    config_class, model_class = (
        (Qwen2Config, Qwen2ForCausalLM) if model_type == "qwen2" else (Qwen3Config, Qwen3ForCausalLM)
    )
    model_config = model_config | {"model_type": model_type, "tie_word_embeddings": tie_embeddings}
    with torch.device("meta"):
        model = model_class(config_class(**model_config))
    assert estimate_parameter_count(model_config) == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("backend", ["delta_proxy", "exact_tiled_head"])
def test_resource_estimate_declares_all_draws_and_selected_geometry(config, model_config, hardware, backend):
    config["features"]["backend"] = backend
    result = estimate_resources(config, model_config, hardware)
    delta = backend == "delta_proxy"
    assert result["geometry"] == (
        "delta_selected_token_gradient_proxy" if delta else "unprojected_output_path_lm_head_proxy"
    )
    assert result["feature_shape"] == ([8] if delta else [32, 8])
    assert result["train_draws"] == 500 * 32
    assert result["heldout_draws"] == 500 * 32
    assert result["total_token_upper_bound"] == 1000 * 32 * 64
    assert result["bytes"]["heldout_head"] == (8 * 4 if delta else 32 * 8 * 4)
    assert result["bytes"]["per_group_atom_tile"] == result["max_group_atoms"] * 8 * 4 * (1 if delta else 32)
    assert result["bytes"]["factor_cache"] == result["total_token_upper_bound"] * (8 * 4 + 20)
    assert result["factor_cache_retention"] == "all"
    assert result["peak_factor_token_upper_bound"] == result["total_token_upper_bound"]
    assert result["max_group_atoms"] == 32 * 128
    assert result["fits"]


def test_group_retention_fits_math_run_without_reducing_samples_or_tokens(config, hardware):
    # Reproduce the original disk-budget failure independently of current defaults.
    config["train"]["n_problem"] = 128
    config["heldout"]["n_problem"] = 100
    config["sampling"]["max_new_tokens"] = 30720
    config["generation"]["tensor_parallel_size"] = 8
    qwen = {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": True,
    }
    hardware["gpu_devices"] = [
        {"index": index, "uuid": f"GPU-FAKE-{index}", "total_bytes": 96 * GIB, "free_bytes": 96 * GIB}
        for index in range(8)
    ]
    hardware["disk_free_bytes"] = 2058927964160  # Available space from the failed run.
    retained = estimate_resources(config, qwen, hardware)
    assert retained["violations"] == ["conservative factor/rollout cache estimate exceeds safe artifact-disk budget"]
    config["features"]["factor_cache_retention"] = "group"
    before = copy.deepcopy(config)
    grouped = estimate_resources(config, qwen, hardware)
    assert grouped["fits"]
    assert config == before
    assert grouped["factor_cache_retention"] == "group"
    assert grouped["peak_factor_token_upper_bound"] == 32 * 30720
    assert grouped["bytes"]["factor_cache"] == 32 * 30720 * (2048 * 4 + 20)
    assert grouped["bytes"]["disk_required"] < 100 * GIB
    for key in (
        "geometry",
        "feature_shape",
        "train_draws",
        "heldout_draws",
        "total_draws",
        "training_token_upper_bound",
        "heldout_token_upper_bound",
        "total_token_upper_bound",
    ):
        assert grouped[key] == retained[key]
    for key in ("feature_gpu_peak", "feature_cpu_peak", "rollout_cache_allowance", "training_gram_cache"):
        assert grouped["bytes"][key] == retained["bytes"][key]


@pytest.mark.parametrize("retention,backend", [("none", "delta_proxy"), ("group", "exact_tiled_head")])
def test_unsupported_factor_retention_cannot_lower_resource_estimate(
    config, model_config, hardware, retention, backend
):
    config["features"].update(factor_cache_retention=retention, backend=backend)
    with pytest.raises(ConfigurationError, match="factor_cache_retention"):
        estimate_resources(config, model_config, hardware)


def test_memory_rejection_is_visible_and_does_not_change_geometry(config, model_config, hardware):
    tiny = copy.deepcopy(hardware)
    tiny["disk_free_bytes"] = 1
    before = copy.deepcopy(config)
    result = estimate_resources(config, model_config, tiny)
    assert not result["fits"]
    assert any("disk" in item for item in result["violations"])
    assert config == before
    assert result["geometry"] == "delta_selected_token_gradient_proxy"


def test_delta_reduces_aggregate_memory_but_still_accounts_for_model_and_normalization(config, model_config, hardware):
    proxy = estimate_resources(config, model_config, hardware)
    config["features"]["backend"] = "exact_tiled_head"
    exact = estimate_resources(config, model_config, hardware)
    for key in ("heldout_head", "per_group_atom_tile", "disk_required"):
        assert proxy["bytes"][key] < exact["bytes"][key]
    for key in ("model_weights", "factor_cache", "normalizer_gpu_workspace", "backbone_workspace"):
        assert proxy["bytes"][key] == exact["bytes"][key]


def test_feature_model_is_not_sharded_by_generation_tensor_parallelism(config, model_config, hardware):
    first = estimate_resources(config, model_config, hardware)
    parallel = copy.deepcopy(config)
    parallel["generation"]["tensor_parallel_size"] = 2
    two_devices = copy.deepcopy(hardware)
    two_devices["gpu_devices"].append(two_devices["gpu_devices"][0] | {"index": 1, "uuid": "GPU-OTHER"})
    second = estimate_resources(parallel, model_config, two_devices)
    assert first["bytes"]["feature_gpu_peak"] == second["bytes"]["feature_gpu_peak"]
    assert second["bytes"]["generation_minimum_per_gpu"] < first["bytes"]["generation_minimum_per_gpu"]


def test_normalizer_chunk_workspace_accounted_separately(config, model_config, hardware):
    config["features"]["logit_vocab_chunk_size"] = 8
    result = estimate_resources(config, model_config, hardware)
    token_chunk, width, logit_rows = config["features"]["token_chunk_size"], 8, 8
    assert result["bytes"]["normalizer_gpu_workspace"] == 4 * (
        2 * logit_rows * width + 3 * token_chunk * logit_rows + 3 * token_chunk * width
    )
    assert result["bytes"]["normalizer_cpu_workspace"] == 4 * (2 * logit_rows * width + token_chunk * width)
    assert result["logit_vocab_chunk_size"] == 8


def test_visible_gpu_count_and_generation_allocation_checked(config, model_config, hardware):
    config["generation"]["tensor_parallel_size"] = 2
    with pytest.raises(ConfigurationError, match="visible GPUs"):
        estimate_resources(config, model_config, hardware)
    config["generation"]["tensor_parallel_size"] = 1
    hardware["gpu_devices"][0]["free_bytes"] = 8 * GIB
    result = estimate_resources(config, model_config, hardware)
    assert any("vLLM" in item and "exceeds" in item for item in result["violations"])


def test_qwen2_math_attention_heads_reject_tp8_but_allow_tp4(config, model_config, hardware):
    model_config = model_config | {
        "model_type": "qwen2",
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "hidden_size": 112,
    }
    hardware["gpu_devices"] = [
        {"index": index, "uuid": f"GPU-FAKE-{index}", "total_bytes": 16 * GIB, "free_bytes": 16 * GIB}
        for index in range(8)
    ]
    config["generation"]["tensor_parallel_size"] = 8
    with pytest.raises(ConfigurationError, match="tensor_parallel_size=8.*num_attention_heads=28"):
        estimate_resources(config, model_config, hardware)
    config["generation"]["tensor_parallel_size"] = 4
    assert estimate_resources(config, model_config, hardware)["fits"]


def test_tensor_parallelism_checks_kv_head_sharding_and_replication(config, model_config, hardware):
    hardware["gpu_devices"] = [
        {"index": index, "uuid": f"GPU-FAKE-{index}", "total_bytes": 16 * GIB, "free_bytes": 16 * GIB}
        for index in range(8)
    ]
    model_config = model_config | {"num_attention_heads": 28, "num_key_value_heads": 4}
    config["generation"]["tensor_parallel_size"] = 7
    with pytest.raises(ConfigurationError, match="tensor_parallel_size=7.*num_key_value_heads=4"):
        estimate_resources(config, model_config, hardware)
    model_config["num_attention_heads"] = 16
    config["generation"]["tensor_parallel_size"] = 8
    assert estimate_resources(config, model_config, hardware)["fits"]


def test_parallel_splits_require_eight_distinct_gpus_for_tp4(parallel_setup):
    config, model_config, hardware = parallel_setup
    result = estimate_resources(config, model_config, hardware)
    assert result["fits"]
    assert result["generation_parallel_splits"] is True
    assert result["generation_engine_count"] == 2
    assert result["generation_required_gpus"] == 8
    assert [device["index"] for device in result["generation_device_groups"]["train"]] == [0, 1, 2, 3]
    assert [device["index"] for device in result["generation_device_groups"]["heldout"]] == [4, 5, 6, 7]
    assert len(result["budgets"]["generation_allocations_bytes"]) == 8
    assert len(result["budgets"]["generation_available_bytes"]) == 8
    hardware["gpu_devices"] = hardware["gpu_devices"][:4]
    with pytest.raises(ConfigurationError, match="requires 8 distinct visible GPUs; detected=4"):
        estimate_resources(config, model_config, hardware)


def test_parallel_splits_check_memory_on_heldout_gpu_group(parallel_setup):
    config, model_config, hardware = parallel_setup
    hardware["gpu_devices"][7]["free_bytes"] = 8 * GIB
    result = estimate_resources(config, model_config, hardware)
    assert result["violations"] == ["vLLM GPU 7 allocation exceeds available/declared memory; lower utilization"]
    assert not result["fits"]


def test_parallel_cpu_budget_covers_two_model_copies_without_doubling_features(parallel_setup):
    config, model_config, hardware = parallel_setup
    concurrent = estimate_resources(config, model_config, hardware)
    config["generation"].pop("parallel_splits")  # Legacy absence means one shared serial engine.
    serial = estimate_resources(config, model_config, hardware)
    assert serial["generation_parallel_splits"] is False
    assert serial["generation_engine_count"] == 1
    assert serial["generation_required_gpus"] == 4
    assert serial["generation_device_groups"]["train"] == serial["generation_device_groups"]["heldout"]
    assert len(serial["budgets"]["generation_allocations_bytes"]) == 4
    assert concurrent["bytes"]["generation_cpu_peak"] == 2 * serial["bytes"]["generation_cpu_peak"]
    for key in ("feature_cpu_peak", "feature_gpu_peak", "factor_cache", "disk_required"):
        assert concurrent["bytes"][key] == serial["bytes"][key]
    assert serial["bytes"]["cpu_peak"] == serial["bytes"]["feature_cpu_peak"]
    assert concurrent["bytes"]["generation_cpu_peak"] > serial["bytes"]["feature_cpu_peak"]
    hardware["cpu_available_bytes"] = int(
        (serial["bytes"]["feature_cpu_peak"] + concurrent["bytes"]["generation_cpu_peak"])
        / (2 * config["memory"]["safety_fraction"])
    )
    assert estimate_resources(config, model_config, hardware)["fits"]
    config["generation"]["parallel_splits"] = True
    rejected = estimate_resources(config, model_config, hardware)
    assert rejected["violations"] == ["concurrent generation model copies exceed safe CPU RAM budget"]


def test_replica_preflight_counts_all_engines_cpu_copies_and_gpu_allocations(parallel_setup):
    config, model_config, hardware = parallel_setup
    config["generation"].update(tensor_parallel_size=1, replicas_per_split=4)
    replicated = estimate_resources(config, model_config, hardware)
    assert replicated["fits"]
    assert replicated["generation_replicas_per_split"] == 4
    assert replicated["generation_engine_count"] == 8
    assert replicated["generation_required_gpus"] == 8
    assert len(replicated["generation_device_groups"]) == 8
    assert [group[0]["index"] for group in replicated["generation_device_groups"].values()] == list(range(8))
    config["generation"].update(parallel_splits=False, replicas_per_split=1)
    serial = estimate_resources(config, model_config, hardware)
    assert replicated["bytes"]["generation_cpu_peak"] == 8 * serial["bytes"]["generation_cpu_peak"]
    assert replicated["bytes"]["feature_gpu_peak"] == serial["bytes"]["feature_gpu_peak"]
    config["generation"].update(parallel_splits=True, replicas_per_split=4)
    hardware["cpu_available_bytes"] = 5 * GIB
    hardware["gpu_devices"][7]["free_bytes"] = 8 * GIB
    result = estimate_resources(config, model_config, hardware)
    assert "concurrent generation model copies exceed safe CPU RAM budget" in result["violations"]
    assert "vLLM GPU 7 allocation exceeds available/declared memory; lower utilization" in result["violations"]
    hardware["gpu_devices"].pop()
    with pytest.raises(ConfigurationError, match="requires 8 distinct visible GPUs; detected=7"):
        estimate_resources(config, model_config, hardware)


@pytest.mark.parametrize("replicas", [True, 0, -1, 1.5])
def test_preflight_rejects_invalid_replica_count(config, model_config, hardware, replicas):
    config["generation"]["replicas_per_split"] = replicas
    with pytest.raises(ConfigurationError, match="replicas_per_split"):
        estimate_resources(config, model_config, hardware)


def test_preflight_rejects_replicas_in_serial_mode(config, model_config, hardware):
    config["generation"]["replicas_per_split"] = 2
    with pytest.raises(ConfigurationError, match="parallel_splits=true"):
        estimate_resources(config, model_config, hardware)


@pytest.mark.parametrize("field", ["index", "uuid"])
def test_parallel_device_groups_cannot_alias_same_gpu(parallel_setup, field):
    config, model_config, hardware = parallel_setup
    hardware["gpu_devices"][4][field] = hardware["gpu_devices"][0][field]
    with pytest.raises(ConfigurationError, match="duplicate GPU index or UUID"):
        estimate_resources(config, model_config, hardware)


@pytest.mark.parametrize("visible", ["0,0", "GPU-ONE,GPU-ONE", "0,GPU-ONE"])
def test_hardware_detection_rejects_duplicate_gpu_aliases(tmp_path, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        "diagnostic_b.preflight.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="0, GPU-ONE, 16384, 16384\n1, GPU-TWO, 16384, 16384\n"),
    )
    with pytest.raises(ConfigurationError, match="distinct visible GPUs"):
        detect_hardware(tmp_path)


def test_hardware_detection_preserves_requested_gpu_order(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-TWO,0")
    monkeypatch.setattr(
        "diagnostic_b.preflight.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="0, GPU-ONE, 16384, 16384\n1, GPU-TWO, 16384, 16384\n"),
    )
    assert [device["index"] for device in detect_hardware(tmp_path)["gpu_devices"]] == [1, 0]


def test_declared_budget_cannot_exceed_detected_capacity(config, model_config, hardware):
    config["memory"]["cpu_budget_gib"] = 1000
    result = estimate_resources(config, model_config, hardware)
    assert result["budgets"]["cpu_safe_bytes"] == hardware["cpu_available_bytes"] * 0.8


def test_local_metadata_lock_reused_without_weight_rehash(config, hardware, tmp_path, monkeypatch):
    output = tmp_path / "output"
    paths = run_preflight(config, output, hardware=hardware)
    assert {path.name for path in paths} == {"model_lock.json", "preflight.json"}
    locked = read_json(output / "model_lock.json")
    assert locked["revision"].startswith("local-sha256:")
    assert locked["tokenizer_revision"] == locked["revision"]
    assert "model.safetensors" in locked["local_file_sha256"]
    monkeypatch.setattr("diagnostic_b.preflight.file_hash", lambda *args: pytest.fail("unexpected file rehash"))
    run_preflight(config, output, hardware=hardware)
    assert read_json(output / "model_lock.json") == locked


def test_qwen2_math_model_lock_supports_native_4096_context(config, model_config, hardware, tmp_path):
    config["model"].update(id="Qwen/Qwen2.5-Math-7B", max_model_len=4096)
    config["prompt"]["max_tokens"] = 1024
    config["sampling"]["max_new_tokens"] = 3072
    metadata = model_config | {"model_type": "qwen2", "max_position_embeddings": 4096}
    (Path(config["model"]["local_path"]) / "config.json").write_text(json.dumps(metadata))
    output = tmp_path / "output"
    run_preflight(config, output, hardware=hardware)
    assert read_json(output / "model_lock.json")["model_config"]["model_type"] == "qwen2"
    config["model"]["max_model_len"] = 8192
    with pytest.raises(ConfigurationError, match="declared 8192 context"):
        run_preflight(config, output, hardware=hardware)


def test_model_lock_rejects_wrong_architecture_for_requested_checkpoint(config, hardware, tmp_path):
    config["model"].update(id="Qwen/Qwen2.5-Math-7B", max_model_len=4096)
    with pytest.raises(ConfigurationError, match="model_type=qwen2"):
        run_preflight(config, tmp_path / "output", hardware=hardware)


def test_model_lock_rejects_context_beyond_native_limit(config, model_config, hardware, tmp_path):
    config["model"].update(id="Qwen/Qwen2.5-Math-7B", max_model_len=8192)
    metadata = model_config | {"model_type": "qwen2", "max_position_embeddings": 4096}
    (Path(config["model"]["local_path"]) / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ConfigurationError, match="declared 8192 context"):
        run_preflight(config, tmp_path / "output", hardware=hardware)


def test_changed_local_model_refuses_cache_reuse(config, hardware, tmp_path):
    output = tmp_path / "output"
    run_preflight(config, output, hardware=hardware)
    (Path(config["model"]["local_path"]) / "model.safetensors").write_bytes(b"changed fake bytes")
    with pytest.raises(CacheConflict, match="LOCAL_MODEL_CHANGED"):
        run_preflight(config, output, hardware=hardware)


def test_bad_config_rejected_before_model_metadata_or_hardware(config, tmp_path, monkeypatch):
    config["sampling"]["max_new_tokens"] = None
    monkeypatch.setattr(
        "diagnostic_b.preflight._lock_model", lambda *args: pytest.fail("metadata accessed before validation")
    )
    monkeypatch.setattr(
        "diagnostic_b.preflight.detect_hardware", lambda *args: pytest.fail("GPU accessed before validation")
    )
    with pytest.raises(ConfigurationError, match="unresolved author setting"):
        run_preflight(config, tmp_path / "output")


@pytest.mark.parametrize("backend", ["delta_proxy", "exact_tiled_head"])
def test_resource_failure_preserves_report_before_generation(config, hardware, tmp_path, caplog, backend):
    config["features"]["backend"] = backend
    hardware["disk_free_bytes"] = 1
    output = tmp_path / "output"
    with caplog.at_level("INFO"), pytest.raises(ConfigurationError, match="FEATURE_RESOURCE_BUDGET_EXCEEDED") as error:
        run_preflight(config, output, hardware=hardware)
    assert "generation has not started" in str(error.value)
    assert f"Feature preflight backend={backend}" in caplog.text
    estimates = read_json(output / "preflight.json")["estimates"]
    assert not estimates["fits"]
    assert estimates["backend"] == backend
    assert (output / "model_lock.json").is_file()


def test_remote_preflight_downloads_config_only_and_reuses_locked_revision(config, hardware, tmp_path, monkeypatch):
    config["model"]["local_path"] = None
    source = tmp_path / "local_model" / "config.json"
    calls = []

    class MetadataAPI:
        def model_info(self, repo_id, revision):
            calls.append(("model_info", repo_id, revision))
            return types.SimpleNamespace(sha="a" * 40)

    def download(**kwargs):
        calls.append(("download", kwargs))
        assert kwargs["filename"] == "config.json"
        assert kwargs["revision"] == "a" * 40
        return str(source)

    fake_hub = types.SimpleNamespace(HfApi=MetadataAPI, hf_hub_download=download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    output = tmp_path / "output"
    run_preflight(config, output, hardware=hardware)
    count = len(calls)
    run_preflight(config, output, hardware=hardware)
    assert len(calls) == count == 2
    assert read_json(output / "model_lock.json")["revision"] == "a" * 40
