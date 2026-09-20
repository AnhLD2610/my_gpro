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
from diagnostic_b.preflight import GIB, estimate_parameter_count, estimate_resources, run_preflight
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
    config["seed"] = 123  # Synthetic fixture, not an approved experiment setting.
    config["sampling"]["max_new_tokens"] = 64
    config["routing"]["radius_coefficient"] = 0.5
    config["train"].update(id="synthetic/not-loaded", revision="b" * 40)
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


def test_parameter_count_includes_full_output_path_and_tying(model_config):
    assert estimate_parameter_count(model_config) == 1848
    untied = model_config | {"tie_word_embeddings": False}
    assert estimate_parameter_count(untied) - estimate_parameter_count(model_config) == 32 * 8


@pytest.mark.parametrize("backend", ["delta_proxy", "exact_tiled_head"])
def test_resource_estimate_declares_all_draws_and_selected_geometry(config, model_config, hardware, backend):
    config["features"]["backend"] = backend
    result = estimate_resources(config, model_config, hardware)
    delta = backend == "delta_proxy"
    assert result["geometry"] == (
        "delta_selected_token_gradient_proxy" if delta else "unprojected_output_path_lm_head_proxy"
    )
    assert result["feature_shape"] == ([8] if delta else [32, 8])
    assert result["train_draws"] == 64 * 32
    assert result["heldout_draws"] == 30 * 32
    assert result["total_token_upper_bound"] == (64 + 30) * 32 * 64
    assert result["bytes"]["heldout_head"] == (8 * 4 if delta else 32 * 8 * 4)
    assert result["bytes"]["per_group_atom_tile"] == result["max_group_atoms"] * 8 * 4 * (1 if delta else 32)
    assert result["bytes"]["factor_cache"] == result["total_token_upper_bound"] * (8 * 4 + 20)
    assert result["max_group_atoms"] == 32 * 128
    assert result["fits"]


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


def test_resource_failure_preserves_report_before_generation(config, hardware, tmp_path):
    hardware["disk_free_bytes"] = 1
    output = tmp_path / "output"
    with pytest.raises(ConfigurationError, match="generation has not started"):
        run_preflight(config, output, hardware=hardware)
    assert not read_json(output / "preflight.json")["estimates"]["fits"]
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
