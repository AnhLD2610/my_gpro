# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Metadata-only model pinning and geometry-specific resource estimates.

No model, tokenizer, dataset, PyTorch, or CUDA library is loaded here. Remote
preflight downloads config.json only. Model weights are only content-hashed
when the author explicitly supplies a local checkpoint.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import ConfigurationError, feature_geometry, validate_config
from .storage import CacheConflict, atomic_json, digest, file_hash, read_json

LOG = logging.getLogger(__name__)
GIB = 1024**3


def _model_dimensions(model_config):
    names = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads")
    values = {}
    for name in names:
        value = model_config.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"Model config requires a positive integer {name}")
        values[name] = value
    values["num_key_value_heads"] = model_config.get("num_key_value_heads", values["num_attention_heads"])
    values["head_dim"] = model_config.get("head_dim", values["hidden_size"] // values["num_attention_heads"])
    if min(values["num_key_value_heads"], values["head_dim"]) <= 0:
        raise ConfigurationError("Invalid attention dimensions in model config")
    return values


def estimate_parameter_count(model_config):
    """Qwen3 dense transformer count, including q/k norms and optional biases."""
    dims = _model_dimensions(model_config)
    d, v, mlp, layers = (dims[key] for key in ("hidden_size", "vocab_size", "intermediate_size", "num_hidden_layers"))
    qwidth = dims["num_attention_heads"] * dims["head_dim"]
    kvwidth = dims["num_key_value_heads"] * dims["head_dim"]
    attention = 2 * d * qwidth + 2 * d * kvwidth
    feedforward = 3 * d * mlp
    normalization = 2 * d + 2 * dims["head_dim"]
    biases = (qwidth + 2 * kvwidth + d) if model_config.get("attention_bias", False) else 0
    if model_config.get("mlp_bias", False):
        biases += 2 * mlp + d
    embeddings = v * d * (1 if model_config.get("tie_word_embeddings", False) else 2)
    return int(embeddings + layers * (attention + feedforward + normalization + biases) + d)


def detect_hardware(output_dir):
    """Inspect visible devices through nvidia-smi; do not initialize CUDA."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError(f"GPU_PREFLIGHT_UNAVAILABLE: nvidia-smi failed: {exc}") from exc
    devices = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, total, free = [part.strip() for part in line.split(",")]
        devices.append(
            {"index": int(index), "uuid": uuid, "total_bytes": int(total) * 1024**2, "free_bytes": int(free) * 1024**2}
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        requested = [item.strip() for item in visible.split(",") if item.strip()]
        selected = []
        for identifier in requested:
            matches = [
                device
                for device in devices
                if str(device["index"]) == identifier or device["uuid"].startswith(identifier)
            ]
            if len(matches) != 1:
                raise ConfigurationError(
                    f"Cannot map CUDA_VISIBLE_DEVICES entry {identifier!r}; specify visible GPU UUIDs"
                )
            selected.append(matches[0])
        devices = selected
    if not devices:
        raise ConfigurationError("GPU_PREFLIGHT_UNAVAILABLE: no visible GPU")
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        meminfo[name] = int(value.strip().split()[0]) * 1024
    return {
        "gpu_devices": devices,
        "cpu_available_bytes": meminfo.get("MemAvailable", meminfo["MemFree"]),
        "cpu_total_bytes": meminfo["MemTotal"],
        "disk_free_bytes": shutil.disk_usage(output_dir).free,
        "cuda_visible_devices": visible,
    }


def estimate_resources(config, model_config, hardware):
    """Pure conservative sizing; token volumes are caps, not runtime predictions.

    The per-response segment budget sizes the preflight only. Extraction must
    stop and reevaluate actual group atoms if it exceeds this declaration.
    Segments, prompts, or responses are never truncated to make this estimate fit.
    """
    dims = _model_dimensions(model_config)
    v, d = dims["vocab_size"], dims["hidden_size"]
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[config["model"]["dtype"]]
    context = config["model"]["max_model_len"]
    new_tokens = config["sampling"]["max_new_tokens"]
    draws = config["n_rollout"]
    training_draws = config["train"]["n_problem"] * draws
    heldout_draws = config["heldout"]["n_problem"] * draws
    total_draws = training_draws + heldout_draws
    features = config["features"]
    token_chunk = features["token_chunk_size"]
    vocab_chunk = min(v, features["vocab_chunk_size"])
    logit_vocab_chunk = min(v, features.get("logit_vocab_chunk_size", 8192))
    segment_budget = features["max_segments_per_response_budget"]
    if min(token_chunk, vocab_chunk, logit_vocab_chunk, segment_budget) <= 0:
        raise ConfigurationError("Feature chunk sizes and segment budget must be positive")
    max_atoms = draws * max(1, segment_budget)
    parameters = estimate_parameter_count(model_config)
    weights = parameters * dtype_bytes
    delta_proxy = features["backend"] == "delta_proxy"
    output_head = d * 4 if delta_proxy else v * d * 4
    atom_tile = max_atoms * d * 4 if delta_proxy else max_atoms * vocab_chunk * d * 4
    gram = max_atoms * max_atoms * 4
    # hidden FP32 + original IDs int64 + log normalizer/token logprob FP32,
    # plus a four-byte per-token metadata allowance.
    factor_bytes_per_token = d * 4 + 20
    factor_tokens = total_draws * new_tokens
    factor_cache = factor_tokens * factor_bytes_per_token
    qwidth = dims["num_attention_heads"] * dims["head_dim"]
    kvwidth = dims["num_key_value_heads"] * dims["head_dim"]
    kv_single_context = 2 * dims["num_hidden_layers"] * context * kvwidth * dtype_bytes
    # Frozen backbone inference with SDPA and use_cache=False; no layerwise
    # activation retention. This covers live Q/K/V, MLP, norms and residuals.
    backbone_workspace = context * (8 * d + 3 * dims["intermediate_size"] + 3 * qwidth + 2 * kvwidth) * dtype_bytes
    response_hidden_copy = new_tokens * d * 4
    token_workspace = 4 * (4 * token_chunk * d + 4 * token_chunk * vocab_chunk + 3 * vocab_chunk * d)
    if delta_proxy:
        token_workspace = 4 * (4 * token_chunk * d + 4 * token_chunk)
    normalizer_gpu_workspace = 4 * (
        2 * logit_vocab_chunk * d + 3 * token_chunk * logit_vocab_chunk + 3 * token_chunk * d
    )
    normalizer_cpu_workspace = 4 * (2 * logit_vocab_chunk * d + token_chunk * d)
    runtime_reserve = 2 * GIB
    feature_extract_peak = (
        weights + backbone_workspace + response_hidden_copy + normalizer_gpu_workspace + runtime_reserve
    )
    feature_gram_peak = weights + atom_tile + 2 * gram + token_workspace + runtime_reserve
    feature_gpu_peak = max(feature_extract_peak, feature_gram_peak)
    cpu_feature_peak = (
        max(2 * atom_tile + 4 * gram + output_head, 2 * response_hidden_copy + weights) + normalizer_cpu_workspace + GIB
    )
    # JSON/decoded-text allowances are deliberately explicit. Compression gains
    # are not assumed, and duplicate partial/final rollout caches are budgeted.
    rollout_cache = total_draws * (new_tokens * 128 + context * 16 + 4096)
    training_gram_cache = config["train"]["n_problem"] * gram * 2
    disk_required = factor_cache + 2 * output_head + 2 * rollout_cache + training_gram_cache + 2 * weights
    devices = hardware.get("gpu_devices", [])
    tp = config["generation"]["tensor_parallel_size"]
    if not isinstance(tp, int) or tp <= 0 or len(devices) < tp:
        raise ConfigurationError(f"tensor_parallel_size={tp} requires that many visible GPUs; detected={len(devices)}")
    utilization = config["generation"]["gpu_memory_utilization"]
    safety = config["memory"].get("safety_fraction", 0.8)
    if not 0 < utilization < 1 or not 0 < safety <= 1:
        raise ConfigurationError("gpu_memory_utilization must be in (0,1), memory.safety_fraction in (0,1]")
    declared = config["memory"]

    def budget(name, observed):
        override = declared.get(name)
        if override is not None and override <= 0:
            raise ConfigurationError(f"memory.{name} must be positive")
        return min(observed, override * GIB) if override is not None else observed

    gpu_free = [budget("gpu_budget_gib", device["free_bytes"]) for device in devices[:tp]]
    cpu_available = budget("cpu_budget_gib", hardware["cpu_available_bytes"])
    disk_available = budget("disk_budget_gib", hardware["disk_free_bytes"])
    generation_allocations = [utilization * device["total_bytes"] for device in devices[:tp]]
    # At least one maximum-context request must fit. vLLM schedules additional
    # requests within its allocated KV pool; request_chunk_size is not residency.
    generation_minimum = weights / tp + kv_single_context / tp + runtime_reserve
    violations = []
    if feature_gpu_peak > safety * gpu_free[0]:
        violations.append("configured feature peak exceeds first visible GPU's safe available budget")
    if cpu_feature_peak > safety * cpu_available:
        violations.append("configured atom/Gram feature peak exceeds safe CPU RAM budget")
    if disk_required > safety * disk_available:
        violations.append("conservative factor/rollout cache estimate exceeds safe artifact-disk budget")
    for index, (allocation, available) in enumerate(zip(generation_allocations, gpu_free, strict=True)):
        if allocation > available:
            violations.append(f"vLLM GPU {index} allocation exceeds available/declared memory; lower utilization")
        if generation_minimum > allocation:
            violations.append(f"vLLM GPU {index} allocation cannot fit weights plus one full-context KV cache")
    byte_estimates = {
        "model_weights": weights,
        "heldout_head": output_head,
        "per_group_atom_tile": atom_tile,
        "per_group_gram": gram,
        "factor_cache": factor_cache,
        "rollout_cache_allowance": rollout_cache,
        "training_gram_cache": training_gram_cache,
        "single_context_kv": kv_single_context,
        "backbone_workspace": backbone_workspace,
        "normalizer_gpu_workspace": normalizer_gpu_workspace,
        "normalizer_cpu_workspace": normalizer_cpu_workspace,
        "feature_extract_gpu_peak": feature_extract_peak,
        "feature_gram_gpu_peak": feature_gram_peak,
        "feature_gpu_peak": feature_gpu_peak,
        "feature_cpu_peak": cpu_feature_peak,
        "disk_required": disk_required,
        "generation_minimum_per_gpu": generation_minimum,
    }
    return {
        "geometry": feature_geometry(config),
        "feature_shape": [d] if delta_proxy else [v, d],
        "backend": features["backend"],
        "estimate_kind": "conservative sizing, not guaranteed peaks or predicted runtimes",
        "vocab_size": v,
        "hidden_size": d,
        "parameter_estimate": parameters,
        "max_group_atoms": max_atoms,
        "max_segments_per_response_budget": segment_budget,
        "logit_vocab_chunk_size": logit_vocab_chunk,
        "train_draws": training_draws,
        "heldout_draws": heldout_draws,
        "total_draws": total_draws,
        "training_token_upper_bound": training_draws * new_tokens,
        "heldout_token_upper_bound": heldout_draws * new_tokens,
        "total_token_upper_bound": factor_tokens,
        "factor_bytes_per_token": factor_bytes_per_token,
        "bytes": byte_estimates,
        "gib": {key: value / GIB for key, value in byte_estimates.items()},
        "budgets": {
            "feature_gpu_safe_bytes": safety * gpu_free[0],
            "cpu_safe_bytes": safety * cpu_available,
            "disk_safe_bytes": safety * disk_available,
            "generation_available_bytes": gpu_free,
            "generation_allocations_bytes": generation_allocations,
            "declared": declared,
        },
        "safety_fraction": safety,
        "violations": violations,
        "fits": not violations,
        "assumptions": [
            "Every draw reaches max_new_tokens; held-out zero-reward factors may be omitted only after verification.",
            "Failed-response segments are never capped; actual atom counts require a repeated preflight check.",
            "Feature model occupies the first visible GPU in full, irrespective of generation tensor parallelism.",
            "Generation and feature models are loaded in separate processes and do not coexist.",
            "SDPA inference avoids an explicit context-squared attention matrix; no eager attention fallback.",
            "Two GiB GPU runtime reserve plus memory.safety_fraction cover allocator/workspace uncertainty.",
            "Disk figures reserve uncompressed text/JSON allowances; actual cache growth must be monitored.",
        ],
    }


def _local_files_snapshot(path):
    files = sorted(
        candidate
        for candidate in Path(path).rglob("*")
        if candidate.is_file()
        and (candidate.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"})
    )
    return {
        str(candidate.relative_to(path)): {"size": candidate.stat().st_size, "mtime_ns": candidate.stat().st_mtime_ns}
        for candidate in files
    }


def _lock_model(config, output_dir):
    model = config["model"]
    lock_path = Path(output_dir) / "model_lock.json"
    request = {
        "id": model["id"],
        "requested_revision": model.get("revision", "main"),
        "requested_tokenizer_revision": model.get("tokenizer_revision"),
        "local_path": model.get("local_path"),
    }
    if lock_path.exists():
        locked = read_json(lock_path)
        if any(locked.get(key) != value for key, value in request.items()):
            raise CacheConflict("MODEL_LOCK_MISMATCH: use a new output directory for different model/tokenizer inputs")
        if request["local_path"] and _local_files_snapshot(request["local_path"]) != locked.get("local_file_snapshot"):
            raise CacheConflict(
                "LOCAL_MODEL_CHANGED: files differ from locked content snapshot; use a new output directory"
            )
        return locked
    local_path = request["local_path"]
    additional = {}
    if local_path:
        source = Path(local_path)
        if not (source / "config.json").is_file():
            raise ConfigurationError("Local model requires config.json")
        snapshot = _local_files_snapshot(source)
        if not any(name.endswith((".safetensors", ".bin")) for name in snapshot):
            raise ConfigurationError("Local model snapshot contains no weight files")
        if not any("tokenizer" in name for name in snapshot):
            raise ConfigurationError("Local model snapshot requires tokenizer files")
        checksums = {name: file_hash(source / name) for name in snapshot}
        revision = "local-sha256:" + digest(checksums)
        tokenizer_revision = revision
        model_config = read_json(source / "config.json")
        additional = {"local_file_snapshot": snapshot, "local_file_sha256": checksums}
    else:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        revision = api.model_info(model["id"], revision=request["requested_revision"]).sha
        tokenizer_request = request["requested_tokenizer_revision"]
        tokenizer_revision = (
            api.model_info(model["id"], revision=tokenizer_request).sha if tokenizer_request else revision
        )
        if not revision or not tokenizer_revision:
            raise ConfigurationError("Unable to resolve model/tokenizer to immutable commit revisions")
        config_path = hf_hub_download(repo_id=model["id"], filename="config.json", revision=revision)
        model_config = read_json(config_path)
        additional = {"model_config_sha256": file_hash(config_path)}
    if model_config.get("model_type") != "qwen3":
        raise ConfigurationError("Locked checkpoint config must declare model_type=qwen3")
    if model_config.get("max_position_embeddings", 0) < model["max_model_len"]:
        raise ConfigurationError("Model config cannot support the declared 32768 context")
    dims = _model_dimensions(model_config)
    locked = (
        request
        | {
            "revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "model_config": model_config,
            "vocab_size": dims["vocab_size"],
            "hidden_size": dims["hidden_size"],
            "parameter_estimate": estimate_parameter_count(model_config),
        }
        | additional
    )
    atomic_json(lock_path, locked, immutable=True)
    return locked


def run_preflight(config, output_dir, *, hardware=None, enable_round2=False):
    """Validate first, pin metadata, estimate exact geometry, write reviewable results."""
    validate_config(config, enable_round2=enable_round2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    locked = _lock_model(config, output_dir)
    hardware = detect_hardware(output_dir) if hardware is None else hardware
    estimates = estimate_resources(config, locked["model_config"], hardware)
    report = {"model_revision": locked["revision"], "hardware": hardware, "estimates": estimates}
    report_path = output_dir / "preflight.json"
    atomic_json(report_path, report)
    LOG.info(
        "Exact-head preflight model_lock=%s report=%s estimates=%s",
        output_dir / "model_lock.json",
        report_path,
        estimates,
    )
    if not estimates["fits"]:
        raise ConfigurationError(
            "EXACT_HEAD_RESOURCE_BUDGET_EXCEEDED: "
            + "; ".join(estimates["violations"])
            + f". Review {report_path}; generation has not started. "
            + "No proxy, token cap, or segment cap was substituted."
        )
    return [output_dir / "model_lock.json", report_path]
