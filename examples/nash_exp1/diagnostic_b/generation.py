# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""One shared, full-support rollout cache with request-level recovery.

No model, dataset or generation library is loaded on import. A single vLLM
instance serves both datasets and all methods consume the resulting same files.
Completed request parts and final split caches are immutable.
"""

from __future__ import annotations

import logging
import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ConfigurationError
from .storage import (
    CacheConflict,
    atomic_json,
    digest,
    file_hash,
    read_compressed_records,
    read_json,
    read_records,
    write_compressed_records,
)

LOG = logging.getLogger(__name__)


def request_identity(master_seed, round_id, split, prompt_hash, rollout_index, prompt_identity=None):
    """SHA256-derived independent request seeds, bounded to signed 64-bit."""
    payload = {
        "master_seed": int(master_seed),
        "round_id": round_id,
        "split": split,
        "prompt_hash": prompt_hash,
        "rollout_index": int(rollout_index),
        "prompt_identity": prompt_identity,
        "seed_derivation_version": 2,
    }
    request_hash = digest(payload)
    seed = int(request_hash[:16], 16) % (2**63 - 1)
    return f"{round_id}-{split}-{request_hash[:24]}", seed


def _libraries():
    result = {}
    for name in ("vllm", "torch", "transformers", "tokenizers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _sampling(config):
    sampling = config["sampling"]
    if (sampling.get("temperature"), sampling.get("top_p"), sampling.get("top_k"), sampling.get("min_p")) != (
        1.0,
        1.0,
        -1,
        0.0,
    ):
        raise ConfigurationError("Diagnostic generation requires approved full-support temperature=1 sampling")
    cap = sampling.get("max_new_tokens")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ConfigurationError("Approved positive integer max_new_tokens is required")
    return {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": cap,
        "min_tokens": 0,
        "ignore_eos": False,
        "stop": None,
        "stop_token_ids": None,
        "include_stop_str_in_output": False,
        "logprobs": 0,
        "skip_special_tokens": False,
    }


def _prompt_contract(row):
    ids = row["prompt_token_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(token) for token in ids]
    if not ids or any(token < 0 for token in ids):
        raise ValueError("Prompt manifests must contain nonempty original token ID sequences")
    if int(row["prompt_token_count"]) != len(ids):
        raise CacheConflict("Prompt token count disagrees with original IDs")
    return {
        "prompt_index": int(row["prompt_index"]),
        "row_id": str(row["row_id"]),
        "prompt_hash": str(row["prompt_hash"]),
        "prompt_token_ids": ids,
    }


def _request_records(config, split, rows, sampling, fingerprint):
    for row in rows:
        prompt = _prompt_contract(row)
        for rollout_index in range(config["n_rollout"]):
            request_id, seed = request_identity(
                config["seed"],
                config["round_id"],
                split,
                prompt["prompt_hash"],
                rollout_index,
                prompt_identity={"row_id": prompt["row_id"], "prompt_index": prompt["prompt_index"]},
            )
            yield {
                **prompt,
                "split": split,
                "rollout_index": rollout_index,
                "request_id": request_id,
                "seed": seed,
                "sampling": sampling,
                "generation_fingerprint": fingerprint,
            }


def _selected_log_probs(completion, token_ids):
    values = getattr(completion, "logprobs", None)
    if values is None:
        return None
    if len(values) != len(token_ids):
        raise ValueError("Generation logprob count differs from sampled action count, including EOS")
    result = []
    for token_id, token_values in zip(token_ids, values, strict=True):
        if token_values is None or token_id not in token_values:
            raise ValueError("Generation did not return the selected action's requested log probability")
        entry = token_values[token_id]
        value = float(getattr(entry, "logprob", entry))
        if not math.isfinite(value):
            raise ValueError("Generation returned a nonfinite sampled-token log probability")
        result.append(value)
    return result


def _validate_record(record, request):
    checksum = record.get("record_checksum")
    if checksum != digest({key: value for key, value in record.items() if key != "record_checksum"}):
        raise CacheConflict(f"Corrupted completed request part: {request['request_id']}")
    if any(record.get(key) != value for key, value in request.items()):
        raise CacheConflict(f"Rollout request cache identity mismatch: {request['request_id']}")
    tokens = record.get("response_token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"EMPTY_GENERATED_RESPONSE: request={request['request_id']}")
    if any(not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError("Generated action IDs must be nonnegative integers")
    if len(tokens) > request["sampling"]["max_tokens"]:
        raise ValueError("Generation exceeded its approved response cap")
    if record.get("output_token_count") != len(tokens):
        raise CacheConflict("Cached generated-token count is inconsistent")
    if record.get("prompt_token_count") != len(request["prompt_token_ids"]):
        raise CacheConflict("Cached prompt-token count is inconsistent")
    if record.get("finish_reason") not in ("stop", "length"):
        raise ValueError(f"INCOMPLETE_GENERATED_RESPONSE: finish_reason={record.get('finish_reason')}")
    if not isinstance(record.get("response_text"), str):
        raise ValueError("Generation returned no decoded response text")
    probabilities = record.get("token_log_probs")
    if probabilities is not None and (
        len(probabilities) != len(tokens) or any(not math.isfinite(value) for value in probabilities)
    ):
        raise CacheConflict("Cached sampled-token log probabilities are inconsistent")
    return record


def _completion_record(output, request, batch_elapsed, batch_size, engine):
    outputs = getattr(output, "outputs", None)
    if not outputs or len(outputs) != 1 or not getattr(output, "finished", True):
        raise ValueError("Generation must return exactly one finished completion for every n=1 request")
    returned_prompt = getattr(output, "prompt_token_ids", None)
    if returned_prompt is not None and list(returned_prompt) != request["prompt_token_ids"]:
        raise ValueError("Generation engine changed original prompt token IDs")
    completion = outputs[0]
    tokens = [int(token) for token in completion.token_ids]
    cumulative = getattr(completion, "cumulative_logprob", None)
    if cumulative is not None:
        cumulative = float(cumulative)
        if not math.isfinite(cumulative):
            raise ValueError("Generation returned nonfinite cumulative log probability")
    metrics = getattr(output, "metrics", None)
    runtime = {"batch_elapsed_seconds": batch_elapsed, "batch_size": batch_size}
    for name in ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time", "finished_time"):
        value = getattr(metrics, name, None)
        if isinstance(value, int | float) and math.isfinite(value):
            runtime[name] = float(value)
    record = {
        **request,
        "response_token_ids": tokens,
        "response_text": completion.text,
        "finish_reason": completion.finish_reason,
        "stop_reason": getattr(completion, "stop_reason", None),
        "prompt_token_count": len(request["prompt_token_ids"]),
        "output_token_count": len(tokens),
        "token_log_probs": _selected_log_probs(completion, tokens),
        "cumulative_logprob": cumulative,
        "engine_request_id": str(getattr(output, "request_id", "")),
        "engine_class": type(engine).__name__,
        "runtime_stats": runtime,
    }
    record["record_checksum"] = digest(record)
    return _validate_record(record, request)


def _final_contract(path, requests, fingerprint):
    """Validate a final split even if interruption preceded its sidecar commit."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".manifest.json")
    expected = {"generation_fingerprint": fingerprint, "request_ids": [item["request_id"] for item in requests]}
    if sidecar.exists():
        metadata = read_json(sidecar)
        if any(metadata.get(key) != value for key, value in expected.items()) or metadata["sha256"] != file_hash(path):
            raise CacheConflict(f"Completed rollout cache changed: {path}; use a new output directory")
        return metadata
    count = 0
    for count, record in enumerate(read_compressed_records(path), start=1):
        if count > len(requests):
            raise CacheConflict(f"Unexpected extra records in {path}")
        _validate_record(record, requests[count - 1])
    if count != len(requests):
        raise CacheConflict(f"Incomplete final rollout cache {path}")
    metadata = expected | {"sha256": file_hash(path), "records": count}
    atomic_json(sidecar, metadata, immutable=True)
    return metadata


def generate_shared(config, output_dir, model_lock, *, engine_factory=None, sampler_factory=None):
    """Generate both immutable split caches using one lazily constructed engine.

    Injection hooks accept the same keyword arguments as vLLM ``LLM`` and
    ``SamplingParams``. They allow CPU tests without importing vLLM or loading a
    model. Parts commit per completed request; batch failure keeps prior parts.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sampling = _sampling(config)
    if not isinstance(config.get("seed"), int) or not isinstance(config.get("n_rollout"), int):
        raise ConfigurationError("Explicit integer seed and rollout count are required")
    chunk_size = config["generation"]["request_chunk_size"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ConfigurationError("generation.request_chunk_size must be positive")
    revision = model_lock.get("model_revision", model_lock.get("revision"))
    tokenizer_revision = model_lock.get("tokenizer_revision")
    if not revision or revision in ("main", "master") or not tokenizer_revision:
        raise ConfigurationError("Immutable model and tokenizer revisions must be locked before generation")
    manifests = {split: read_records(root / f"{split}_prompt_manifest.parquet") for split in ("train", "heldout")}
    for split, rows in manifests.items():
        if len(rows) != config[split]["n_problem"]:
            raise CacheConflict(f"{split} manifest does not contain the prescribed prompt count")
        identities = []
        for row in rows:
            prompt = _prompt_contract(row)
            identities.append((prompt["row_id"], prompt["prompt_index"]))
            if len(prompt["prompt_token_ids"]) + sampling["max_tokens"] > config["model"]["max_model_len"]:
                raise ConfigurationError(
                    "Original prompt plus generation cap exceeds model context; no truncation allowed"
                )
        if len(set(identities)) != len(identities):
            raise CacheConflict(f"Duplicate prompt-row identities in {split} manifest")
    engine_kwargs = {
        "model": config["model"].get("local_path") or config["model"]["id"],
        "revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "dtype": config["model"]["dtype"],
        "max_model_len": config["model"]["max_model_len"],
        "tensor_parallel_size": config["generation"]["tensor_parallel_size"],
        "gpu_memory_utilization": config["generation"]["gpu_memory_utilization"],
        "enable_prefix_caching": config["generation"]["enable_prefix_caching"],
        "generation_config": "vllm",
        "trust_remote_code": False,
        "seed": config["seed"],
    }
    identity = {
        "schema_version": 2,
        "round_id": config["round_id"],
        "master_seed": config["seed"],
        "seed_derivation": (
            "sha256(canonical JSON master/round/split/prompt_hash/prompt_row_identity/rollout_index/version=2), "
            "first64 mod (2**63-1)"
        ),
        "n_rollout": config["n_rollout"],
        "sampling": sampling,
        "model_lock": model_lock,
        "engine_kwargs": engine_kwargs,
        "request_chunk_size": chunk_size,
        "libraries": _libraries(),
        "engine_backend": "vllm" if engine_factory is None else "injected_test_engine",
        "prompt_manifests": {
            split: {
                "sha256": file_hash(root / f"{split}_prompt_manifest.parquet"),
                "prompts": [_prompt_contract(row) for row in rows],
            }
            for split, rows in manifests.items()
        },
    }
    fingerprint = digest(identity)
    parts_root = root / "rollout_parts"
    atomic_json(parts_root / "generation_identity.json", identity, immutable=True)
    engine = None
    split_metadata = {}
    paths = []
    for split, rows in manifests.items():
        requests = list(_request_records(config, split, rows, sampling, fingerprint))
        destination = root / f"{split}_rollouts.jsonl.zst"
        parts = parts_root / split
        parts.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            split_metadata[split] = _final_contract(destination, requests, fingerprint)
            paths.append(destination)
            LOG.info("Reusing immutable %s rollout cache: %s", split, destination)
            continue
        pending = []
        for request in requests:
            part = parts / f"{request['request_id']}.json"
            if part.exists():
                _validate_record(read_json(part), request)
            else:
                pending.append(request)
        LOG.info(
            "Generation split=%s completed_requests=%s pending=%s", split, len(requests) - len(pending), len(pending)
        )
        for start in range(0, len(pending), chunk_size):
            batch = pending[start : start + chunk_size]
            if engine is None:
                if engine_factory is None:
                    from vllm import LLM, SamplingParams

                    engine_factory, sampler_factory = LLM, SamplingParams
                elif sampler_factory is None:
                    raise ValueError("Tests injecting an engine must also inject sampler_factory")
                LOG.info("Constructing one shared rollout engine: %s", engine_kwargs)
                engine = engine_factory(**engine_kwargs)
            params = [sampler_factory(**sampling, seed=request["seed"]) for request in batch]
            prompts = [{"prompt_token_ids": request["prompt_token_ids"]} for request in batch]
            started = time.monotonic()
            outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            elapsed = time.monotonic() - started
            if len(outputs) != len(batch):
                raise ValueError("Generation engine returned an incomplete request batch")
            for request, output in zip(batch, outputs, strict=True):
                record = _completion_record(output, request, elapsed, len(batch), engine)
                atomic_json(parts / f"{request['request_id']}.json", record, immutable=True)
                LOG.info(
                    "Cached rollout request=%s seed=%s tokens=%s finish=%s batch_elapsed=%.3f",
                    request["request_id"],
                    request["seed"],
                    record["output_token_count"],
                    record["finish_reason"],
                    elapsed,
                )
        write_compressed_records(
            destination, (read_json(parts / f"{request['request_id']}.json") for request in requests)
        )
        split_metadata[split] = _final_contract(destination, requests, fingerprint)
        paths.append(destination)
    manifest = {
        "generation_fingerprint": fingerprint,
        "identity": identity,
        "splits": split_metadata,
        "shared_by_methods": ["GRPO", "Linear Credit Routing", "Nash Credit Routing"],
    }
    atomic_json(root / "generation_manifest.json", manifest, immutable=True)
    paths.append(root / "generation_manifest.json")
    LOG.info("Shared generation complete; exactly one train and one heldout cache: %s", paths)
    return paths
