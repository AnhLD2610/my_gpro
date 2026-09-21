# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""One shared, full-support rollout cache with request-level recovery.

No model, dataset or generation library is loaded on import. Splits may share one
engine or run in isolated engines on disjoint GPUs. All methods consume the same
files, and completed request parts and final split caches remain immutable.
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


def _generation_contract(config, output_dir, model_lock, *, injected=False):
    """Validate both prompt contracts without loading an engine or changing files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sampling = _sampling(config)
    if not isinstance(config.get("seed"), int) or not isinstance(config.get("n_rollout"), int):
        raise ConfigurationError("Explicit integer seed and rollout count are required")
    chunk_size = config["generation"]["request_chunk_size"]
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ConfigurationError("generation.request_chunk_size must be positive")
    dispatch = config["generation"].get("dispatch_mode", "batch")
    if dispatch not in ("batch", "continuous"):
        raise ConfigurationError("generation.dispatch_mode must be batch or continuous")
    window = config["generation"].get("max_in_flight", 256)
    if type(window) is not int or window <= 0:
        raise ConfigurationError("generation.max_in_flight must be a positive integer")
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
            prompt_limit = config.get("prompt", {}).get("max_tokens")
            if prompt_limit is not None and len(prompt["prompt_token_ids"]) > prompt_limit:
                raise ConfigurationError(
                    f"Prompt {split}/{prompt['prompt_index']} exceeds prompt.max_tokens={prompt_limit}; "
                    "no truncation allowed"
                )
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
    for name in ("max_num_seqs", "max_num_batched_tokens"):
        value = config["generation"].get(name)
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"generation.{name} must be a positive integer")
            engine_kwargs[name] = value
    if "enable_chunked_prefill" in config["generation"]:
        value = config["generation"]["enable_chunked_prefill"]
        if type(value) is not bool:
            raise ConfigurationError("generation.enable_chunked_prefill must be a boolean")
        engine_kwargs["enable_chunked_prefill"] = value
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
        "prompt_max_tokens": config.get("prompt", {}).get("max_tokens"),
        "model_lock": model_lock,
        "engine_kwargs": engine_kwargs,
        "request_chunk_size": chunk_size,
        "libraries": _libraries(),
        "engine_backend": "injected_test_engine" if injected else "vllm",
        "prompt_manifests": {
            split: {
                "sha256": file_hash(root / f"{split}_prompt_manifest.parquet"),
                "prompts": [_prompt_contract(row) for row in rows],
            }
            for split, rows in manifests.items()
        },
    }
    # Keep historical batch contracts readable; new scheduling is recorded
    # explicitly because a different batch shape can change GPU roundoff.
    if dispatch == "continuous":
        identity["dispatch"] = {"mode": dispatch, "max_in_flight": window, "output_kind": "FINAL_ONLY"}
    fingerprint = digest(identity)
    return {
        "identity": identity,
        "fingerprint": fingerprint,
        "sampling": sampling,
        "engine_kwargs": engine_kwargs,
        "requests": {
            split: list(_request_records(config, split, rows, sampling, fingerprint))
            for split, rows in manifests.items()
        },
    }


def _split_pending(root, split, requests, fingerprint):
    """Check all reusable data before any new engine is constructed."""
    destination = root / f"{split}_rollouts.jsonl.zst"
    if destination.exists():
        _final_contract(destination, requests, fingerprint)
        return []
    pending = []
    for request in requests:
        part = root / "rollout_parts" / split / f"{request['request_id']}.json"
        if part.exists():
            _validate_record(read_json(part), request)
        else:
            pending.append(request)
    return pending


class _LazyEngine:
    """One process-local engine, shared only in serial execution."""

    def __init__(self, kwargs, engine_factory=None, sampler_factory=None):
        self.kwargs = kwargs
        self.engine_factory = engine_factory
        self.sampler_factory = sampler_factory
        self.engine = None
        self.final_only = 2  # Injection hook for CPU fake samplers.

    def get(self):
        if self.engine is None:
            if self.engine_factory is None:
                from vllm import LLM, SamplingParams
                from vllm.sampling_params import RequestOutputKind

                self.engine_factory, self.sampler_factory = LLM, SamplingParams
                self.final_only = RequestOutputKind.FINAL_ONLY
            elif self.sampler_factory is None:
                raise ValueError("Tests injecting an engine must also inject sampler_factory")
            LOG.info("Constructing rollout engine: %s", self.kwargs)
            self.engine = self.engine_factory(**self.kwargs)
        return self.engine, self.sampler_factory

    def close(self):
        # vLLM v1 exposes process cleanup on the engine-core client. Worker
        # process groups are also reaped by the supervisor before the next stage.
        core = getattr(getattr(self.engine, "llm_engine", None), "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.engine = None


def _finalize_split(root, split, requests, fingerprint):
    """Only the split owner/supervisor assembles the canonical shared cache."""
    destination = root / f"{split}_rollouts.jsonl.zst"
    if destination.exists():
        return _final_contract(destination, requests, fingerprint)

    def records():
        for request in requests:
            part = root / "rollout_parts" / split / f"{request['request_id']}.json"
            if not part.is_file():
                raise CacheConflict(f"Cannot finalize {split}: missing completed request {request['request_id']}")
            yield _validate_record(read_json(part), request)

    write_compressed_records(destination, records())
    return _final_contract(destination, requests, fingerprint)


def _generate_continuous(config, parts, pending, contract, lazy_engine):
    """Refill freed slots immediately and durably commit out-of-order completions.

    Uses the pinned vLLM 0.11 LLMEngine add_request/step API underneath LLM.
    FINAL_ONLY avoids materializing cumulative token/logprob arrays every step.
    """
    engine, sampler_factory = lazy_engine.get()
    core = engine.llm_engine
    window = config["generation"].get("max_in_flight", 256)
    active = {}
    next_index, completed, tokens = 0, 0, 0
    started = last_report = time.monotonic()
    try:
        while next_index < len(pending) or active:
            while next_index < len(pending) and len(active) < window:
                request = pending[next_index]
                params = sampler_factory(**contract["sampling"], seed=request["seed"])
                params.output_kind = lazy_engine.final_only
                request_id = request["request_id"]
                active[request_id] = (request, time.monotonic())
                core.add_request(
                    request_id=request_id, prompt={"prompt_token_ids": request["prompt_token_ids"]}, params=params
                )
                next_index += 1
            if not core.has_unfinished_requests():
                raise ValueError("Generation engine drained before returning all completed requests")
            outputs = core.step()
            for output in outputs:
                request_id = str(output.request_id)
                if request_id not in active:
                    raise ValueError(f"Unknown or duplicate generation output request_id={request_id}")
                if not output.finished:
                    continue
                request, admitted = active[request_id]
                elapsed = time.monotonic() - admitted
                record = _completion_record(output, request, elapsed, 1, engine)
                record["runtime_stats"].update(dispatch_mode="continuous", request_elapsed_seconds=elapsed)
                record["record_checksum"] = digest({k: v for k, v in record.items() if k != "record_checksum"})
                atomic_json(parts / f"{request_id}.json", record, immutable=True)
                del active[request_id]
                completed += 1
                tokens += record["output_token_count"]
            now = time.monotonic()
            if now - last_report >= 10 or completed == len(pending):
                LOG.info(
                    "Continuous generation completed=%s/%s in_flight=%s tokens=%s elapsed=%.2fs tokens/s=%.1f",
                    completed,
                    len(pending),
                    len(active),
                    tokens,
                    now - started,
                    tokens / max(now - started, 1e-9),
                )
                last_report = now
    except BaseException:
        if active:
            try:
                core.abort_request(list(active))
            except Exception:
                LOG.exception("Failed to abort remaining requests; engine teardown will reclaim them")
        raise
    return {
        "dispatch_mode": "continuous",
        "generated_requests": completed,
        "output_tokens": tokens,
        "engine_elapsed_seconds": time.monotonic() - started,
        "max_in_flight": window,
    }


def _generate_split(config, root, split, contract, lazy_engine, *, shard_index=0, shard_count=1):
    """Write only one split's request parts, final cache and final sidecar."""
    requests = contract["requests"][split]
    fingerprint = contract["fingerprint"]
    destination = root / f"{split}_rollouts.jsonl.zst"
    parts = root / "rollout_parts" / split
    parts.mkdir(parents=True, exist_ok=True)
    if (
        type(shard_count) is not int
        or shard_count <= 0
        or type(shard_index) is not int
        or not 0 <= shard_index < shard_count
    ):
        raise ConfigurationError("Invalid generation shard index/count")
    if destination.exists():
        LOG.info("Reusing immutable %s rollout cache: %s", split, destination)
        return _final_contract(destination, requests, fingerprint)
    assigned = [request for request in requests if request["prompt_index"] % shard_count == shard_index]
    pending = _split_pending(root, split, assigned, fingerprint)
    LOG.info(
        "Generation split=%s shard=%s/%s completed_requests=%s pending=%s",
        split,
        shard_index,
        shard_count,
        len(assigned) - len(pending),
        len(pending),
    )
    if pending and config["generation"].get("dispatch_mode", "batch") == "continuous":
        metrics = _generate_continuous(config, parts, pending, contract, lazy_engine)
        metrics.update(
            split=split,
            shard_index=shard_index,
            shard_count=shard_count,
            generation_fingerprint=fingerprint,
            reused_requests=len(assigned) - len(pending),
        )
        atomic_json(root / "generation_metrics" / f"{split}-{shard_index:03d}.json", metrics)
        pending = []
    chunk_size = config["generation"]["request_chunk_size"]
    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        engine, sampler_factory = lazy_engine.get()
        params = [sampler_factory(**contract["sampling"], seed=request["seed"]) for request in batch]
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
    if shard_count > 1:
        return {
            "split": split,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "records": len(assigned),
            "generation_fingerprint": fingerprint,
        }
    return _finalize_split(root, split, requests, fingerprint)


def generate_shared(config, output_dir, model_lock, *, engine_factory=None, sampler_factory=None):
    """Generate shared immutable caches, serially or on disjoint split workers.

    Execution topology is deliberately outside the statistical/cache identity:
    switching to parallel workers preserves request seeds and completed parts.
    Injection hooks for CPU fake engines continue to support serial execution.
    """
    root = Path(output_dir)
    parallel = config["generation"].get("parallel_splits", False)
    if parallel and (engine_factory is not None or sampler_factory is not None):
        raise ValueError("Parallel workers require their own process-local generation engine")
    contract = _generation_contract(config, root, model_lock, injected=engine_factory is not None)
    atomic_json(root / "rollout_parts" / "generation_identity.json", contract["identity"], immutable=True)
    # Check both split caches/parts before the first engine can reserve GPUs.
    for split, requests in contract["requests"].items():
        _split_pending(root, split, requests, contract["fingerprint"])
    if parallel:
        from .parallel_generation import run_split_workers

        run_split_workers(config, root, model_lock, contract)
    else:
        engine = _LazyEngine(contract["engine_kwargs"], engine_factory, sampler_factory)
        try:
            for split in contract["requests"]:
                _generate_split(config, root, split, contract, engine)
        finally:
            engine.close()
    split_metadata = {
        split: _final_contract(root / f"{split}_rollouts.jsonl.zst", requests, contract["fingerprint"])
        for split, requests in contract["requests"].items()
    }
    manifest = {
        "generation_fingerprint": contract["fingerprint"],
        "identity": contract["identity"],
        "splits": split_metadata,
        "shared_by_methods": ["GRPO", "Linear Credit Routing", "Nash Credit Routing"],
    }
    atomic_json(root / "generation_manifest.json", manifest, immutable=True)
    paths = [root / f"{split}_rollouts.jsonl.zst" for split in contract["requests"]]
    paths.append(root / "generation_manifest.json")
    LOG.info("Shared generation complete; exactly one train and one heldout cache: %s", paths)
    return paths
