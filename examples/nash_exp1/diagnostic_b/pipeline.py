# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Stage orchestration; held-out tensors are kept outside routing APIs."""

import dataclasses
import importlib
import importlib.metadata
import inspect
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

from .config import FEATURE_LABELS, feature_geometry
from .storage import (
    CacheConflict,
    StageRunner,
    atomic_json,
    canonical_json,
    digest,
    file_hash,
    read_compressed_records,
    read_json,
    read_records,
    write_records,
)

LOG = logging.getLogger(__name__)
STAGES = (
    "preflight",
    "prepare",
    "generate",
    "verify",
    "heldout_features",
    "training_features_and_routes",
    "statistics",
)


def code_hash(*modules):
    """Stage-local implementation identity keeps solver edits out of rollout cache keys."""
    return {name: file_hash(Path(__file__).with_name(name + ".py")) for name in modules}


def prompt_code_hash():
    """Verifier fixes must not invalidate immutable prepared prompts or rollouts."""
    module = importlib.import_module(".data", __package__)
    return {
        name: digest(inspect.getsource(getattr(module, name)))
        for name in (
            "normalized_problem",
            "adapt_row",
            "validate_wrapper",
            "render_prompt",
            "check_overlap",
            "local_snapshot",
            "load_dataset_snapshot",
            "prepare_manifests",
        )
    }


def package_versions(*names):
    """Record execution library versions without importing GPU packages."""
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def rollout_key(row):
    """Stable identity shared by cache, verification, and features."""
    return row["split"], int(row["prompt_index"]), int(row["rollout_index"])


def feature_extractor(config, root):
    """Build a lazy extractor using immutable checkpoint metadata."""
    from .features import DeltaProxyExtractor, FeatureExtractor

    lock = read_json(root / "model_lock.json")
    model = config["model"]
    extractors = {"delta_proxy": DeltaProxyExtractor, "exact_tiled_head": FeatureExtractor}
    return extractors[config["features"]["backend"]](
        model.get("local_path") or model["id"],
        lock["revision"],
        tokenizer_revision=lock["tokenizer_revision"],
        dtype=model["dtype"],
        device="cuda:0",
        temperature=config["sampling"]["temperature"],
        token_chunk_size=config["features"]["token_chunk_size"],
        vocab_chunk_size=config["features"]["vocab_chunk_size"],
        logit_vocab_chunk_size=config["features"].get("logit_vocab_chunk_size", 8192),
    )


def verify_rollouts(config, root):
    """Use one declared verifier on both caches and expose every parse failure."""
    from .data import MathVerifier

    verifier = MathVerifier(config["verifier"])
    records, counts = [], {}
    started = time.monotonic()
    for split in ("train", "heldout"):
        prompts = {int(row["prompt_index"]): row for row in read_records(root / f"{split}_prompt_manifest.parquet")}
        split_counts = {"draws": 0, "successes": 0, "parse_failures": 0, "statuses": {}}
        observed = set()
        for row in read_compressed_records(root / f"{split}_rollouts.jsonl.zst"):
            key = rollout_key(row)
            if key in observed:
                raise ValueError(f"Duplicate rollout in cache: {key}")
            observed.add(key)
            prompt = prompts[int(row["prompt_index"])]
            if row["prompt_hash"] != prompt["prompt_hash"]:
                raise CacheConflict(f"Prompt mismatch in rollout {key}")
            score = verifier.score(row["response_text"], prompt["answer"])
            status = score["parse_status"]
            split_counts["draws"] += 1
            split_counts["successes"] += score["reward"]
            split_counts["parse_failures"] += status != "parsed"
            split_counts["statuses"][status] = split_counts["statuses"].get(status, 0) + 1
            records.append(
                {
                    "split": split,
                    "prompt_index": int(row["prompt_index"]),
                    "rollout_index": int(row["rollout_index"]),
                    "prompt_hash": row["prompt_hash"],
                    **score,
                }
            )
            if status != "parsed":
                LOG.warning("Verifier key=%s status=%s reason=%s", key, status, score.get("reason"))
        expected = {(split, index, rollout) for index in prompts for rollout in range(config["n_rollout"])}
        if observed != expected:
            raise ValueError(f"Incomplete {split} cache: observed {len(observed)}, expected {len(expected)}")
        split_counts["success_rate"] = split_counts["successes"] / split_counts["draws"]
        counts[split] = split_counts
    counts["elapsed_seconds"] = time.monotonic() - started
    write_records(root / "verifier_results.parquet", records)
    atomic_json(root / "verification_summary.json", counts)
    LOG.info("Verification summary: %s", counts)
    return [root / "verifier_results.parquet", root / "verification_summary.json"]


def extract_response(extractor, row, root, feature_identity):
    """Cache original-token factors and log inference-backend score differences."""
    split, prompt, rollout = rollout_key(row)
    cache_key = digest(
        {
            "identity": feature_identity,
            "request_id": row["request_id"],
            "prompt_hash": row["prompt_hash"],
            "response_ids": row["response_token_ids"],
        }
    )
    directory = (
        root / "features" / "factors" / digest(feature_identity)[:16] / split / f"{prompt:05d}" / f"{rollout:03d}"
    )
    LOG.info(
        "Feature extraction last_processed=%s/%s/%s tokens=%s", split, prompt, rollout, len(row["response_token_ids"])
    )
    path = extractor.extract(row["prompt_token_ids"], row["response_token_ids"], directory, cache_key=cache_key)
    sampled = row.get("token_log_probs")
    if sampled is not None:
        current = np.load(path / "token_log_probs.npy", allow_pickle=False)
        if len(sampled) != len(current):
            raise ValueError("Stored sampling log probabilities omit generated action tokens")
        difference = np.asarray(sampled, dtype=np.float64) - current
        if not np.isfinite(difference).all():
            raise ValueError("Nonfinite sampled/recomputed score discrepancy")
        LOG.info(
            "Policy numeric audit key=%s mean_abs_logp_difference=%g max_abs_logp_difference=%g",
            (split, prompt, rollout),
            float(np.mean(np.abs(difference))),
            float(np.max(np.abs(difference))),
        )
    return path


def compute_heldout_features(config, root):
    """One method-independent h_Q, summing scores and dividing by ALL M*K draws."""
    from .features import HeadAtom

    verification = {rollout_key(row): row for row in read_records(root / "verifier_results.parquet")}
    counts = read_json(root / "verification_summary.json")["heldout"]
    if counts["successes"] == 0:
        raise RuntimeError(
            "INSUFFICIENT_HELDOUT_SUCCESSES: all held-out rewards are zero; h_Q=0 and correlations undefined. "
            "Rollouts and verifier results are cached. Do not change sample size or Q without approval."
        )
    started = time.monotonic()
    extractor = feature_extractor(config, root)
    atoms = []
    token_count = 0
    identity = {
        "code": code_hash("features"),
        "sampling": config["sampling"],
        "model": read_json(root / "model_lock.json"),
        "feature_config": config["features"],
        "libraries": package_versions("torch", "transformers", "numpy"),
    }
    for row in read_compressed_records(root / "heldout_rollouts.jsonl.zst"):
        token_count += len(row["response_token_ids"])
        if verification[rollout_key(row)]["reward"]:
            path = extract_response(extractor, row, root, identity)
            atoms.append(HeadAtom(path))  # No per-response length normalization.
    target = root / "features" / "heldout_head.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    extractor.build_heldout_head(atoms, np.ones(len(atoms)), counts["draws"], target)
    manifest = {
        "geometry": feature_geometry(config),
        "head_shape": list(extractor.head_shape),
        "feature_shape": list(extractor.feature_shape),
        "feature_dtype": "float32",
        "heldout_head_path": str(target.relative_to(root)),
        "heldout_head_sha256": file_hash(target),
        "heldout_draws": counts["draws"],
        "heldout_successes": counts["successes"],
        "denominator": counts["draws"],
        "response_aggregation": "token_sum",
        "heldout_tokens": token_count,
        "elapsed_seconds": time.monotonic() - started,
        "identity": identity,
    }
    atomic_json(root / "heldout_feature_manifest.json", manifest)
    return [target, root / "heldout_feature_manifest.json"]


def make_group_atoms(rows, rewards, extractor, tokenizer, root, identity, max_atoms):
    """Raw response means and negative segment means at the fresh rollout checkpoint."""
    from .features import HeadAtom
    from .segmentation import decode_original_token_spans, segment_token_spans

    positive, negative = [], []
    for row, reward in zip(rows, rewards, strict=True):
        length = len(row["response_token_ids"])
        if length == 0:
            raise ValueError(f"Empty generated response {rollout_key(row)} has undefined response mean")
        path = extract_response(extractor, row, root, identity)
        if reward:
            weights = np.full(length, 1 / length, dtype=np.float32)
            positive.append((HeadAtom(path, weights), {"rollout_index": int(row["rollout_index"]), "length": length}))
        else:
            text, spans = decode_original_token_spans(tokenizer, row["response_token_ids"])
            segments = segment_token_spans(text, spans)
            if len(negative) + len(segments) + int(np.sum(rewards)) > max_atoms:
                raise RuntimeError(f"ACTUAL_ATOMS_EXCEED_PREFLIGHT: budget={max_atoms}; no segments dropped")
            for start, end in segments:
                weights = np.zeros(length, dtype=np.float32)
                weights[start:end] = 1 / (end - start)
                negative.append(
                    (
                        HeadAtom(path, weights),
                        {
                            "rollout_index": int(row["rollout_index"]),
                            "start": start,
                            "end": end,
                            "length": end - start,
                            "response_length": length,
                        },
                    )
                )
    return [atom for atom, _ in positive + negative], [info for _, info in positive], [info for _, info in negative]


def route_training_gram(gram, positive_info, negative_info, rewards, config):
    """The routing boundary accepts no held-out features, cross terms, or rewards."""
    from .routing import SolverConfig, TrainingGeometry, solve_group, standardized_advantages

    advantages = standardized_advantages(np.asarray(rewards), config["routing"]["epsilon"])
    group_size = len(rewards)
    positive = np.asarray([advantages[item["rollout_index"]] / group_size for item in positive_info])
    negative = np.asarray(
        [
            -advantages[item["rollout_index"]] / group_size * item["length"] / item["response_length"]
            for item in negative_info
        ]
    )
    # A new narrow object is constructed; no held-out data can flow into the solver.
    geometry = TrainingGeometry(gram=np.asarray(gram), positive_coefficients=positive, negative_coefficients=negative)
    return solve_group(geometry, config["routing"]["radius_coefficient"], SolverConfig(**config["routing"]["solver"]))


def compute_training_features_and_routes(config, root):
    """Process one training group at a time, reusing cached exact small Gram matrices."""
    from transformers import AutoTokenizer

    from .routing import SolverConfig

    started = time.monotonic()
    verifier = {rollout_key(row): row for row in read_records(root / "verifier_results.parquet")}
    prompts = read_records(root / "train_prompt_manifest.parquet")
    grouped = {}
    for row in read_compressed_records(root / "train_rollouts.jsonl.zst"):
        grouped.setdefault(int(row["prompt_index"]), []).append(row)
    lock = read_json(root / "model_lock.json")
    extractor = feature_extractor(config, root)
    tokenizer = None
    heldout = root / "features" / "heldout_head.npy"
    feature_identity = {
        "code": code_hash("features", "segmentation"),
        "sampling": config["sampling"],
        "model": lock,
        "feature_config": config["features"],
        "libraries": package_versions("torch", "transformers", "numpy"),
    }
    directions, routes, diagnostics = [], [], []
    counts = {
        "sampled_prompts": len(prompts),
        "all_zero_groups": 0,
        "all_one_groups": 0,
        "mixed_groups": 0,
        "successful_directions": 0,
        "zero_nonfinite_feature_exclusions": 0,
        "power_solver_failures": 0,
        "linear_solver_failures": 0,
        "ncr_solver_failures": 0,
        "route_fallbacks": 0,
        "training_tokens": 0,
        "pre_solver_group_fallbacks": 0,
    }
    feature_files = []
    heldout_hash = file_hash(heldout)
    for prompt in prompts:
        index = int(prompt["prompt_index"])
        rows = sorted(grouped[index], key=lambda row: row["rollout_index"])
        rewards = np.asarray([verifier[rollout_key(row)]["reward"] for row in rows])
        m, size = int(rewards.sum()), len(rows)
        counts["training_tokens"] += sum(len(row["response_token_ids"]) for row in rows)
        base_record = {
            "round_id": config["round_id"],
            "prompt_index": index,
            "row_id": prompt["row_id"],
            "prompt_hash": prompt["prompt_hash"],
            "group_success_count": m,
            "group_size": size,
        }
        if m in (0, size):
            counts["all_zero_groups" if m == 0 else "all_one_groups"] += 1
            routes.append({**base_record, "status": "homogeneous", "reason": "zero standardized advantages"})
            continue
        counts["mixed_groups"] += 1
        counts["successful_directions"] += m
        cache_dir = root / "features" / f"train_group_{index:05d}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path, metadata_path = cache_dir / "gram.npz", cache_dir / "metadata.json"
        identity = digest(
            {
                "features": feature_identity,
                "requests": [row["request_id"] for row in rows],
                "responses": [row["response_token_ids"] for row in rows],
                "rewards": rewards.tolist(),
                "heldout_hash": heldout_hash,
            }
        )
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        if (
            metadata.get("identity") == identity
            and cache_path.exists()
            and file_hash(cache_path) == metadata.get("sha256")
        ):
            with np.load(cache_path, allow_pickle=False) as archive:
                gram, cross = archive["gram"], archive["heldout_cross"]
            positive_info, negative_info = metadata["positive_info"], metadata["negative_info"]
            LOG.info(
                "Reusing group Gram geometry=%s prompt=%s hash=%s", feature_geometry(config), index, metadata["sha256"]
            )
        else:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config["model"].get("local_path") or config["model"]["id"],
                    revision=lock["tokenizer_revision"],
                    trust_remote_code=False,
                )
            max_atoms = config["n_rollout"] * config["features"]["max_segments_per_response_budget"]
            atoms, positive_info, negative_info = make_group_atoms(
                rows, rewards, extractor, tokenizer, root, feature_identity, max_atoms
            )
            if len(atoms) > max_atoms:
                raise RuntimeError(
                    f"ACTUAL_ATOMS_EXCEED_PREFLIGHT: group={index} atoms={len(atoms)} budget={max_atoms}. "
                    "Increase the declared memory/segment estimate after reviewing preflight; no segments dropped."
                )
            result = extractor.build_gram(atoms, heldout)
            gram, cross = result["gram"], result["heldout_cross"]
            LOG.info(
                "Group %s Gram geometry=%s atoms=%s min_eigenvalue=%g",
                index,
                feature_geometry(config),
                len(atoms),
                result["min_eigenvalue"],
            )
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, gram=gram, heldout_cross=cross)
            os.replace(temporary, cache_path)
            metadata = {
                "identity": identity,
                "sha256": file_hash(cache_path),
                "positive_info": positive_info,
                "negative_info": negative_info,
                "min_eigenvalue": result["min_eigenvalue"],
            }
            atomic_json(metadata_path, metadata)
            extractor.clear_factor_cache()
        feature_files.append(
            {"prompt_index": index, "path": str(cache_path.relative_to(root)), "sha256": file_hash(cache_path)}
        )
        solution = route_training_gram(gram, positive_info, negative_info, rewards, config)
        gram = (gram + gram.T) / 2
        norms = np.sqrt(np.maximum(np.diag(gram)[:m], 0))
        delta_difference = solution.ncr.coefficients - solution.linear.coefficients
        difference_norm = math.sqrt(max(0, float(delta_difference @ gram @ delta_difference)))
        power_failed = bool(solution.power_diagnostics.get("fallback"))
        counts["power_solver_failures"] += power_failed and bool(solution.power_diagnostics.get("attempted"))
        counts["pre_solver_group_fallbacks"] += power_failed and not solution.power_diagnostics.get("attempted", False)
        for method, route in (("Linear", solution.linear), ("NCR", solution.ncr)):
            counts["route_fallbacks"] += route.fallback
            counts["linear_solver_failures" if method == "Linear" else "ncr_solver_failures"] += (
                route.fallback and route.diagnostics.get("solve_attempted", False)
            )
            diagnostics.append(
                {
                    **base_record,
                    "method": method,
                    "fallback": route.fallback,
                    "reason": route.reason,
                    "diagnostics_json": canonical_json(route.diagnostics),
                    "power_diagnostics_json": canonical_json(solution.power_diagnostics),
                    "group_diagnostics_json": canonical_json(solution.diagnostics),
                }
            )
            LOG.info(
                "Group=%s method=%s fallback=%s reason=%s residuals=%s",
                index,
                method,
                route.fallback,
                route.reason,
                route.diagnostics,
            )
            routes.append(
                {
                    **base_record,
                    "status": "fallback" if route.fallback else "accepted",
                    "method": method,
                    "radius": solution.radius,
                    "baseline_norm": solution.baseline_norm,
                    "positive_shares_json": canonical_json(route.positive_shares),
                    "negative_segments_json": canonical_json(
                        [
                            {**item, "refund": float(refund)}
                            for item, refund in zip(negative_info, route.refunds, strict=True)
                        ]
                    ),
                    "reason": route.reason,
                    "geometry": feature_geometry(config),
                }
            )
        for position, positive in enumerate(positive_info):
            norm = float(norms[position])
            if not np.isfinite(norm) or norm <= SolverConfig(**config["routing"]["solver"]).zero_norm_tolerance:
                counts["zero_nonfinite_feature_exclusions"] += 1
                continue
            usefulness = float(cross[position] / norm)
            baseline_support = float((gram[:m] @ solution.baseline_coefficients)[position] / norm)
            added_linear = float(gram[position] @ solution.linear.coefficients / norm)
            added_ncr = float(gram[position] @ solution.ncr.coefficients / norm)
            linear_support, ncr_support = baseline_support + added_linear, baseline_support + added_ncr
            if not np.isfinite([usefulness, baseline_support, linear_support, ncr_support]).all():
                counts["zero_nonfinite_feature_exclusions"] += 1
                continue
            directions.append(
                {
                    **base_record,
                    "train_dataset_row_id": prompt["row_id"],
                    "rollout_index": positive["rollout_index"],
                    "response_token_length": positive["length"],
                    "reward": 1,
                    "p_norm": norm,
                    "B_i": usefulness,
                    "S_GRPO": baseline_support,
                    "S_Linear": linear_support,
                    "S_NCR": ncr_support,
                    "DeltaS_Linear": added_linear,
                    "DeltaS_NCR": added_ncr,
                    "q_i": float(solution.powers[position]),
                    "w_Linear": float(solution.linear.positive_shares[position]),
                    "w_NCR": float(solution.ncr.positive_shares[position]),
                    "utility_Linear": float(solution.linear.utilities[position]),
                    "utility_NCR": float(solution.ncr.utilities[position]),
                    "fallback_Linear": solution.linear.fallback,
                    "fallback_NCR": solution.ncr.fallback,
                    "fallback_reason_Linear": solution.linear.reason,
                    "fallback_reason_NCR": solution.ncr.reason,
                    "solver_residuals_Linear": canonical_json(solution.linear.diagnostics),
                    "solver_residuals_NCR": canonical_json(solution.ncr.diagnostics),
                    "route_difference_norm": difference_norm,
                    "feature_geometry": feature_geometry(config),
                }
            )
    counts["eligible_directions"] = len(directions)
    counts["elapsed_seconds"] = time.monotonic() - started
    counts["verifier_parse_failures"] = read_json(root / "verification_summary.json")
    for filename, records in (
        ("direction_records.parquet", directions),
        ("route_records.parquet", routes),
        ("solver_diagnostics.parquet", diagnostics),
    ):
        write_records(root / filename, records)
    manifest = {
        "geometry": feature_geometry(config),
        "backend": config["features"]["backend"],
        "feature_identity": feature_identity,
        "heldout": read_json(root / "heldout_feature_manifest.json"),
        "training_group_files": feature_files,
        "solver_config": dataclasses.asdict(SolverConfig(**config["routing"]["solver"])),
        "counts": counts,
    }
    atomic_json(root / "feature_manifest.json", manifest)
    atomic_json(root / "eligibility_summary.json", counts)
    return [
        root / name
        for name in (
            "direction_records.parquet",
            "route_records.parquet",
            "solver_diagnostics.parquet",
            "feature_manifest.json",
            "eligibility_summary.json",
        )
    ]


def compute_statistics(config, root):
    """Evaluate the same retained successful directions for every routing mechanism."""
    from .statistics import analyze_records

    stage_state = read_json(root / "stage_state.json")
    runtime = {name: stage.get("elapsed_seconds") for name, stage in stage_state["stages"].items()}
    atomic_json(root / "runtime_breakdown.json", runtime)
    solver_rows = read_records(root / "solver_diagnostics.parquet")
    solver_summary = {}
    for method in ("Linear", "NCR"):
        selected = [row for row in solver_rows if row["method"] == method]
        numerical = {}
        for row in selected:
            import json

            for key, value in json.loads(row["diagnostics_json"]).items():
                if type(value) in (int, float) and math.isfinite(value):
                    numerical.setdefault(key, []).append(value)
        solver_summary[method] = {
            "groups": len(selected),
            "fallback_groups": sum(row["fallback"] for row in selected),
            "residuals": {
                key: {"minimum": min(values), "maximum": max(values), "median": float(np.median(values))}
                for key, values in numerical.items()
            },
        }
    summary = {
        "counts": read_json(root / "eligibility_summary.json"),
        "heldout": read_json(root / "verification_summary.json")["heldout"],
        "runtime_seconds": runtime,
        "generation": read_json(root / "generation_manifest.json"),
        "features": read_json(root / "feature_manifest.json"),
        "solver_records": solver_rows,
        "solver_residual_summary": solver_summary,
        "feature_geometry": feature_geometry(config),
        "interpretation": f"Frozen-checkpoint {FEATURE_LABELS[feature_geometry(config)]} alignment diagnostic; "
        "no accuracy update was run. "
        "Higher correlation is not proof of transfer or causal accuracy improvement.",
    }
    seed = config["statistics"].get("bootstrap_seed")
    if seed is None:
        seed = int(digest({"master_seed": config["seed"], "purpose": "prompt-cluster-bootstrap"})[:8], 16)
    analyze_records(
        read_records(root / "direction_records.parquet"),
        root,
        config["round_id"],
        config["statistics"]["bootstrap_replicates"],
        seed,
        config["statistics"]["active_tolerance"],
        summary=summary,
    )
    files = [root / "runtime_breakdown.json"]
    for pattern in (
        "correlations.*",
        "analysis_metadata.csv",
        "report_*.md",
        "diagnostic_b_*.pdf",
        "diagnostic_b_*.png",
    ):
        files.extend(sorted(root.glob(pattern)))
    return files


def run_stage(name, config, output_dir, *, resume=False, force_stage=None, enable_round2=False):
    """Execute one resumable stage; `all` invokes these in isolated child processes."""
    from .data import local_snapshot, prepare_manifests
    from .generation import generate_shared
    from .preflight import run_preflight

    root = Path(output_dir).resolve()
    runner = StageRunner(root, resume=resume, force_stage=force_stage)
    if name == "preflight":
        return runner.run(
            name,
            {
                "model": config["model"],
                "memory": config["memory"],
                "features": config["features"],
                "sampling": config["sampling"],
                "generation": config["generation"],
                "counts": [config["train"]["n_problem"], config["heldout"]["n_problem"], config["n_rollout"]],
                "code": code_hash("preflight"),
            },
            lambda: run_preflight(config, root, enable_round2=enable_round2),
        )
    if not (root / "model_lock.json").exists():
        raise RuntimeError("Run preflight before preparing prompts or loading the generation engine")
    lock = read_json(root / "model_lock.json")
    if name == "prepare":
        inputs = {
            "train": config["train"],
            "heldout": config["heldout"],
            "model_lock": lock,
            "seed": config["seed"],
            "prompt": file_hash(config["prompt"]["wrapper_file"]),
            "sampling": config["sampling"],
            "code": prompt_code_hash(),
            "libraries": package_versions("datasets", "transformers"),
        }
        if config["train"].get("local_path"):
            inputs["local_train_content"] = local_snapshot(config["train"]["local_path"])
        return runner.run(name, inputs, lambda: prepare_manifests(config, root, lock), immutable=True)
    if name == "generate":
        inputs = {
            "prepare": runner.dependencies(["prepare"]),
            "model_lock": lock,
            "sampling": config["sampling"],
            "generation": config["generation"],
            "seed": config["seed"],
            "n_rollout": config["n_rollout"],
            "code": code_hash("generation"),
            "libraries": package_versions("vllm", "transformers", "torch"),
        }
        return runner.run(name, inputs, lambda: generate_shared(config, root, lock), immutable=True)
    if name == "verify":
        inputs = {
            "generation": runner.dependencies(["generate"]),
            "verifier": config["verifier"],
            "code": code_hash("data"),
            "libraries": package_versions("math-verify", "sympy"),
        }
        return runner.run(name, inputs, lambda: verify_rollouts(config, root))
    if name == "heldout_features":
        inputs = {
            "dependencies": runner.dependencies(["generate", "verify"]),
            "model_lock": lock,
            "features": config["features"],
            "code": code_hash("features"),
            "libraries": package_versions("torch", "transformers", "numpy"),
        }
        return runner.run(name, inputs, lambda: compute_heldout_features(config, root))
    if name == "training_features_and_routes":
        inputs = {
            "dependencies": runner.dependencies(["generate", "verify", "heldout_features"]),
            "features": config["features"],
            "routing": config["routing"],
            "code": code_hash("features", "segmentation", "routing", "pipeline"),
            "libraries": package_versions("torch", "transformers", "numpy", "cvxpy", "clarabel", "scipy"),
        }
        return runner.run(name, inputs, lambda: compute_training_features_and_routes(config, root))
    if name == "statistics":
        inputs = {
            "dependencies": runner.dependencies(["training_features_and_routes"]),
            "statistics": config["statistics"],
            "seed": config["seed"],
            "code": code_hash("statistics", "pipeline"),
            "libraries": package_versions("scipy", "numpy", "matplotlib"),
        }
        return runner.run(name, inputs, lambda: compute_statistics(config, root))
    raise ValueError(f"Unknown stage {name}")
