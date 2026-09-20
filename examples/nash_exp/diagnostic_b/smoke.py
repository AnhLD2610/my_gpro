# Copyright 2026 Nash Credit Routing contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Synthetic CPU integration run; never evidence about Qwen or benchmark accuracy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import FEATURE_GEOMETRIES
from .features import delta_proxy_aggregate
from .generation import generate_shared
from .pipeline import route_training_gram
from .statistics import analyze_records
from .storage import StageRunner, atomic_json, digest, read_compressed_records, read_json, write_records


class SyntheticEngine:
    """Deterministic fake generation interface for pipeline/cache smoke testing."""

    def __init__(self, **kwargs):
        self.instances = 1

    def generate(self, prompts, sampling_params, use_tqdm=False):
        """Emit small toy vocabulary sequences, preserving per-request seeds."""
        outputs = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            rng = np.random.default_rng(params.seed)
            ids = rng.integers(1, 6, size=int(rng.integers(2, 5))).tolist() + [0]
            completion = SimpleNamespace(
                token_ids=ids,
                text="SYNTHETIC FIXTURE",
                finish_reason="stop",
                stop_reason=0,
                logprobs=None,
                cumulative_logprob=None,
            )
            outputs.append(
                SimpleNamespace(
                    outputs=[completion],
                    prompt_token_ids=prompt["prompt_token_ids"],
                    finished=True,
                    request_id="synthetic",
                )
            )
        return outputs


def run_cpu_smoke(output_dir, *, resume=True):
    """Run shared generation, toy DelTA proxy geometry, routing and reports on CPU."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "round_id": "round_cpu_smoke",
        "seed": 17,
        "n_rollout": 4,
        "model": {"id": "SYNTHETIC-NO-MODEL", "dtype": "float32", "max_model_len": 32},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_new_tokens": 6},
        "generation": {
            "tensor_parallel_size": 1,
            "request_chunk_size": 4,
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": True,
        },
        "train": {"n_problem": 4},
        "heldout": {"n_problem": 2},
        "routing": {"radius_coefficient": 0.3, "epsilon": 1e-6, "solver": {}},
        "notice": "SYNTHETIC CPU FIXTURE ONLY. Constants are not approved production experimental settings.",
    }
    lock = {"revision": "synthetic", "tokenizer_revision": "synthetic"}
    runner = StageRunner(root, resume=resume)

    def prepare():
        files = []
        for split in ("train", "heldout"):
            rows = [
                {
                    "prompt_index": index,
                    "row_id": f"synthetic-{split}-{index}",
                    "prompt_hash": digest([split, index]),
                    "prompt_token_ids": [1, index + 1],
                    "prompt_token_count": 2,
                }
                for index in range(config[split]["n_problem"])
            ]
            path = root / f"{split}_prompt_manifest.parquet"
            write_records(path, rows)
            files.append(path)
        atomic_json(root / "synthetic_config.json", config)
        return files + [root / "synthetic_config.json"]

    runner.run("prepare", {"synthetic_config": config}, prepare, immutable=True)
    runner.run(
        "generate",
        {"prepare": runner.dependencies(["prepare"])},
        lambda: generate_shared(config, root, lock, engine_factory=SyntheticEngine, sampler_factory=SimpleNamespace),
        immutable=True,
    )

    def measure():
        rng = np.random.default_rng(18)
        weight = rng.normal(size=(6, 4))

        def score(row, mean):
            local_rng = np.random.default_rng(row["seed"])
            ids = np.asarray(row["response_token_ids"])
            hidden = local_rng.normal(size=(len(ids), 4))
            weights = np.full(len(ids), 1 / len(ids)) if mean else None
            return delta_proxy_aggregate(hidden, hidden @ weight.T, ids, weights)

        heldout_rows = list(read_compressed_records(root / "heldout_rollouts.jsonl.zst"))
        h = sum((score(row, False) for row in heldout_rows if row["rollout_index"] == 0), np.zeros(4)) / len(
            heldout_rows
        )
        all_rows = list(read_compressed_records(root / "train_rollouts.jsonl.zst"))
        records = []
        for index in range(config["train"]["n_problem"]):
            rows = [row for row in all_rows if row["prompt_index"] == index]
            atoms = np.stack([score(row, True) for row in rows])
            gram = atoms @ atoms.T
            info = [{"rollout_index": k, "length": len(rows[k]["response_token_ids"])} for k in range(2)]
            negative = [
                {
                    "rollout_index": k,
                    "length": len(rows[k]["response_token_ids"]),
                    "response_length": len(rows[k]["response_token_ids"]),
                }
                for k in range(2, 4)
            ]
            solution = route_training_gram(gram, info, negative, [1, 1, 0, 0], config)
            norms = np.linalg.norm(atoms[:2], axis=1)
            gap = solution.ncr.coefficients - solution.linear.coefficients
            for i in range(2):
                baseline = float((gram[i] @ solution.baseline_coefficients) / norms[i])
                added_linear = float(gram[i] @ solution.linear.coefficients / norms[i])
                added_ncr = float(gram[i] @ solution.ncr.coefficients / norms[i])
                records.append(
                    {
                        "round_id": "round_cpu_smoke",
                        "prompt_index": index,
                        "rollout_index": i,
                        "group_success_count": 2,
                        "group_size": 4,
                        "reward": 1,
                        "p_norm": norms[i],
                        "B_i": float(atoms[i] @ h / norms[i]),
                        "S_GRPO": baseline,
                        "S_Linear": baseline + added_linear,
                        "S_NCR": baseline + added_ncr,
                        "DeltaS_Linear": added_linear,
                        "DeltaS_NCR": added_ncr,
                        "route_difference_norm": np.sqrt(max(0, float(gap @ gram @ gap))),
                        "fallback_Linear": solution.linear.fallback,
                        "fallback_NCR": solution.ncr.fallback,
                        "feature_geometry": FEATURE_GEOMETRIES["delta_proxy"],
                        "synthetic": True,
                    }
                )
        write_records(root / "direction_records.parquet", records)
        atomic_json(root / "smoke_records.json", records)
        return [root / "direction_records.parquet", root / "smoke_records.json"]

    runner.run("features_and_routes", {"generation": runner.dependencies(["generate"]), "fixture_version": 2}, measure)

    def statistics():
        analyze_records(
            read_json(root / "smoke_records.json"),
            root,
            "round_cpu_smoke",
            32,
            19,
            1e-5,
            summary={"notice": config["notice"], "no_model_or_dataset_loaded": True},
        )
        return [root / "correlations.json", root / "correlations.csv", root / "report_round_cpu_smoke.md"]

    runner.run("statistics", {"measurements": runner.dependencies(["features_and_routes"])}, statistics)
    return {
        "status": "passed",
        "synthetic": True,
        "output_dir": str(root),
        "model_loaded": False,
        "dataset_loaded": False,
    }
