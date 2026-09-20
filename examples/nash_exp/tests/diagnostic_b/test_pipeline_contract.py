# Copyright 2026 The Nash Credit Routing Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training/statistics integration with exact synthetic Grams; no models or datasets."""

import dataclasses
import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from diagnostic_b import pipeline, routing, storage


def _synthetic_config():
    return {
        "round_id": "round1",
        "seed": 7,
        "n_rollout": 3,
        "model": {"id": "synthetic-model-not-loaded", "dtype": "float32"},
        "sampling": {"temperature": 1.0},
        "features": {"backend": "synthetic_exact_gram", "max_segments_per_response_budget": 4},
        "routing": {"epsilon": 1e-6, "radius_coefficient": 0.3, "solver": {}},
        "statistics": {"bootstrap_replicates": 12, "bootstrap_seed": 4, "active_tolerance": 1e-5},
    }


def _prepare_synthetic_inputs(root, reward_groups):
    prompts, rollouts, verification = [], [], []
    for prompt_index, rewards in enumerate(reward_groups):
        prompt_hash = f"synthetic-prompt-{prompt_index}"
        prompts.append({"prompt_index": prompt_index, "row_id": f"row-{prompt_index}", "prompt_hash": prompt_hash})
        for rollout_index, reward in enumerate(rewards):
            key = {"split": "train", "prompt_index": prompt_index, "rollout_index": rollout_index}
            rollouts.append(
                {
                    **key,
                    "prompt_hash": prompt_hash,
                    "request_id": f"request-{prompt_index}-{rollout_index}",
                    "response_token_ids": [1, 2, 3],
                }
            )
            verification.append({**key, "reward": reward, "parse_status": "parsed"})
    storage.write_records(root / "train_prompt_manifest.parquet", prompts)
    storage.write_records(root / "verifier_results.parquet", verification)
    storage.write_compressed_records(root / "train_rollouts.jsonl.zst", rollouts)
    storage.atomic_json(root / "model_lock.json", {"revision": "synthetic", "tokenizer_revision": "synthetic"})
    storage.atomic_json(
        root / "verification_summary.json",
        {
            "heldout": {"draws": 3, "successes": 1, "success_rate": 1 / 3, "parse_failures": 0},
            "train": {
                "draws": len(rollouts),
                "successes": sum(row["reward"] for row in verification),
                "parse_failures": 0,
            },
        },
    )
    storage.atomic_json(root / "heldout_feature_manifest.json", {"denominator": 3, "response_aggregation": "token_sum"})
    storage.atomic_json(root / "generation_manifest.json", {"synthetic_fixture": True})
    storage.atomic_json(root / "stage_state.json", {"stages": {}})
    (root / "features").mkdir()
    np.save(root / "features" / "heldout_head.npy", np.array([0.4, -0.1]))


def _mock_model_boundary(monkeypatch, basis, *, gram=None):
    """Supply head atom geometry directly while detecting accidental real loads."""
    calls = {"build_gram": 0, "tokenizer": 0}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer"] += 1
            return cls()

    class ExactSyntheticExtractor:
        def build_gram(self, atoms, heldout):
            calls["build_gram"] += 1
            h = basis.T @ basis if gram is None else gram
            return {
                "gram": h.copy(),
                "heldout_cross": basis.T @ np.load(heldout),
                "min_eigenvalue": float(np.linalg.eigvalsh(h).min()),
            }

        def clear_factor_cache(self):
            pass

    def make_atoms(rows, rewards, *args):
        positive = [{"rollout_index": i, "length": 3} for i, reward in enumerate(rewards) if reward]
        negative = [
            {"rollout_index": i, "start": 0, "end": 3, "length": 3, "response_length": 3}
            for i, reward in enumerate(rewards)
            if not reward
        ]
        return [None] * (len(positive) + len(negative)), positive, negative

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr(pipeline, "feature_extractor", lambda *args: ExactSyntheticExtractor())
    monkeypatch.setattr(pipeline, "make_group_atoms", make_atoms)
    return calls


def _coefficients_from_records(route_records, directions, config):
    advantages = routing.standardized_advantages(np.array([1, 1, 0]), config["routing"]["epsilon"])
    a, beta = advantages[:2] / 3, -advantages[2:] / 3
    baseline = np.r_[a, -beta]
    coefficients = {}
    for record in route_records:
        method = record["method"]
        shares = np.array(json.loads(record["positive_shares_json"]))
        refunds = np.array([segment["refund"] for segment in json.loads(record["negative_segments_json"])])
        coefficients[method] = np.r_[a.sum() * (shares - a / a.sum()), beta * refunds]
    return baseline, coefficients


def test_routing_boundary_has_no_heldout_argument_or_object_field():
    assert {field.name for field in dataclasses.fields(routing.TrainingGeometry)} == {
        "gram",
        "positive_coefficients",
        "negative_coefficients",
    }
    for function in (pipeline.route_training_gram, routing.solve_group, routing.collision_powers):
        assert not any("heldout" in name or name == "B_i" for name in inspect.signature(function).parameters)
    with pytest.raises(TypeError, match="heldout_cross"):
        pipeline.route_training_gram(np.eye(3), [], [], [1, 1, 0], _synthetic_config(), heldout_cross=np.ones(3))


def test_switching_to_delta_reuses_generation_and_rebuilds_features(tmp_path, monkeypatch):
    from diagnostic_b import generation

    config = _synthetic_config() | {"generation": {}}
    storage.atomic_json(tmp_path / "model_lock.json", {"revision": "pinned", "tokenizer_revision": "pinned"})
    runner = storage.StageRunner(tmp_path, resume=True)
    marker = tmp_path / "prepared.json"
    storage.atomic_json(marker, {"fixed_prompts": True})
    runner.run("prepare", {"fixed": True}, lambda: [marker], immutable=True)
    runner.run("verify", {"fixed": True}, lambda: [marker])
    calls = {"generate": 0, "features": 0}

    def generate_shared(config, root, lock):
        calls["generate"] += 1
        storage.atomic_json(root / "cached_responses.json", {"responses": "fixed synthetic rollouts"})
        return [root / "cached_responses.json"]

    def heldout_features(config, root):
        calls["features"] += 1
        storage.atomic_json(root / "heldout.json", {"backend": config["features"]["backend"]})
        return [root / "heldout.json"]

    monkeypatch.setattr(generation, "generate_shared", generate_shared)
    monkeypatch.setattr(pipeline, "compute_heldout_features", heldout_features)
    snapshots = []
    for backend in ("exact_tiled_head", "delta_proxy"):
        config["features"]["backend"] = backend
        pipeline.run_stage("generate", config, tmp_path, resume=True)
        cache = tmp_path / "cached_responses.json"
        snapshots.append((storage.file_hash(cache), cache.stat().st_mtime_ns))
        pipeline.run_stage("heldout_features", config, tmp_path, resume=True)
    assert calls == {"generate": 1, "features": 2}
    assert snapshots[0] == snapshots[1]
    assert storage.read_json(tmp_path / "heldout.json")["backend"] == "delta_proxy"


def test_pipeline_support_and_added_support_match_explicit_vectors(tmp_path, monkeypatch):
    config = _synthetic_config()
    basis = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.3]])
    _prepare_synthetic_inputs(tmp_path, [[1, 1, 0]])
    _mock_model_boundary(monkeypatch, basis)
    pipeline.compute_training_features_and_routes(config, tmp_path)
    directions = storage.read_records(tmp_path / "direction_records.parquet")
    baseline, deltas = _coefficients_from_records(
        storage.read_records(tmp_path / "route_records.parquet"), directions, config
    )
    for i, row in enumerate(directions):
        unit = basis[:, i] / np.linalg.norm(basis[:, i])
        assert row["S_GRPO"] == pytest.approx(unit @ basis @ baseline, abs=1e-10)
        assert row["B_i"] == pytest.approx(unit @ np.load(tmp_path / "features" / "heldout_head.npy"), abs=1e-10)
        for method in ("Linear", "NCR"):
            assert row[f"S_{method}"] == pytest.approx(unit @ basis @ (baseline + deltas[method]), abs=1e-10)
            assert row[f"DeltaS_{method}"] == pytest.approx(unit @ basis @ deltas[method], abs=1e-10)


def test_heldout_perturbation_changes_only_usefulness_not_routes(tmp_path, monkeypatch):
    config = _synthetic_config()
    basis = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.3]])
    _prepare_synthetic_inputs(tmp_path, [[1, 1, 0]])
    calls = _mock_model_boundary(monkeypatch, basis)
    pipeline.compute_training_features_and_routes(config, tmp_path)
    first = storage.read_records(tmp_path / "direction_records.parquet")
    first_routes = storage.read_records(tmp_path / "route_records.parquet")
    np.save(tmp_path / "features" / "heldout_head.npy", np.array([-0.8, 0.7]))
    pipeline.compute_training_features_and_routes(config, tmp_path)
    second = storage.read_records(tmp_path / "direction_records.parquet")
    second_routes = storage.read_records(tmp_path / "route_records.parquet")
    assert calls["build_gram"] == 2
    assert first_routes == second_routes
    protected = [
        "q_i",
        "w_Linear",
        "w_NCR",
        "utility_Linear",
        "utility_NCR",
        "S_GRPO",
        "S_Linear",
        "S_NCR",
        "DeltaS_Linear",
        "DeltaS_NCR",
        "route_difference_norm",
    ]
    for before, after in zip(first, second, strict=True):
        assert before["B_i"] != after["B_i"]
        for field in protected:
            assert before[field] == pytest.approx(after[field], abs=1e-12)


def test_reported_support_uses_same_measured_gram_as_grpo_and_usefulness(tmp_path, monkeypatch):
    config = _synthetic_config()
    basis = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.3]])
    gram = basis.T @ basis
    gram[2, 2] -= 1e-9  # Admissible numerical PSD roundoff; solver logs its repair.
    _prepare_synthetic_inputs(tmp_path, [[1, 1, 0]])
    _mock_model_boundary(monkeypatch, basis, gram=gram)
    pipeline.compute_training_features_and_routes(config, tmp_path)
    directions = storage.read_records(tmp_path / "direction_records.parquet")
    baseline, deltas = _coefficients_from_records(
        storage.read_records(tmp_path / "route_records.parquet"), directions, config
    )
    for i, row in enumerate(directions):
        norm = np.sqrt(gram[i, i])
        for method in ("Linear", "NCR"):
            assert row[f"S_{method}"] == pytest.approx(gram[i] @ (baseline + deltas[method]) / norm, abs=1e-12, rel=0)
            assert row[f"DeltaS_{method}"] == pytest.approx(gram[i] @ deltas[method] / norm, abs=1e-12, rel=0)


def test_all_homogeneous_groups_produce_empty_eligible_artifacts_and_undefined_report(tmp_path, monkeypatch):
    config = _synthetic_config()
    _prepare_synthetic_inputs(tmp_path, [[0, 0, 0], [1, 1, 1]])
    calls = _mock_model_boundary(monkeypatch, np.eye(3))
    artifacts = pipeline.compute_training_features_and_routes(config, tmp_path)
    assert all(path.is_file() for path in artifacts)
    assert calls == {"build_gram": 0, "tokenizer": 0}
    assert storage.read_records(tmp_path / "direction_records.parquet") == []
    assert storage.read_records(tmp_path / "solver_diagnostics.parquet") == []
    summary = storage.read_json(tmp_path / "eligibility_summary.json")
    assert summary["sampled_prompts"] == 2
    assert summary["all_zero_groups"] == summary["all_one_groups"] == 1
    assert summary["mixed_groups"] == summary["eligible_directions"] == 0
    assert all(row["status"] == "homogeneous" for row in storage.read_records(tmp_path / "route_records.parquet"))
    output = pipeline.compute_statistics(config, tmp_path)
    assert all(path.is_file() for path in output)
    result = storage.read_json(tmp_path / "correlations.json")
    assert result["populations"]["all_eligible"]["n_directions"] == 0
    assert result["populations"]["all_eligible"]["estimates"]["spearman"]["NCR"]["status"] == "undefined"
    assert (tmp_path / "report_round1.md").is_file()
