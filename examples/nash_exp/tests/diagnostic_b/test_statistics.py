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

"""CPU-only statistical contract tests with synthetic cached directions."""

import copy
import csv
import json
import warnings

import numpy as np
import pytest
from diagnostic_b import statistics
from scipy.stats import rankdata, spearmanr


def _records():
    records = []
    for prompt, successes in enumerate((2, 3, 1)):
        for rollout in range(successes):
            value = len(records) + 1
            records.append(
                {
                    "prompt_index": prompt,
                    "rollout_index": rollout,
                    "B_i": float((value * 3) % 7),
                    "S_GRPO": float(value),
                    "S_Linear": float(value**2),
                    "S_NCR": float((value * 2) % 5),
                    "group_success_count": successes,
                    "group_size": 4,
                    "route_difference_norm": 0.2 if prompt == 1 else 0.0,
                    "p_norm": 1.0,
                    "reward": 1,
                    "ncr_fallback": prompt == 0,
                }
            )
    return records


def test_delta_geometry_is_carried_into_report_and_plot_labels(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(statistics, "_plot_results", lambda rows, analysis, *args: observed.append(analysis))
    records = [row | {"feature_geometry": "delta_selected_token_gradient_proxy"} for row in _records()]
    result = statistics.analyze_records(records, tmp_path, 1, 10, 3, 0.1)
    assert result["feature_label"] == observed[0]["feature_label"] == "DelTA token-gradient proxy"
    report = (tmp_path / "report_round1.md").read_text()
    assert "DelTA token-gradient proxy" in report
    assert "not an exact expected-reward gradient" in report


def test_statistics_rejects_mixed_proxy_geometries(tmp_path):
    records = [row | {"feature_geometry": "delta_selected_token_gradient_proxy"} for row in _records()]
    records[0]["feature_geometry"] = "unprojected_output_path_lm_head_proxy"
    with pytest.raises(ValueError, match="different feature geometries"):
        statistics.analyze_records(records, tmp_path, 1, 10, 3, 0.1)


def test_spearman_uses_average_ranks_and_pearson_is_secondary():
    x, y = np.array([1.0, 1.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 3, 2, 4])[0, 1]
    assert statistics.correlation(x, y)["estimate"] == pytest.approx(expected)
    assert statistics.correlation(x, y, "pearson")["estimate"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert statistics.correlation(x, y, "spearman")["estimate"] != statistics.correlation(x, y, "pearson")["estimate"]


@pytest.mark.parametrize(
    ("x", "y", "reason"),
    [
        ([], [], "fewer_than_three_observations"),
        ([1, 2], [2, 1], "fewer_than_three_observations"),
        ([1, 1, 1], [1, 2, 3], "constant_array"),
        ([1, 2, 3], [0, 0, 0], "constant_array"),
        ([1, float("nan"), 3], [1, 2, 3], "nonfinite_observations"),
    ],
)
def test_undefined_correlations_are_explicit(x, y, reason, caplog):
    with caplog.at_level("INFO"):
        result = statistics.correlation(x, y)
    assert result["estimate"] is None
    assert result["status"] == "undefined"
    assert result["reason"] == reason
    assert reason in caplog.text


def test_prompt_cluster_bootstrap_matches_manual_paired_draws():
    records = _records()
    prompts, supports, usefulness = statistics._arrays(records)
    replicates, seed = 80, 31
    result = statistics.paired_cluster_bootstrap(prompts, supports, usefulness, replicates, seed)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(prompts == prompt) for prompt in np.unique(prompts)]
    samples = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(replicates):
            draw = rng.integers(0, len(groups), size=len(groups))
            # Unequal group sizes must carry every direction in each selected prompt.
            indices = np.concatenate([groups[index] for index in draw])
            samples.append([spearmanr(supports[indices, j], usefulness[indices]).statistic for j in range(3)])
    samples = np.array(samples)
    for j, method in enumerate(statistics.METHODS):
        finite = samples[np.isfinite(samples[:, j]), j]
        actual = result["estimates"]["spearman"][method]
        np.testing.assert_allclose(actual["ci95"], np.percentile(finite, [2.5, 97.5]))
        assert actual["bootstrap_valid"] == len(finite)
        assert actual["bootstrap_undefined"] == replicates - len(finite)
    for left, right in statistics.DIFFERENCES:
        differences = samples[:, statistics.METHODS.index(left)] - samples[:, statistics.METHODS.index(right)]
        differences = differences[np.isfinite(differences)]
        actual = result["differences"]["spearman"][f"{left}-{right}"]
        np.testing.assert_allclose(actual["ci95"], np.percentile(differences, [2.5, 97.5]), atol=1e-15)
    assert result["bootstrap"]["paired_across_methods_and_differences"]
    again = statistics.paired_cluster_bootstrap(prompts, supports, usefulness, replicates, seed)
    assert again == result


def test_single_cluster_has_no_fabricated_confidence_interval():
    result = statistics.paired_cluster_bootstrap([0, 0, 0], [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [1, 2, 3], 20, 1)
    estimate = result["estimates"]["spearman"]["GRPO"]
    assert estimate["estimate"] == pytest.approx(1.0)
    assert estimate["ci95"] == [None, None]
    assert estimate["ci_reason"] == "fewer_than_two_prompt_clusters"


def test_within_prompt_sensitivity_centers_ranks_and_drops_singletons():
    prompts, supports, usefulness = statistics._arrays(_records())
    result = statistics.paired_cluster_bootstrap(prompts, supports, usefulness, 30, 2, within_prompt=True)
    assert result["n_prompts"] == 2
    assert result["n_directions"] == 5
    expected_x, expected_y = [], []
    for prompt in ("0", "1"):
        mask = prompts == prompt
        size = mask.sum()
        expected_x.extend((rankdata(supports[mask, 2]) - (size + 1) / 2) / size)
        expected_y.extend((rankdata(usefulness[mask]) - (size + 1) / 2) / size)
    expected = np.corrcoef(expected_x, expected_y)[0, 1]
    assert result["estimates"]["within_prompt_rank"]["NCR"]["estimate"] == pytest.approx(expected)


def test_subsets_do_not_depend_on_heldout_values_and_fallbacks_are_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(statistics, "_plot_results", lambda *args: None)
    records = _records()
    first = statistics.analyze_records(records, tmp_path / "first", 1, 15, 10, 0.1)
    changed = copy.deepcopy(records)
    for row in changed:
        row["B_i"] = -100 * row["B_i"] + 25
    second = statistics.analyze_records(changed, tmp_path / "second", 1, 15, 10, 0.1)
    for name, expected_directions in (("all_eligible", 6), ("selector_capable", 5), ("nash_linear_active", 3)):
        a, b = first["populations"][name], second["populations"][name]
        assert a["n_directions"] == b["n_directions"] == expected_directions
        assert a["n_prompts"] == b["n_prompts"]
        assert a["bootstrap"]["draw_sha256"] == b["bootstrap"]["draw_sha256"]
    assert first["counts_from_direction_records"]["flags_direction_counts"]["ncr_fallback"] == 2
    assert first["counts_from_direction_records"]["flags_prompt_counts"]["ncr_fallback"] == 1


def test_nonfinite_and_feature_exclusions_are_common_to_all_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(statistics, "_plot_results", lambda *args: None)
    records = _records()
    records[0]["S_NCR"] = float("nan")
    records[1]["p_norm"] = 0
    records[2]["B_i"] = float("inf")
    result = statistics.analyze_records(records, tmp_path, 1, 10, 3, 0.1)
    counts = result["counts_from_direction_records"]
    assert counts["excluded_directions"] == 3
    assert counts["eligible_successful_directions"] == 3
    for method in statistics.METHODS:
        assert result["populations"]["all_eligible"]["estimates"]["spearman"][method]["n"] == 3


@pytest.mark.parametrize("empty", [False, True])
def test_report_plots_and_tables_are_real_artifacts_with_metadata(tmp_path, empty):
    metadata = {
        "sampled_prompts": 6,
        "all_zero_groups": 2,
        "all_one_groups": 1,
        "heldout_successes": 0 if empty else 5,
        "heldout_success_rate": 0.0 if empty else 0.125,
        "solver_residual_summary": {"max": 1e-7},
        "runtime_seconds": {"generation": 12.5, "features": 8.0},
        "generation_tokens": 600,
    }
    result = statistics.analyze_records([] if empty else _records(), tmp_path, "round1", 12, 3, 0.1, metadata)
    loaded = json.loads((tmp_path / "correlations.json").read_text())
    assert loaded == result
    assert result["upstream_summary"] == metadata
    report = (tmp_path / "report_round1.md").read_text()
    assert "not proof of the transfer theorem" in report
    assert "heldout_success_rate" in report
    assert "solver_residual_summary.max" in report
    assert "runtime_seconds.generation" in report
    assert "generation_tokens" in (tmp_path / "correlations.tex").read_text().replace(r"\_", "_")
    with (tmp_path / "correlations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["population"] for row in rows} == {
        "all_eligible",
        "selector_capable",
        "nash_linear_active",
        "within_prompt_ranks",
    }
    for figure in ("support_vs_usefulness", "correlation_comparison", "rank_supplement"):
        pdf = tmp_path / f"diagnostic_b_{figure}_round1.pdf"
        png = tmp_path / f"diagnostic_b_{figure}_round1.png"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if empty:
        assert result["populations"]["all_eligible"]["n_directions"] == 0
        assert all(row["status"] == "undefined" for row in rows)


def test_record_contract_rejects_duplicates_and_unknown_group_distance():
    records = _records()
    with pytest.raises(ValueError, match="Duplicate"):
        statistics._prepare_records(records + [records[0]])
    records[0]["route_difference_norm"] = float("nan")
    with pytest.raises(ValueError, match="route difference norm"):
        statistics._prepare_records(records)


@pytest.mark.parametrize("replicates,seed", [(0, 1), (5, None), (5, -1), (2.5, 1)])
def test_bootstrap_configuration_must_be_explicit_and_valid(replicates, seed):
    with pytest.raises(ValueError):
        statistics.paired_cluster_bootstrap([], np.empty((0, 3)), [], replicates, seed)
