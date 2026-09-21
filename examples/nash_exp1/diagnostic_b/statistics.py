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

"""Offline, common-population analysis of cached Diagnostic-B direction records.

The bootstrap sampling unit is a training prompt, never a direction. A sampled
prompt carries *all* its retained directions, and each replicate uses exactly the
same rows for every mechanism and paired difference. No model or data is loaded.
"""

import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

LOGGER = logging.getLogger(__name__)
METHODS = ("GRPO", "Linear", "NCR")
DIFFERENCES = (("NCR", "GRPO"), ("Linear", "GRPO"), ("NCR", "Linear"))
POPULATIONS = {
    "all_eligible": "All eligible mixed-group successes (primary)",
    "selector_capable": "Selector-capable groups: m >= 2 (secondary)",
    "nash_linear_active": "Nash/linear-active groups (diagnostic)",
}
REQUIRED_FIELDS = (
    "prompt_index",
    "rollout_index",
    "B_i",
    "S_GRPO",
    "S_Linear",
    "S_NCR",
    "group_success_count",
    "route_difference_norm",
)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float | None, str | None]:
    if len(x) < 3:
        return None, "fewer_than_three_observations"
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None, "nonfinite_observations"
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "constant_array"
    if method == "spearman":
        x, y = rankdata(x, method="average"), rankdata(y, method="average")
    # Scale before centering to avoid overflow for finite large proxy values.
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    x, y = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0 or not np.isfinite(norm):
        return None, "degenerate_numerical_variance"
    return float(np.clip(np.dot(x, y) / norm, -1.0, 1.0)), None


def correlation(x: Any, y: Any, method: str = "spearman") -> dict:
    """Return a finite coefficient or an explicit undefined result; never drop rows."""
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("correlation requires equal-length one-dimensional arrays")
    estimate, reason = _correlation(x, y, method)
    if reason:
        LOGGER.info("Undefined %s correlation: %s (n=%d)", method, reason, len(x))
    return {"estimate": estimate, "status": "undefined" if reason else "defined", "reason": reason, "n": len(x)}


def _within_prompt_ranks(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Centered fractional average ranks: (rank - (n+1)/2)/n within each prompt."""
    transformed = np.empty_like(values, dtype=float)
    for rows in groups:
        transformed[rows] = (rankdata(values[rows], method="average") - (len(rows) + 1) / 2) / len(rows)
    return transformed


def paired_cluster_bootstrap(
    prompt_indices: Any,
    supports: Any,
    usefulness: Any,
    replicates: int,
    seed: int,
    *,
    within_prompt: bool = False,
) -> dict:
    """Analyze aligned arrays, with supports[:, 0:3] ordered GRPO, Linear, NCR.

    Percentile intervals use paired prompt-cluster draws. Undefined replicates
    are counted explicitly. A CI requires at least two observed prompt clusters.
    For the declared sensitivity analysis, within-prompt centered fractional
    ranks are computed first and their pooled Pearson association is reported.
    """
    if not isinstance(replicates, int | np.integer) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    if not isinstance(seed, int | np.integer) or isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap seed must be an explicit nonnegative integer")
    prompt_indices = np.asarray(prompt_indices)
    supports, usefulness = np.asarray(supports, dtype=float), np.asarray(usefulness, dtype=float)
    if prompt_indices.ndim != 1 or usefulness.ndim != 1 or supports.shape != (len(usefulness), 3):
        raise ValueError("expected prompt_indices (n,), usefulness (n,), supports (n, 3)")
    if len(prompt_indices) != len(usefulness):
        raise ValueError("prompt indices and observations must be aligned")
    if not np.all(np.isfinite(supports)) or not np.all(np.isfinite(usefulness)):
        raise ValueError("all bootstrap methods require the same finite common observations")
    unique_prompts = np.unique(prompt_indices)
    groups = [np.flatnonzero(prompt_indices == key) for key in unique_prompts]
    if within_prompt:
        retained = [rows for rows in groups if len(rows) >= 2]
        rows = np.concatenate(retained) if retained else np.array([], dtype=int)
        prompt_indices, supports, usefulness = prompt_indices[rows], supports[rows], usefulness[rows]
        unique_prompts = np.unique(prompt_indices)
        groups = [np.flatnonzero(prompt_indices == key) for key in unique_prompts]
        supports = np.column_stack([_within_prompt_ranks(supports[:, j], groups) for j in range(3)])
        usefulness = _within_prompt_ranks(usefulness, groups)
    metrics = ("within_prompt_rank",) if within_prompt else ("spearman", "pearson")
    estimates, samples = {}, {}
    for metric in metrics:
        estimates[metric], samples[metric] = {}, np.full((replicates, 3), np.nan)
        for j, method in enumerate(METHODS):
            estimates[metric][method] = correlation(
                supports[:, j], usefulness, "spearman" if metric == "spearman" else "pearson"
            )
    rng, draw_hash = np.random.default_rng(seed), hashlib.sha256()
    if len(groups) >= 2:
        for replicate in range(replicates):
            draw = rng.integers(0, len(groups), size=len(groups))
            draw_hash.update(draw.astype("<i8").tobytes())
            rows = np.concatenate([groups[int(i)] for i in draw])
            for metric in metrics:
                for j in range(3):
                    value, _ = _correlation(supports[rows, j], usefulness[rows], metric)
                    if value is not None:
                        samples[metric][replicate, j] = value
    interval_reason = "fewer_than_two_prompt_clusters" if len(groups) < 2 else None

    def interval(values: np.ndarray) -> dict:
        valid = values[np.isfinite(values)]
        reason = interval_reason or ("no_defined_bootstrap_replicates" if not len(valid) else None)
        if reason:
            LOGGER.info("Undefined prompt-cluster interval: %s", reason)
        return {
            "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else [None, None],
            "bootstrap_valid": len(valid),
            "bootstrap_undefined": replicates - len(valid),
            "ci_reason": reason,
        }

    differences = {}
    for metric in metrics:
        differences[metric] = {}
        for j, method in enumerate(METHODS):
            estimates[metric][method].update(interval(samples[metric][:, j]))
        for left, right in DIFFERENCES:
            left_idx, right_idx = METHODS.index(left), METHODS.index(right)
            a, b = estimates[metric][left]["estimate"], estimates[metric][right]["estimate"]
            reason = "undefined_component_correlation" if a is None or b is None else None
            differences[metric][f"{left}-{right}"] = {
                "estimate": None if reason else a - b,
                "status": "undefined" if reason else "defined",
                "reason": reason,
                "n": len(usefulness),
                **interval(samples[metric][:, left_idx] - samples[metric][:, right_idx]),
            }
    return {
        "n_directions": len(usefulness),
        "n_prompts": len(groups),
        "estimates": estimates,
        "differences": differences,
        "bootstrap": {
            "unit": "training_prompt",
            "paired_across_methods_and_differences": True,
            "replicates": int(replicates),
            "seed": int(seed),
            "confidence_level": 0.95,
            "interval_method": "percentile; undefined replicates excluded and counted",
            "draw_sha256": draw_hash.hexdigest(),
        },
    }


def _prepare_records(records: Any) -> tuple[list[dict], dict]:
    records = records.to_dict(orient="records") if hasattr(records, "to_dict") else list(records)
    kept, pairs = [], set()
    exclusions, flagged_directions, flagged_prompts = Counter(), Counter(), {}
    group_properties = {}
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Direction record {index} lacks required fields: {missing}")
        pair = (record["prompt_index"], record["rollout_index"])
        if pair in pairs:
            raise ValueError(f"Duplicate prompt/rollout direction: {pair}")
        pairs.add(pair)
        successes, distance = int(record["group_success_count"]), float(record["route_difference_norm"])
        if successes != record["group_success_count"] or successes < 0 or not np.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid group success count or route difference norm at {pair}")
        properties = (successes, distance)
        previous = group_properties.setdefault(record["prompt_index"], properties)
        if previous != properties:
            raise ValueError(f"Group success count and route difference norm must be constant within prompt {pair[0]}")
        for name, value in record.items():
            if any(part in name.lower() for part in ("fallback", "failed", "failure")):
                if isinstance(value, bool | np.bool_) and value:
                    flagged_directions[name] += 1
                    flagged_prompts.setdefault(name, set()).add(record["prompt_index"])
        rejected = False
        group_size = record.get("group_size", record.get("n_rollout"))
        if record.get("reward", 1) != 1 or successes == 0 or (group_size is not None and successes >= group_size):
            exclusions["unsuccessful_or_homogeneous"] += 1
            rejected = True
        norm = record.get("raw_feature_norm", record.get("p_norm"))
        if norm is not None and (not np.isfinite(float(norm)) or float(norm) <= 0):
            exclusions["zero_or_nonfinite_raw_feature_norm"] += 1
            rejected = True
        if not np.isfinite(float(record["B_i"])):
            exclusions["nonfinite_usefulness"] += 1
            rejected = True
        if not all(np.isfinite(float(record[f"S_{method}"])) for method in METHODS):
            exclusions["nonfinite_support_any_method"] += 1
            rejected = True
        if not rejected:
            kept.append(record)
    counts = {
        "input_direction_records": len(records),
        "eligible_successful_directions": len(kept),
        "eligible_prompt_groups": len({record["prompt_index"] for record in kept}),
        "excluded_directions": len(records) - len(kept),
        "exclusion_reasons_nonexclusive": dict(exclusions),
        "flags_direction_counts": dict(flagged_directions),
        "flags_prompt_counts": {name: len(prompts) for name, prompts in flagged_prompts.items()},
    }
    LOGGER.info("Diagnostic-B common-population eligibility counts: %s", counts)
    return kept, counts


def _arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # String IDs support integer and textual upstream prompt identifiers uniformly.
    prompts = np.asarray([str(row["prompt_index"]) for row in records], dtype=str)
    supports = np.asarray([[row[f"S_{method}"] for method in METHODS] for row in records], dtype=float).reshape(-1, 3)
    usefulness = np.asarray([row["B_i"] for row in records], dtype=float)
    return prompts, supports, usefulness


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _format_estimate(result: dict) -> str:
    if result["estimate"] is None:
        reason = result["reason"].replace("_", " ")
        return f"undefined\n({reason})"
    interval = result["ci95"]
    ci = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval[0] is not None else "CI undefined"
    return f"{result['estimate']:.3f} {ci}"


def _table_rows(analysis: dict) -> list[dict]:
    rows = []
    populations = {**analysis["populations"], "within_prompt_ranks": analysis["within_prompt_sensitivity"]}
    for population, result in populations.items():
        for category in ("estimates", "differences"):
            for metric, comparisons in result[category].items():
                for comparison, item in comparisons.items():
                    rows.append(
                        {
                            "population": population,
                            "metric": metric,
                            "kind": category,
                            "comparison": comparison,
                            "estimate": item["estimate"],
                            "ci95_low": item["ci95"][0],
                            "ci95_high": item["ci95"][1],
                            "status": item["status"],
                            "reason": item["reason"],
                            "ci_reason": item["ci_reason"],
                            "n_directions": result["n_directions"],
                            "n_prompts": result["n_prompts"],
                            "bootstrap_valid": item["bootstrap_valid"],
                            "bootstrap_undefined": item["bootstrap_undefined"],
                        }
                    )
    return rows


def _flatten_metadata(value: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict) and item:
            rows.extend(_flatten_metadata(item, name))
        else:
            rows.append((name, json.dumps(_json_safe(item), ensure_ascii=False)))
    return rows


def _latex_escape(value: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    replacements.update({"$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"})
    return "".join(replacements.get(char, char) for char in str(value))


def _write_tables(analysis: dict, output_dir: Path, round_label: str) -> None:
    rows = _table_rows(analysis)
    with (output_dir / "correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "correlations.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    metadata = {
        key: value for key, value in analysis.items() if key not in ("populations", "within_prompt_sensitivity")
    }
    metadata_rows = _flatten_metadata(metadata)
    with (output_dir / "analysis_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        writer.writerows(metadata_rows)
    markdown = [
        f"# Diagnostic B: {round_label}",
        "",
        analysis["interpretation"],
        "",
        "Pooled Spearman (average ranks for ties) is primary; Pearson is a secondary linear-alignment check.",
        "Intervals use paired training-prompt cluster bootstrap draws and the percentile method.",
        "Undefined bootstrap replicates are excluded from interval quantiles and counted in the CSV/JSON.",
        "A confidence interval requires at least two training prompt clusters.",
        "",
        "| Population | Metric | Method/difference | Estimate | 95% CI | Directions | Prompts | Status |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    latex = [
        "% Requires longtable. Undefined coefficients are retained explicitly.",
        r"\begin{longtable}{lllrrrl}",
        r"Population & Metric & Method/difference & Estimate & CI lower & CI upper & Status \\",
        r"\hline",
    ]
    for row in rows:
        estimate = "undefined" if row["estimate"] is None else f"{row['estimate']:.4f}"
        lo = "undefined" if row["ci95_low"] is None else f"{row['ci95_low']:.4f}"
        hi = "undefined" if row["ci95_high"] is None else f"{row['ci95_high']:.4f}"
        reason = row["reason"] or row["ci_reason"] or row["status"]
        markdown.append(
            f"| {row['population']} | {row['metric']} | {row['comparison']} | {estimate} | [{lo}, {hi}] | "
            f"{row['n_directions']} | {row['n_prompts']} | {reason} |"
        )
        latex.append(
            " & ".join(
                _latex_escape(item)
                for item in (row["population"], row["metric"], row["comparison"], estimate, lo, hi, reason)
            )
            + r" \\"
        )
    latex.extend([r"\end{longtable}", "", r"\begin{longtable}{p{0.45\linewidth}p{0.5\linewidth}}"])
    latex.extend([r"Metadata field & Value \\", r"\hline"])
    markdown.extend(
        [
            "",
            "## Prompt-stratified sensitivity",
            "",
            analysis["within_prompt_rank_definition"],
            "",
            "## Eligibility, fallbacks, held-out outcomes, runtime, and configuration",
            "",
            "Upstream metadata are reported as supplied; absent quantities are not inferred as zero.",
            "Exclusion-reason counts may overlap. Fallback directions remain in the common population.",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )
    for key, value in metadata_rows:
        markdown.append(f"| {key.replace('|', '&#124;')} | {value.replace('|', '&#124;').replace(chr(10), ' ')} |")
        latex.append(f"{_latex_escape(key)} & {_latex_escape(value)}" + r" \\")
    latex.append(r"\end{longtable}")
    text = "\n".join(markdown) + "\n"
    (output_dir / f"report_{round_label}.md").write_text(text, encoding="utf-8")
    (output_dir / "correlations.md").write_text(text, encoding="utf-8")
    (output_dir / "correlations.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def _plot_results(records: list[dict], analysis: dict, output_dir: Path, round_label: str) -> None:
    # Import lazily so numeric-only callers need neither plotting nor display packages.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, supports, usefulness = _arrays(records)
    primary = analysis["populations"]["all_eligible"]
    colors = ("#555555", "#0072B2", "#D55E00")
    feature_label = analysis["feature_label"]

    def save(fig: Any, stem: str) -> None:
        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        for ranked in (False, True):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
            for j, (method, axis) in enumerate(zip(METHODS, axes, strict=True)):
                x = rankdata(supports[:, j], method="average") if ranked else supports[:, j]
                y = rankdata(usefulness, method="average") if ranked else usefulness
                axis.scatter(x, y, s=14, alpha=0.4, linewidths=0, color=colors[j], rasterized=True)
                if not ranked:
                    axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
                    axis.axvline(0, color="black", linewidth=0.6, alpha=0.5)
                axis.set_title(method)
                axis.set_xlabel(f"{feature_label} support rank" if ranked else feature_label + r" support $S_i$")
                rho, pearson = primary["estimates"]["spearman"][method], primary["estimates"]["pearson"][method]
                annotation = (
                    f"Spearman rho: {_format_estimate(rho)}\nPearson r: {pearson['estimate']:.3f}"
                    if pearson["estimate"] is not None
                    else f"Spearman rho: {_format_estimate(rho)}\nPearson r: undefined"
                )
                annotation += f"\nDirections: {primary['n_directions']}; prompts: {primary['n_prompts']}"
                axis.text(
                    0.02,
                    0.98,
                    annotation,
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                if not len(records):
                    axis.text(0.5, 0.4, "No eligible directions", transform=axis.transAxes, ha="center")
                axis.grid(alpha=0.15)
            axes[0].set_ylabel(
                f"{feature_label} usefulness rank" if ranked else feature_label + r" held-out usefulness $B_i$"
            )
            title = "Supplementary pooled average ranks" if ranked else "Support versus held-out usefulness"
            fig.suptitle(f"Diagnostic B, {round_label}: {title}")
            fig.tight_layout()
            kind = "rank_supplement" if ranked else "support_vs_usefulness"
            save(fig, f"diagnostic_b_{kind}_{round_label}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for axis, category, title in zip(
            axes, ("estimates", "differences"), ("Spearman correlations", "Paired Spearman differences"), strict=True
        ):
            values = primary[category]["spearman"]
            for i, (label, result) in enumerate(values.items()):
                center, interval = result["estimate"], result["ci95"]
                if center is None:
                    axis.text(0, i, "undefined", va="center", ha="center", fontsize=9)
                else:
                    if interval[0] is not None:
                        axis.hlines(i, interval[0], interval[1], color=colors[i], linewidth=2)
                        axis.plot(interval, [i, i], "|", color=colors[i])
                    axis.plot(center, i, "o", color=colors[i])
            axis.set_yticks(range(len(values)), list(values))
            axis.set_ylim(-0.6, len(values) - 0.4)
            axis.invert_yaxis()
            axis.axvline(0, color="black", linewidth=0.6, alpha=0.6)
            axis.set_title(title)
            axis.set_xlabel("Estimate and 95% paired prompt-cluster interval")
            axis.grid(axis="x", alpha=0.15)
        fig.suptitle(f"Diagnostic B, {round_label}: primary common population")
        fig.tight_layout()
        save(fig, f"diagnostic_b_correlation_comparison_{round_label}")


def analyze_records(
    records: Any,
    output_dir: str | Path,
    round_id: str | int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    active_tolerance: float,
    summary: dict | None = None,
) -> dict:
    """Write all statistical tables, metadata, plots, and a qualified analysis report.

    Records are successful mixed-group directions emitted by the upstream feature
    stage. If present, reward, group_size/n_rollout, and raw_feature_norm/p_norm
    are also checked. Nonfinite values exclude a direction from *all* methods.
    Missing sampled/homogeneous/solver/held-out/runtime counts cannot be recovered
    from successful direction records; supply them in ``summary`` for the report.
    Every experiment seed, bootstrap count, and active tolerance is caller-owned.
    """
    if active_tolerance is None or not np.isfinite(active_tolerance) or active_tolerance < 0:
        raise ValueError("active_tolerance must be an explicit finite nonnegative value")
    round_label = str(round_id)
    if not round_label.startswith("round"):
        round_label = f"round{round_label}"
    if not round_label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("round_id must be a simple round label, not a path")
    from .config import FEATURE_LABELS

    # Reject mixed geometries even if a direction is subsequently excluded.
    records = records.to_dict(orient="records") if hasattr(records, "to_dict") else list(records)
    geometries = {row["feature_geometry"] for row in records if row.get("feature_geometry")}
    if summary and summary.get("feature_geometry"):
        geometries.add(summary["feature_geometry"])
    if len(geometries) > 1:
        raise ValueError("Cannot compare direction records from different feature geometries")
    geometry = next(iter(geometries), "unspecified_proxy")
    feature_label = FEATURE_LABELS.get(geometry, "Feature proxy")
    records, counts = _prepare_records(records)
    populations = {
        "all_eligible": records,
        "selector_capable": [row for row in records if row["group_success_count"] >= 2],
        "nash_linear_active": [row for row in records if row["route_difference_norm"] > active_tolerance],
    }
    analysis = {
        "round_id": round_label,
        "feature_geometry": geometry,
        "feature_label": feature_label,
        "interpretation": (
            f"This diagnostic measures alignment between local {feature_label} support and an independently "
            "estimated reward-weighted held-out token sum in the same proxy geometry. The DelTA proxy, when "
            "selected, discards output-row identities and other vocabulary rows; its held-out sum is not an "
            "exact expected-reward gradient in a shared parameter block. Correlation is not proof of the transfer "
            "theorem or "
            "evidence of a causal held-out accuracy improvement. Null and adverse outcomes are retained."
        ),
        "active_tolerance": float(active_tolerance),
        "subset_selection": "Fixed group success count and route difference norm; never usefulness or method rank.",
        "population_labels": POPULATIONS,
        "counts_from_direction_records": counts,
        "upstream_summary": summary or {},
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "within_prompt_rank_definition": (
            "For prompts with at least two retained successful directions, replace each support and usefulness "
            "value by (average within-prompt rank - (n+1)/2)/n. Report the pooled Pearson correlation of these "
            "centered fractional ranks, with the same paired prompt-cluster bootstrap. This is a sensitivity "
            "analysis; pooled Spearman remains primary."
        ),
        "populations": {},
    }
    for name, population in populations.items():
        analysis["populations"][name] = paired_cluster_bootstrap(
            *_arrays(population), replicates=bootstrap_replicates, seed=bootstrap_seed
        )
    analysis["within_prompt_sensitivity"] = paired_cluster_bootstrap(
        *_arrays(records), replicates=bootstrap_replicates, seed=bootstrap_seed, within_prompt=True
    )
    analysis = _json_safe(analysis)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(analysis, output_dir, round_label)
    _plot_results(records, analysis, output_dir, round_label)
    LOGGER.info("Diagnostic-B statistics and publication figures written to %s", output_dir)
    return analysis
