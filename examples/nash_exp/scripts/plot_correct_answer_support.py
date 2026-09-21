#!/usr/bin/env python3
"""Replot saved correct-answer directions on CPU; no generation or solver replay.

Dependencies: numpy scipy pandas pyarrow matplotlib pyyaml.
Run from nash_exp: python scripts/plot_correct_answer_support.py
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
import yaml

METHODS = ("GRPO", "Linear", "NCR")
LABELS = ("GRPO", "Linear routing", "Nash routing (NCR)")
COLORS = ("#475569", "#007EA8", "#CE5A25")
EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = EXPERIMENT / "artifacts/diagnostic_b/round1_qwen25math7b_math96_test32_p1024_g3072"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_records(root, frame, config):
    """Check source identities and reconstruct every dot product from saved Grams."""
    state = read_json(root / "stage_state.json")
    hashes = {}
    for stage in ("prepare", "verify", "heldout_features", "training_features_and_routes", "statistics"):
        for name, expected in state["stages"][stage]["outputs"].items():
            actual = sha256(root / name)
            if actual != expected:
                raise ValueError(f"Saved artifact changed: {name}")
            hashes[name] = actual
    if frame.duplicated(["prompt_index", "rollout_index"]).any():
        raise ValueError("A correct answer appears more than once")
    numeric = frame[["p_norm", "B_i", *("S_" + method for method in METHODS)]].to_numpy()
    if not np.isfinite(numeric).all() or not (frame.p_norm > 0).all():
        raise ValueError("Nonfinite or zero-norm direction in saved analysis set")
    if not (frame.reward.eq(1) & frame.group_success_count.gt(0)
            & frame.group_success_count.lt(frame.group_size)).all():
        raise ValueError("Records must be correct answers from mixed groups")
    if set(frame.feature_geometry) != {"delta_selected_token_gradient_proxy"}:
        raise ValueError("This plot is labeled for the recorded DelTA feature geometry")

    verification = pd.read_parquet(root / "verifier_results.parquet")
    train = verification[verification.split.eq("train")]
    counts = train.groupby("prompt_index").reward.agg(["sum", "count"])
    mixed = counts.index[(counts["sum"] > 0) & (counts["sum"] < counts["count"])]
    expected = train[train.prompt_index.isin(mixed) & train.reward.eq(1)]
    expected_keys = set(zip(expected.prompt_index, expected.rollout_index))
    actual_keys = set(zip(frame.prompt_index, frame.rollout_index))
    if expected_keys != actual_keys:
        raise ValueError("Plot records do not cover exactly the saved mixed-group successes")
    routes = pd.read_parquet(root / "route_records.parquet")
    feature_manifest = read_json(root / "feature_manifest.json")
    gram_hashes = {item["prompt_index"]: item["sha256"] for item in feature_manifest["training_group_files"]}
    heldout = read_json(root / "heldout_feature_manifest.json")
    heldout_verification = verification[verification.split.eq("heldout")]
    if (heldout["denominator"] != len(heldout_verification)
            or heldout["heldout_successes"] != int(heldout_verification.reward.sum())
            or heldout["response_aggregation"] != "token_sum"):
        raise ValueError("Held-out estimator metadata disagrees with saved rewards")
    max_error = {key: 0.0 for key in ("p_norm", "B_i", "S_GRPO", "S_Linear", "S_NCR")}

    def compare(name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7, err_msg=name)
        max_error[name] = max(max_error[name], float(np.max(np.abs(actual - expected))))

    for prompt_index, records in frame.groupby("prompt_index", sort=True):
        folder = root / "features" / f"train_group_{prompt_index:05d}"
        meta = read_json(folder / "metadata.json")
        if sha256(folder / "gram.npz") != meta["sha256"] or meta["sha256"] != gram_hashes[prompt_index]:
            raise ValueError(f"Changed Gram for prompt {prompt_index}")
        with np.load(folder / "gram.npz", allow_pickle=False) as archive:
            gram, cross = archive["gram"], archive["heldout_cross"]
        gram = (gram + gram.T) / 2
        positives, negatives = meta["positive_info"], meta["negative_info"]
        records = records.set_index("rollout_index").loc[[item["rollout_index"] for item in positives]]
        rewards = train[train.prompt_index.eq(prompt_index)].sort_values("rollout_index").reward.to_numpy()
        advantages = (rewards - rewards.mean()) / (rewards.std(ddof=0) + config["routing"]["epsilon"])
        a = np.array([advantages[item["rollout_index"]] / len(rewards) for item in positives])
        beta = np.array([-advantages[item["rollout_index"]] / len(rewards)
                         * item["length"] / item["response_length"] for item in negatives])
        baseline = np.concatenate((a, -beta))
        m = len(positives)
        # The recorded pipeline converts each FP32 norm to a Python float before
        # division. Preserve that FP64 arithmetic across NumPy 1.x and 2.x.
        norms = np.sqrt(np.maximum(np.diag(gram)[:m], 0)).astype(np.float64)
        compare("p_norm", norms, records.p_norm.to_numpy())
        compare("B_i", cross[:m] / norms, records.B_i.to_numpy())
        baseline_support = (gram[:m] @ baseline) / norms
        compare("S_GRPO", baseline_support, records.S_GRPO.to_numpy())
        for method in METHODS[1:]:
            route = routes[routes.prompt_index.eq(prompt_index) & routes.method.eq(method)].iloc[0]
            w = np.asarray(json.loads(route.positive_shares_json))
            segments = json.loads(route.negative_segments_json)
            if [{key: segment[key] for key in negative}
                    for negative, segment in zip(negatives, segments, strict=True)] != negatives:
                raise ValueError("Saved route segment ordering differs from Gram ordering")
            refunds = np.array([segment["refund"] for segment in segments])
            delta = np.concatenate((a.sum() * (w - a / a.sum()), beta * refunds))
            support = baseline_support + (gram[:m] @ delta) / norms
            compare("S_" + method, support, records["S_" + method].to_numpy())
    return {"artifact_sha256": hashes, "dot_product_max_absolute_errors": max_error,
            "checked_prompt_groups": int(frame.prompt_index.nunique()),
            "checked_correct_answers": len(frame),
            "scope": "Cached Gram algebra and saved verifier labels; no feature extraction or solver rerun."}


def paired_correlations(values):
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.sum(centered * centered, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (centered[:, :3].T @ centered[:, 3]) / (scale[:3] * scale[3])


def correlations(frame, seed, replicates):
    values = frame[[*("S_" + method for method in METHODS), "B_i"]].to_numpy()
    groups = [np.flatnonzero(frame.prompt_index.to_numpy() == prompt)
              for prompt in np.sort(frame.prompt_index.unique())]
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(replicates, len(groups)))
    samples = {name: np.empty((replicates, 3)) for name in ("spearman", "pearson")}
    for index, draw in enumerate(draws):
        sample = values[np.concatenate([groups[group] for group in draw])]
        samples["pearson"][index] = paired_correlations(sample)
        samples["spearman"][index] = paired_correlations(rankdata(sample, axis=0, method="average"))
    rows = []
    for index, method in enumerate(METHODS):
        row = {"method": method, "correct_answers": len(frame), "prompt_groups": len(groups)}
        for metric, function in (("spearman", spearmanr), ("pearson", pearsonr)):
            row[metric] = float(function(values[:, index], values[:, 3]).statistic)
            valid = samples[metric][:, index]
            valid = valid[np.isfinite(valid)]
            if len(valid) == 0:
                raise ValueError("No defined bootstrap correlations")
            lo, hi = np.quantile(valid, [0.025, 0.975])
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = float(lo), float(hi)
            row[metric + "_bootstrap_valid"] = len(valid)
        rows.append(row)
    differences = {}
    for metric, sample in samples.items():
        differences[metric] = {}
        for name, left, right in (("NCR-GRPO", 2, 0), ("NCR-Linear", 2, 1)):
            delta = sample[:, left] - sample[:, right]
            differences[metric][name] = {
                "estimate": rows[left][metric] - rows[right][metric],
                "ci95": np.quantile(delta[np.isfinite(delta)], [0.025, 0.975]).tolist(),
            }
    return rows, differences


def make_plots(frame, rows, output, subtitle):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    xs = frame[["S_" + method for method in METHODS]].to_numpy()
    ys = frame.B_i.to_numpy()

    def limits(values):
        low, high = min(0, float(values.min())), max(0, float(values.max()))
        margin = max((high - low) * 0.055, 1e-6)
        return low - margin, high + margin

    def panel(ax, index):
        row = rows[index]
        ax.scatter(xs[:, index], ys, s=13, color=COLORS[index], alpha=0.48, linewidths=0)
        ax.axvline(0, color="#a0a0a0", linewidth=0.9, zorder=0)
        ax.axhline(0, color="#a0a0a0", linewidth=0.9, zorder=0)
        ax.set(xlim=limits(xs), ylim=limits(ys), axisbelow=True)
        ax.grid(color="#e8e8e8", linewidth=0.65)
        ax.set_title(LABELS[index] + "\n" + rf"Spearman $\rho$ = {row['spearman']:.4f}"
                     + "     " + rf"Pearson $r$ = {row['pearson']:.4f}", fontsize=13, pad=15)
        direction = (r"\mathbf{g}", r"(\mathbf{g}+\delta_{\mathrm{Linear}})",
                     r"(\mathbf{g}+\delta_{\mathrm{NCR}})")[index]
        ax.set_xlabel(r"Support $S_i=\mathbf{v}_i^\top " + direction + "$", labelpad=9)
        ax.text(0.5, -0.23,
                rf"95% prompt-bootstrap CI:  $\rho$ [{row['spearman_ci95_low']:.3f}, {row['spearman_ci95_high']:.3f}]"
                + "\n" + rf"$r$ [{row['pearson_ci95_low']:.3f}, {row['pearson_ci95_high']:.3f}]",
                ha="center", va="top", transform=ax.transAxes, fontsize=10, color="#454545")

    ylabel = r"Held-out usefulness $B_i=\mathbf{v}_i^\top\mathbf{h}_Q$"
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharex=True, sharey=True)
    for index, ax in enumerate(axes):
        panel(ax, index)
    axes[0].set_ylabel(ylabel, labelpad=10)
    fig.suptitle("Correct-answer support versus held-out usefulness", fontsize=18, y=0.975)
    fig.text(0.5, 0.918, subtitle, ha="center", fontsize=11, color="#454545")
    fig.text(0.5, 0.885, f"Each panel contains the same {len(frame):,} correct answers from "
             f"{frame.prompt_index.nunique()} training prompts.  DelTA gradient proxy; raw, shared axes.",
             ha="center", fontsize=11, color="#454545")
    fig.subplots_adjust(left=0.068, right=0.99, bottom=0.23, top=0.735, wspace=0.08)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"correct_answer_support_vs_usefulness.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    for index, method in enumerate(METHODS):
        fig, ax = plt.subplots(figsize=(7, 6.8))
        panel(ax, index)
        ax.set_ylabel(ylabel)
        fig.text(0.5, 0.975, f"{len(frame):,} correct answers · {frame.prompt_index.nunique()} prompts · DelTA proxy",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.83, bottom=0.25)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{method.lower()}_correct_answers.{suffix}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_report(output, summary, counts, heldout):
    lines = [
        "# Correct-answer support versus held-out usefulness", "",
        f"The recorded model is **{summary['model']}** (revision `{summary['model_revision']}`). "
        f"The dataset is **{summary['dataset']}** (revision `{summary['dataset_revision']}`). "
        f"This run sampled **{summary['train_problems']} train problems** and "
        f"**{summary['heldout_problems']} test problems**, with {summary['responses_per_problem']} responses "
        f"per problem and seed {summary['seed']}. "
        "These counts come from the immutable run manifest; the older 128/100 overview row in the source README is stale.",
        "", "![Three mechanisms](correct_answer_support_vs_usefulness.png)", "",
        "Each dot is one saved verifier-correct **training response**, rather than one training problem. "
        f"There are **{counts['eligible_directions']:,} dots per panel** from **{counts['mixed_groups']} mixed groups**. "
        f"The other {counts['all_zero_groups']} training groups had no correct responses. "
        f"There were {counts['all_one_groups']} all-correct groups and "
        f"{counts['zero_nonfinite_feature_exclusions']} feature exclusions. All three panels use the same observations, "
        "raw linear scales, and identical x/y limits; no outliers are removed.",
        "", "## Axes", "",
        r"The saved DelTA feature is $u_t=(1-p(y_t))h_t$. For a correct response, "
        r"$p_i=T_i^{-1}\sum_t u_{i,t}$ and $v_i=p_i/\|p_i\|$.", "",
        r"- GRPO: $S_i=v_i^\top g$.",
        r"- Linear routing: $S_i=v_i^\top(g+\delta_{\mathrm{Linear}})$.",
        r"- Nash routing: $S_i=v_i^\top(g+\delta_{\mathrm{NCR}})$.",
        r"- All three y-axes: $B_i=v_i^\top h_Q$, using the same held-out estimator.", "",
        f"The held-out estimator sums token proxies for {heldout['heldout_successes']} correct held-out responses "
        f"and divides by **all {heldout['denominator']} held-out draws**. It does not divide by response length "
        "or by the number of correct responses. These measurements use the DelTA proxy geometry, not full-model gradients.",
        "", "## Correlations", "",
        "| Mechanism | Spearman ρ | 95% CI | Pearson r | 95% CI |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in summary["estimates"]:
        lines.append(f"| {row['method']} | {row['spearman']:.6f} | "
                     f"[{row['spearman_ci95_low']:.4f}, {row['spearman_ci95_high']:.4f}] | "
                     f"{row['pearson']:.6f} | [{row['pearson_ci95_low']:.4f}, {row['pearson_ci95_high']:.4f}] |")
    lines += [
        "", "Point estimates independently reproduce the saved results. Spearman uses average ranks for ties. "
        f"The intervals were recomputed with {summary['bootstrap']['replicates']:,} paired bootstrap resamples of "
        f"whole training prompts, seed {summary['bootstrap']['seed']}, with prompt IDs in numeric order. "
        "Ranks are recomputed within every resample. These newly generated draws can yield slightly different "
        "intervals from the original report. Uncertainty is conditional on the fixed held-out estimator.",
        "", "All three methods have weak positive pooled correlations. The intervals include zero. "
        "This run does not provide clear evidence that NCR improves support/usefulness alignment over GRPO or Linear.",
    ]
    for name, difference in summary["paired_differences"]["spearman"].items():
        lines += ["", f"Paired Spearman difference {name}: **{difference['estimate']:.6f}**, "
                  f"95% CI [{difference['ci95'][0]:.6f}, {difference['ci95'][1]:.6f}]."]
    lines += [
        "", "## Data checks and limitations", "",
        f"- Reconstructed every S and B value from the {counts['mixed_groups']} cached Gram matrices, "
        "held-out cross products, and saved route shares/refunds. Maximum absolute disagreement is "
        f"{max(summary['source_audit']['dot_product_max_absolute_errors'].values()):.3g}. "
        f"All {len(summary['source_audit']['artifact_sha256'])} recorded artifact hashes checked against the saved stage state.",
        f"- NCR fell back to GRPO in {summary['fallback_prompt_groups']['NCR']} groups, affecting "
        f"{summary['fallback_correct_answers']['NCR']} plotted answers; those observations are retained. "
        f"Linear had {summary['fallback_prompt_groups']['Linear']} fallback groups.",
        *[
            f"- A separate parser audit changes the reward of {change['split']} prompt "
            f"{change['prompt_index']}, rollout {change['rollout_index']}, from "
            f"{change['before']['reward']} to {change['after']['reward']}. This audit was not applied "
            "to the saved verifier and routing artifacts. The figures preserve those consistent records; "
            "applying the changed labels requires recomputing the affected downstream features and routes."
            for change in summary["parser_audit_changes_not_applied"]
        ],
        "- This is a frozen-checkpoint diagnostic. No training update or causal accuracy gain was measured.",
        "", "## Files and reproduction", "",
        "[Combined PDF](correct_answer_support_vs_usefulness.pdf) · "
        "[GRPO](grpo_correct_answers.png) · [Linear](linear_correct_answers.png) · [NCR](ncr_correct_answers.png)",
        "", "[All plotted coordinates](correct_answer_points.csv) · [Correlation table](correlations.csv) · "
        "[Exact values and audit](analysis.json)",
        "", "From `examples/nash_exp`, with the CPU packages listed in the script installed:",
        "", "```bash", "python scripts/plot_correct_answer_support.py", "```", "",
        "The script reads saved artifacts and writes this separate analysis directory. "
        "It requires no model loading, new rollouts, or solver replay.", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = args.output_dir or root / "correct_answer_analysis"
    frame = pd.read_parquet(root / "direction_records.parquet").reset_index(drop=True)
    config = yaml.safe_load((root / "config_resolved.yaml").read_text())
    manifest = read_json(root / "manifest.json")
    stored = read_json(root / "correlations.json")
    audit = audit_records(root, frame, config)
    print("Verified saved artifacts and reconstructed every plotted dot product.", flush=True)
    seed, replicates = stored["bootstrap_seed"], stored["bootstrap_replicates"]
    rows, differences = correlations(frame, seed, replicates)
    for row in rows:
        for metric in ("spearman", "pearson"):
            expected = stored["populations"]["all_eligible"]["estimates"][metric][row["method"]]
            np.testing.assert_allclose(row[metric], expected["estimate"], rtol=0, atol=1e-12)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "correlations.csv", index=False)
    point_columns = ["prompt_index", "train_dataset_row_id", "rollout_index", "reward", "response_token_length",
                     "group_success_count", "p_norm", "B_i", "S_GRPO", "S_Linear", "S_NCR",
                     "fallback_Linear", "fallback_NCR"]
    frame[point_columns].to_csv(output / "correct_answer_points.csv", index=False)
    subtitle = (f"{manifest['model']['id']} · MATH train: {len(manifest['train_row_indices'])} problems"
                f" · MATH500 test: {len(manifest['heldout_row_indices'])} problems · {config['n_rollout']} responses/problem")
    make_plots(frame, rows, output, subtitle)
    parser_audit_path = root / "parsing_audit/rescore_comparison.json"
    parser_audit = read_json(parser_audit_path) if parser_audit_path.exists() else None
    summary = {
        "dataset": manifest["train"]["id"], "dataset_revision": manifest["train"]["revision"],
        "model": manifest["model"]["id"], "model_revision": manifest["model"]["revision"],
        "train_problems": len(manifest["train_row_indices"]), "heldout_problems": len(manifest["heldout_row_indices"]),
        "seed": manifest["seed"], "responses_per_problem": config["n_rollout"],
        "geometry": "DelTA selected-token gradient proxy; v_i is the normalized response-mean feature",
        "estimates": rows, "paired_differences": differences,
        "bootstrap": {"unit": "training_prompt", "replicates": replicates, "seed": seed,
                      "paired_across_methods": True, "interval": "95% percentile", "heldout_estimator": "fixed",
                      "prompt_order": [int(index) for index in np.sort(frame.prompt_index.unique())]},
        "fallback_correct_answers": {method: int(frame["fallback_" + method].sum()) for method in METHODS[1:]},
        "fallback_prompt_groups": {method: int(frame.loc[frame["fallback_" + method], "prompt_index"].nunique())
                                   for method in METHODS[1:]},
        "source_audit": audit,
        "parser_audit_changes_not_applied": [] if parser_audit is None else parser_audit["changes"],
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "pandas", "pyarrow", "matplotlib", "pyyaml")},
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_report(output, summary, read_json(root / "eligibility_summary.json"),
                 read_json(root / "heldout_feature_manifest.json"))
    print(pd.DataFrame(rows)[["method", "spearman", "pearson"]].to_string(index=False), flush=True)
    print(f"Saved figures, coordinates, correlations, and audit to {output}", flush=True)


if __name__ == "__main__":
    main()
