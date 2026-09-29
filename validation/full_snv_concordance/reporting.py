"""Human-readable scientific report and bounded-density diagnostic figures."""

from __future__ import annotations

import json
import math
import base64
import html
from pathlib import Path
import re
from typing import Mapping, Sequence

from .aggregate import DOMINANT_LABELS, EVENTS, SCORE_LABELS


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _run_status(summary: Mapping) -> str:
    finality = summary.get("finality", {})
    status = str(finality.get("status", "provisional")).lower()
    if status not in {"final", "provisional"}:
        raise ValueError(f"invalid reducer finality status: {status!r}")
    provenance = summary.get("verified_provenance", {})
    if status == "final" and (
        finality.get("contract") != "explicit-reducer-finality-v1"
        or provenance.get("status") != "verified"
        or provenance.get("contract") != "audited-vcf-snapshot-v1"
    ):
        raise ValueError(
            "a final report requires verified reducer finality and provenance contracts"
        )
    return status


def _score_row(label: str, value: Mapping) -> str:
    distribution = value.get("distribution", {})
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        label,
        value.get("n", 0),
        _fmt(value.get("pearson_r")),
        _fmt(distribution.get("spearman_r_binned")),
        _fmt(value.get("lin_ccc")),
        _fmt(value.get("bias_right_minus_left")),
        _fmt(value.get("mae")),
        _fmt(value.get("rmse")),
        _fmt(distribution.get("ks_distance_binned")),
        _fmt(distribution.get("wasserstein_1_binned")),
    )


def _append_strata_summary(lines: list[str], metrics: Mapping) -> None:
    strata = metrics.get("strata", {})
    lines.extend(["", "## Genomic and annotation strata", ""])
    lines.append(
        "Strata are descriptive MAX-score summaries. A gene row is an annotation-level "
        "observation; it is not an independent biological replicate. The substitution "
        "stratum is REF>ALT only because the score VCF does not retain flanking sequence, "
        "so it is not a trinucleotide mutational context."
    )
    for dimension in ("chrom", "substitution", "site_event", "dominant_pair"):
        groups = strata.get(dimension, {})
        if not groups:
            continue
        lines.extend(
            [
                "",
                f"### {dimension}",
                "",
                "| Stratum | N | Pearson r | Bias | MAE | RMSE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        ordered = sorted(
            groups.items(),
            key=lambda item: (-int(item[1]["metrics"].get("n", 0)), item[0]),
        )
        for key, values in ordered[:40]:
            score = values["metrics"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    key,
                    score.get("n", 0),
                    _fmt(score.get("pearson_r")),
                    _fmt(score.get("bias_right_minus_left")),
                    _fmt(score.get("mae")),
                    _fmt(score.get("rmse")),
                )
            )
        if len(ordered) > 40:
            lines.append(f"\nOnly the 40 largest of {len(ordered):,} strata are shown.")


def render_report(summary_path: str | Path, output_dir: str | Path) -> Path:
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    metrics = summary["metrics"]
    left_label = str(summary.get("left_label", "left"))
    right_label = str(summary.get("right_label", "right"))
    title = "OpenSpliceAI–SpliceAI concordance"
    if "seeds" in str(summary.get("kind", "")):
        title = "OpenSpliceAI seed reproducibility"
    status = _run_status(summary)
    finality = summary.get("finality", {})
    provenance = summary.get("verified_provenance", {})
    provenance_status = str(provenance.get("status", "unverified")).upper()
    manifest_digests = provenance.get("audit_manifest_sha256s", [])
    provenance_classes = provenance.get("output_provenance_class_counts", {})
    expected_pair_count = finality.get("expected_overlap_count")
    if expected_pair_count is None:
        expected_pair_count = finality.get("expected_total_chunks", "unknown")
    estimand = metrics.get("estimand", {})
    lines = [
        f"# {title}",
        "",
        f"**Analysis status: {status.upper()}.** Generated from `{summary_path}`.",
        "",
        "> Agreement is descriptive. Neither predictor is biological ground truth in this report.",
        "",
        "## Verified provenance and reducer finality",
        "",
        "| Contract field | Value |",
        "|---|---|",
        f"| Provenance status | {provenance_status} |",
        f"| Provenance contract | `{provenance.get('contract', 'none')}` |",
        f"| Frozen pairs SHA-256 | `{provenance.get('pairs_sha256', 'unavailable')}` |",
        f"| Audit manifest SHA-256 count | {len(manifest_digests)} |",
        f"| Verified chunks | {provenance.get('verified_chunk_count', 0)} |",
        f"| Verified VCF snapshots | {provenance.get('verified_vcf_count', 0)} |",
        f"| Receipt-bound outputs | {provenance_classes.get('receipt_bound', 0)} |",
        f"| Outputs with invalid receipts | {provenance_classes.get('receipt_invalid', 0)} |",
        f"| Legacy log-supported outputs | {provenance_classes.get('legacy_log_supported', 0)} |",
        f"| Legacy unprovenanced outputs | {provenance_classes.get('legacy_unprovenanced', 0)} |",
        f"| Outputs carrying receipt evidence | {provenance.get('receipt_evidence_output_count', 0)} |",
        f"| Outputs carrying log evidence | {provenance.get('log_evidence_output_count', 0)} |",
        f"| Finality contract | `{finality.get('contract', 'none')}` |",
        f"| Finality policy | `{finality.get('policy', 'partial_coverage_allowed')}` |",
        f"| Observed / expected pairs | {finality.get('observed_pair_count', 'unknown')} / {expected_pair_count} |",
        f"| Excluded incomplete edge fragments | {finality.get('excluded_incomplete_edge_fragments', metrics.get('coverage', {}).get('excluded_incomplete_edge_fragments', 0))} |",
        "",
        "Mapper run labels are descriptive only. The status above comes from the explicit "
        "reducer contract after frozen-pair coverage and audit provenance verification.",
        "`VERIFIED` means that mapped bytes and canonical records matched the frozen audit "
        "snapshot. It does not by itself prove which model generated a legacy output; only "
        "receipt-bound outputs cryptographically bind generation to the frozen run, while "
        "invalid receipts, log-supported outputs, and unprovenanced legacy outputs retain "
        "the stated limitation.",
        "",
        "## What is being compared",
        "",
        f"The primary estimand is **{estimand.get('primary', 'exact gene-matched annotation pairs')}**. "
        f"`{left_label}` is the left value and `{right_label}` is the right value; all reported "
        "biases are right minus left. Repeated source rows and identical annotations are "
        "deduplicated before pairing, while conflicting annotations are excluded and counted.",
        "",
        "A separate annotation-agnostic sensitivity view collapses each predictor across genes "
        "independently. It answers whether the predictors identify signal somewhere for the same "
        "allele, but it can pair different genes and must not be interpreted as gene concordance.",
        "",
        "## Coverage",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(metrics["coverage"].items()):
        lines.append(f"| {key} | {value:,} |")

    lines.extend(
        [
            "",
            "## Continuous-score agreement: exact-gene primary view",
            "",
            f"| Score | N | Pearson r | Approx. Spearman | Lin CCC | Bias ({right_label} − {left_label}) | MAE | RMSE | Binned KS | Binned W1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in SCORE_LABELS:
        lines.append(_score_row(label, metrics["scores"][label]))

    max_score = metrics["scores"]["MAX"]
    max_distribution = max_score.get("distribution", {})
    lines.extend(
        [
            "",
            "The KS, Wasserstein-1, Jensen–Shannon, difference quantiles, and Spearman values "
            "are histogram-derived approximations. Pearson, Lin CCC, bias, MAE, RMSE, exact "
            "match, zero rates, and threshold tables are computed from additive exact sums/counts.",
            "",
            f"For MAX, binned Jensen–Shannon divergence is **{_fmt(max_distribution.get('jensen_shannon_divergence_base2'))}** "
            f"and binned Spearman is **{_fmt(max_distribution.get('spearman_r_binned'))}**.",
        ]
    )

    rounded = metrics.get("right_rounded_2dp", {})
    if rounded:
        lines.extend(
            [
                "",
                "## Two-decimal right-score sensitivity analysis",
                "",
                "This analysis rounds only the right predictor to two decimals before comparison. "
                "For OpenSpliceAI-versus-SpliceAI this quantifies how much discrepancy could be "
                "masked by the precision of published SpliceAI VCF scores. It does not prove that "
                "remaining differences are model differences, and in a seed-versus-seed run it is "
                "only a generic rounding sensitivity analysis.",
                "",
                "| Score | Raw MAE | Rounded MAE | Δ MAE | Raw exact | Rounded exact | Exact-rate gain |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in SCORE_LABELS:
            value = rounded["impact_vs_raw"][label]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    label,
                    _fmt(value.get("raw_mae")),
                    _fmt(value.get("right_rounded_2dp_mae")),
                    _fmt(value.get("mae_change_rounded_minus_raw")),
                    _fmt(value.get("raw_exact_match_rate")),
                    _fmt(value.get("right_rounded_2dp_exact_match_rate")),
                    _fmt(value.get("exact_match_rate_gain")),
                )
            )

    subsets = metrics.get("signal_subsets", {}).get("MAX", {})
    if subsets:
        lines.extend(
            [
                "",
                "## Union-signal MAX subsets",
                "",
                "These subsets require either predictor to exceed the named signal gate. They "
                "remove the shared zero/near-zero mass that can make whole-genome agreement look "
                "artificially strong.",
                "",
                "| Inclusion rule | N | Pearson r | Lin CCC | Bias | MAE | RMSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for subset, value in subsets.items():
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    subset,
                    value.get("n", 0),
                    _fmt(value.get("pearson_r")),
                    _fmt(value.get("lin_ccc")),
                    _fmt(value.get("bias_right_minus_left")),
                    _fmt(value.get("mae")),
                    _fmt(value.get("rmse")),
                )
            )

    collapsed = metrics.get("variant_collapsed_view", {})
    if collapsed:
        lines.extend(
            [
                "",
                "## Annotation-agnostic per-variant sensitivity view",
                "",
                "| View | N (MAX) | Pearson r | Lin CCC | Bias | MAE | RMSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for view_label, value in (
            ("Exact gene", metrics["scores"]["MAX"]),
            ("Variant max across genes", collapsed["scores"]["MAX"]),
        ):
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    view_label,
                    value.get("n", 0),
                    _fmt(value.get("pearson_r")),
                    _fmt(value.get("lin_ccc")),
                    _fmt(value.get("bias_right_minus_left")),
                    _fmt(value.get("mae")),
                    _fmt(value.get("rmse")),
                )
            )

    lines.extend(
        [
            "",
            "## Maximum-score threshold agreement",
            "",
            "Overall agreement includes both-negative calls; Jaccard, Dice/positive agreement, "
            "and the left-only/right-only counts are more informative for sparse signal.",
            "",
            "| Threshold | Both + | Left only | Right only | Both - | Agreement | Jaccard | MCC |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold, value in metrics["thresholds"]["MAX"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                threshold,
                value["both_positive"],
                value["left_only"],
                value["right_only"],
                value["both_negative"],
                _fmt(value["overall_agreement"]),
                _fmt(value["jaccard"]),
                _fmt(value["mcc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Predicted splice-position agreement",
            "",
            "DP agreement is evaluated only when both methods pass the same event/score gate. "
            "DP=0 is a valid at-variant location.",
            "",
            "| Event | Threshold | Eligible | Exact DP | Within 1 nt | Within 2 nt | Within 5 nt | Within 10 nt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for event in EVENTS:
        for threshold, value in metrics["dp"][event].items():
            rates = value["within_rates"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    event,
                    threshold,
                    value["eligible"],
                    _fmt(rates.get("0")),
                    _fmt(rates.get("1")),
                    _fmt(rates.get("2")),
                    _fmt(rates.get("5")),
                    _fmt(rates.get("10")),
                )
            )

    bootstrap = metrics.get("cluster_bootstrap_95ci", {})
    if bootstrap:
        lines.extend(
            [
                "",
                "## Cluster-bootstrap uncertainty for MAX",
                "",
                f"Deterministic {bootstrap.get('replicates', 0):,}-replicate percentile intervals "
                "resample additive strata with replacement. Gene- and 1-Mb-block intervals "
                "describe different dependence assumptions and are sensitivity analyses, not "
                "formal confidence intervals for biological truth.",
                "",
                "| Cluster | Clusters | Observations | Metric | Estimate | 95% interval |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for dimension, values in bootstrap.get("dimensions", {}).items():
            for metric_name, interval in values.get("metrics", {}).items():
                lines.append(
                    "| {} | {} | {} | {} | {} | [{}, {}] |".format(
                        dimension,
                        values.get("cluster_count", 0),
                        values.get("observation_count", 0),
                        metric_name,
                        _fmt(interval.get("estimate")),
                        _fmt(interval.get("lower")),
                        _fmt(interval.get("upper")),
                    )
                )

    _append_strata_summary(lines, metrics)
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Whole-genome observations are highly imbalanced toward zero; use union-signal and threshold-positive metrics alongside global correlations.",
            "- Correlation measures association, not calibration or interchangeable predictions. Lin CCC, bias, error magnitudes, threshold tables, and DP agreement address different questions.",
            "- Exact-gene and variant-collapsed views answer different estimands. A stronger collapsed result can indicate annotation/gene assignment differences rather than closer score calibration.",
            "- Histogram-derived metrics are approximate at the configured bin width. The JSON metadata records that approximation.",
            "- Gene annotations and nearby variants are dependent. Cluster bootstrap summaries expose sensitivity to gene and genomic-block clustering but do not correct every dependency.",
            f"- This run is **{status}**. A provisional run may omit chunks; conclusions must be refreshed from the fail-closed final reduction before publication.",
            "",
            "## Generated figures",
            "",
            "- `score_distributions.png`: marginal score distributions.",
            "- `joint_score_heatmaps.png`: bounded-density joint score maps.",
            "- `difference_distributions.png`: signed right-minus-left distributions.",
            "- `threshold_agreement.png`: signal-focused threshold metrics.",
            "- `dp_agreement.png`: site-position agreement conditional on both-positive events.",
            "- `dominant_event_agreement.png`: raw and row-normalized dominant-event matrices.",
            "- `chrom_gene_block_summaries.png`: bounded chromosome/gene/block heterogeneity view.",
            "- `quantization_sensitivity.png`: raw versus right-rounded error.",
        ]
    )
    markdown_text = "\n".join(lines) + "\n"
    report = output_dir / "report.md"
    report.write_text(markdown_text, encoding="utf-8")
    _render_plots(summary, output_dir)
    _render_html_report(markdown_text, output_dir, summary)
    return report


def _render_html_report(markdown_text: str, output_dir: Path, summary: Mapping) -> Path:
    """Write a self-contained HTML rendering alongside the Markdown report."""

    try:
        import markdown

        body = markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "toc"],
            output_format="html5",
        )
        renderer = f"Python-Markdown {getattr(markdown, '__version__', 'unknown')}"
    except ImportError:
        # Keep report generation usable in a minimal analysis environment while
        # making the degraded rendering explicit in the document.
        body = f"<pre>{html.escape(markdown_text)}</pre>"
        renderer = "plain-text fallback (Python-Markdown unavailable)"

    # Embed generated PNGs so report.html can be copied or archived as a single
    # file without losing the visual diagnostics.
    def embed_image(match: re.Match[str]) -> str:
        filename = match.group(1)
        image_path = output_dir / filename
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            return match.group(0)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    body = re.sub(r'src="([^"/]+\.png)"', embed_image, body)
    figure_blocks = []
    for image_path in sorted(output_dir.glob("*.png")):
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        figure_blocks.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(image_path.stem.replace("_", " "))}">'
            f'<figcaption>{html.escape(image_path.name)}</figcaption></figure>'
        )
    if figure_blocks:
        body += "<h2>Embedded figures</h2>" + "\n".join(figure_blocks)
    title = html.escape(str(summary.get("left_label", "left")))
    title += " versus " + html.escape(str(summary.get("right_label", "right")))
    status = html.escape(str(summary.get("finality", {}).get("status", "provisional")).upper())
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenSpliceAI concordance: {title}</title>
<style>
body {{ max-width: 1400px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.5 system-ui, sans-serif; color: #202124; }}
h1, h2, h3 {{ line-height: 1.2; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #d0d7de; padding: .35rem .55rem; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; }}
code, pre {{ background: #f6f8fa; border-radius: 4px; }}
code {{ padding: .1rem .25rem; }}
pre {{ overflow-x: auto; padding: 1rem; }}
img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto; }}
.status {{ border-left: .4rem solid #0969da; background: #ddf4ff; padding: .75rem 1rem; }}
</style>
</head>
<body>
<div class="status">Analysis finality: <strong>{status}</strong>. HTML renderer: {html.escape(renderer)}.</div>
{body}
</body>
</html>
"""
    path = output_dir / "report.html"
    temporary = output_dir / ".report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path


def _safe_log_counts(values):
    import numpy as np

    return np.log10(np.asarray(values, dtype=float) + 1.0)


def _label_axes(axis, labels: Sequence[str]) -> None:
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)


def _render_plots(summary: Mapping, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    raw = summary["raw"]
    metrics = summary["metrics"]
    bins = int(raw["config"]["score_bins"])
    centers = [(index + 0.5) / bins for index in range(bins)]
    left_label = summary.get("left_label", "left")
    right_label = summary.get("right_label", "right")

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, label in zip(axes.flat, SCORE_LABELS):
        histogram = raw["score_hist"][label]
        axis.plot(centers, histogram["left"], label=left_label, linewidth=1)
        axis.plot(centers, histogram["right"], label=right_label, linewidth=1)
        axis.set_yscale("symlog", linthresh=1)
        axis.set_title(label)
        axis.set_xlabel("delta score")
        axis.set_ylabel("count")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right")
    figure.savefig(output_dir / "score_distributions.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    for axis, label in zip(axes.flat, SCORE_LABELS):
        matrix = np.asarray(raw["joint_hist"][label], dtype=float).reshape(bins, bins)
        image = axis.imshow(
            np.log10(matrix.T + 1.0),
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="equal",
            interpolation="nearest",
        )
        axis.plot([0, 1], [0, 1], color="white", alpha=0.55, linewidth=0.7)
        axis.set_title(label)
        axis.set_xlabel(str(left_label))
        axis.set_ylabel(str(right_label))
        figure.colorbar(image, ax=axis, label="log10(count + 1)")
    axes.flat[-1].axis("off")
    figure.savefig(output_dir / "joint_score_heatmaps.png", dpi=180)
    plt.close(figure)

    difference_centers = [(index - bins) / bins for index in range(2 * bins + 1)]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, label in zip(axes.flat, SCORE_LABELS):
        axis.plot(difference_centers, raw["diff_hist"][label], linewidth=1)
        axis.axvline(0, color="black", alpha=0.4, linewidth=0.7)
        axis.set_yscale("symlog", linthresh=1)
        axis.set_title(label)
        axis.set_xlabel(f"{right_label} − {left_label}")
        axis.set_ylabel("count")
    axes.flat[-1].axis("off")
    figure.savefig(output_dir / "difference_distributions.png", dpi=180)
    plt.close(figure)

    thresholds = [float(value) for value in metrics["thresholds"]["MAX"]]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label in SCORE_LABELS:
        values = metrics["thresholds"][label]
        ordered = [values[f"{threshold:.10g}"] for threshold in thresholds]
        axes[0].plot(thresholds, [item["jaccard"] for item in ordered], marker="o", label=label)
        axes[1].plot(thresholds, [item["mcc"] for item in ordered], marker="o", label=label)
        axes[2].plot(
            thresholds,
            [item["call_rate_ratio_right_over_left"] for item in ordered],
            marker="o",
            label=label,
        )
    for axis, title_text in zip(axes, ("Jaccard", "MCC", "Right/left call-rate ratio")):
        axis.set_title(title_text)
        axis.set_xlabel("threshold")
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(-1.02, 1.02)
    axes[0].legend(fontsize="small")
    figure.savefig(output_dir / "threshold_agreement.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    tolerances = ("0", "1", "2", "5", "10")
    for axis, event in zip(axes.flat, EVENTS):
        for threshold, values in metrics["dp"][event].items():
            rates = values["within_rates"]
            axis.plot(
                [int(value) for value in tolerances],
                [rates.get(value) for value in tolerances],
                marker="o",
                label=f"score ≥ {threshold} (n={values['eligible']:,})",
            )
        axis.set_title(event)
        axis.set_xlabel("absolute DP tolerance (nt)")
        axis.set_ylabel("conditional agreement")
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize="x-small")
        axis.grid(alpha=0.2)
    figure.savefig(output_dir / "dp_agreement.png", dpi=180)
    plt.close(figure)

    dominant_labels = list(DOMINANT_LABELS)
    matrix = np.asarray(
        [[raw["dominant"][left][right] for right in dominant_labels] for left in dominant_labels],
        dtype=float,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    first_image = axes[0].imshow(np.log10(matrix + 1.0), interpolation="nearest", aspect="auto")
    second_image = axes[1].imshow(normalized, interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    for axis, title_text in zip(axes, ("log10(count + 1)", "row-normalized fraction")):
        _label_axes(axis, dominant_labels)
        axis.set_title(title_text)
        axis.set_xlabel(f"{right_label} dominant event")
        axis.set_ylabel(f"{left_label} dominant event")
    figure.colorbar(first_image, ax=axes[0])
    figure.colorbar(second_image, ax=axes[1])
    figure.savefig(output_dir / "dominant_event_agreement.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, dimension in zip(axes, ("chrom", "gene", "block_1mb")):
        groups = metrics.get("strata", {}).get(dimension, {})
        selected = sorted(
            groups.items(),
            key=lambda item: (-int(item[1]["metrics"].get("n", 0)), item[0]),
        )[:25]
        selected.reverse()
        labels = [item[0] for item in selected]
        maes = [item[1]["metrics"].get("mae") or 0.0 for item in selected]
        counts = [max(1, int(item[1]["metrics"].get("n", 0))) for item in selected]
        positions = np.arange(len(selected))
        axis.barh(
            positions, maes, color=plt.cm.viridis(np.log10(counts) / max(1.0, math.log10(max(counts, default=1))))
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize="x-small")
        axis.set_title(f"{dimension}: 25 largest strata")
        axis.set_xlabel("MAX MAE (color tracks log10 N)")
    figure.savefig(output_dir / "chrom_gene_block_summaries.png", dpi=180)
    plt.close(figure)

    rounded = metrics.get("right_rounded_2dp", {}).get("impact_vs_raw", {})
    if rounded:
        x = np.arange(len(SCORE_LABELS))
        width = 0.36
        raw_mae = [rounded[label].get("raw_mae") or 0.0 for label in SCORE_LABELS]
        rounded_mae = [rounded[label].get("right_rounded_2dp_mae") or 0.0 for label in SCORE_LABELS]
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.bar(x - width / 2, raw_mae, width, label="raw")
        axis.bar(x + width / 2, rounded_mae, width, label="right rounded to 2dp")
        axis.set_xticks(x)
        axis.set_xticklabels(SCORE_LABELS)
        axis.set_ylabel("MAE")
        axis.set_title("Two-decimal quantization sensitivity")
        axis.legend()
        figure.savefig(output_dir / "quantization_sensitivity.png", dpi=180)
        plt.close(figure)
