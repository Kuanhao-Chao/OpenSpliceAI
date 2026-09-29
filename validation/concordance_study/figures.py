"""Figures for the cross-run synthesis.

Every figure is derived from reduced aggregates, carries a companion table in the
report (the relief rule for the low-contrast categorical slot, and the honest way
to present a figure whose numbers matter), and is labelled directly rather than
relying on colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import derive, style  # noqa: E402
from .loading import EVENTS, SCORE_LABELS, Run  # noqa: E402

FigureIndex = List[Dict[str, str]]


def _save(fig, out_dir: Path, name: str, index: FigureIndex, caption: str) -> None:
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor=style.SURFACE)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=style.SURFACE)
    plt.close(fig)
    index.append({"file": name, "caption": caption})


def _thousands(value: float) -> str:
    return f"{value:,.0f}"


# --------------------------------------------------------------------------
def fig_coverage(run: Run, out_dir: Path, index: FigureIndex) -> None:
    c = run.coverage
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12.4, 3.3))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis("off")
    nodes = [(0.10,.52,"Retained VCF rows",c["source_rows"],style.INK_MUTED),
             (0.34,.52,"Variant groups",c["variant_groups"],style.INK_MUTED),
             (0.60,.76,"Valid SpliceAI\nannotations",c["left_valid_annotations"],style.SPLICEAI),
             (0.60,.29,"Valid OpenSpliceAI\nannotations",c["right_valid_annotations"],style.OPENSPLICEAI),
             (0.88,.52,"Exact-gene pairs",c["paired_annotations"],style.NEUTRAL_FILL)]
    for x,y,label,n,color in nodes:
        ax.add_patch(FancyBboxPatch((x-.085,y-.125),.17,.25,
                    boxstyle="round,pad=.01",facecolor=style.SURFACE,edgecolor=color,linewidth=1.5))
        ax.text(x,y+.035,label,ha="center",va="center",fontsize=9,color=style.INK)
        ax.text(x,y-.065,f"{n:,}",ha="center",va="center",fontsize=10,fontweight="bold",color=color)
    for start,end in (((.20,.52),(.24,.52)),((.44,.55),(.50,.74)),((.44,.48),(.50,.31)),
                      ((.70,.74),(.78,.56)),((.70,.31),(.78,.48))):
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color=style.INK_MUTED,lw=1.3))
    ax.text(.5,.035,f"Unpaired annotations: left {c.get('left_only_annotations',0):,} · right {c.get('right_only_annotations',0):,}"
            f"     |     Excluded boundary fragments: {c.get('excluded_incomplete_edge_fragments',0):,}",
            ha="center",va="center",fontsize=9,color=style.INK_SECONDARY)
    ax.set_title("From source records to an exact-gene comparison",loc="left")
    _save(fig, out_dir, "f01_coverage.png", index,
          "Coverage flow with the unit labelled at every stage. Multiple genes can annotate one "
          "variant group, so annotation counts are not a decreasing sequence of row counts. "
          "Arrows describe processing; their widths do not encode quantities.")


def _column_quantiles(joint: np.ndarray, bins: int,
                      probabilities: Sequence[float]) -> Dict[float, np.ndarray]:
    """Quantiles of OpenSpliceAI within each SpliceAI column of the joint histogram."""
    centres = (np.arange(bins) + 0.5) / bins
    totals = joint.sum(axis=1)
    out = {p: np.full(bins, np.nan) for p in probabilities}
    for column in range(bins):
        if totals[column] <= 0:
            continue
        cumulative = np.cumsum(joint[column]) / totals[column]
        for p in probabilities:
            out[p][column] = centres[int(np.searchsorted(cumulative, p))]
    return out


def fig_joint_density(run: Run, out_dir: Path, index: FigureIndex) -> None:
    """Calibration of OpenSpliceAI against SpliceAI, as conditional quantile bands.

    A raw joint-density heatmap is useless at this scale -- one cell near the
    origin holds billions of pairs and the rest of the plane washes out to a
    single tone. The question a reader actually has is *when SpliceAI says x,
    what does OpenSpliceAI say?*, which is a conditional distribution; showing
    its median and two central bands answers it directly and quantitatively.
    """
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.3), sharey=True)
    bins = run.score_bins
    centres = (np.arange(bins) + 0.5) / bins
    for ax, label in zip(axes, SCORE_LABELS):
        joint = np.asarray(run.joint_hist(label), dtype=float).reshape(bins, bins)
        q = _column_quantiles(joint, bins, (0.05, 0.25, 0.5, 0.75, 0.95))
        ax.vlines(centres, q[0.05], q[0.95], color=style.SPLICEAI, alpha=0.18,
                  linewidth=2, label="5-95%")
        ax.vlines(centres, q[0.25], q[0.75], color=style.SPLICEAI, alpha=0.45,
                  linewidth=2, label="25-75%")
        ax.plot([0, 1], [0, 1], color=style.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                label="exact agreement")
        ax.plot(centres, q[0.5], color=style.OPENSPLICEAI, linewidth=1.0,
                marker="o", markersize=2, label="median")
        ax.set_title(style.EVENT_TITLES[label])
        ax.set_xlabel("SpliceAI score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style.clean(ax, grid="both")
    axes[0].set_ylabel("OpenSpliceAI score")
    axes[0].legend(loc="upper left", fontsize=7.5)
    _save(fig, out_dir, "f02_joint_density.png", index,
          "Calibration: the distribution of the OpenSpliceAI score conditional on the SpliceAI "
          "score. Orange dots mark the conditional median; vertical intervals are the central "
          "50% and 90%. A perfectly agreeing pair would track the dashed line. Empty comparator "
          "bins remain gaps; the two-decimal score grid naturally leaves some bins unoccupied.")


def fig_difference_distributions(run: Run, out_dir: Path, index: FigureIndex) -> None:
    """Tail asymmetry: how much more often each predictor is the higher one.

    The signed-difference histogram this replaces was five near-identical spiky
    combs on a ten-decade log axis, in which the actual finding -- that the gain
    scores skew one way and the loss scores do not -- was invisible. Comparing the
    two tails directly states it: the curve is the ratio of "SpliceAI higher by at
    least x" to "OpenSpliceAI higher by at least x", so 1.0 is symmetry and
    everything above it is SpliceAI scoring higher more often.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.7))
    for label, colour, marker in zip(SCORE_LABELS, style.CATEGORICAL, style.MARKERS):
        tails = derive.tail_asymmetry(run, label)
        magnitudes = tails["magnitudes"]
        left = [np.nan if v is None else v for v in tails["p_left_tail"]]
        right = [np.nan if v is None else v for v in tails["p_right_tail"]]
        ratio = [np.nan if v is None else v for v in tails["left_over_right"]]
        axes[0].plot(magnitudes, left, color=colour, linewidth=1.9,
                     label=style.EVENT_TITLES[label])
        axes[0].plot(magnitudes, right, color=colour, linewidth=1.9, linestyle=(0, (3, 2)))
        axes[1].plot(magnitudes, ratio, color=colour, linewidth=2.0, marker=marker,
                     markevery=12, markersize=5, label=style.EVENT_TITLES[label])

    axes[0].set_yscale("log")
    axes[0].set_xlabel("difference magnitude x")
    axes[0].set_ylabel("share of paired annotations")
    axes[0].set_title("Solid: SpliceAI higher by $\\geq$ x   ·   Dashed: OpenSpliceAI higher")
    axes[0].legend(loc="upper right", fontsize=7.5)
    style.clean(axes[0], grid="both")

    axes[1].axhline(1.0, color=style.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[1].text(0.98, 1.06, "symmetric", fontsize=7.5, color=style.INK_SECONDARY,
                 ha="right", transform=axes[1].get_yaxis_transform())
    axes[1].set_yscale("log")
    axes[1].set_xlabel("difference magnitude x")
    axes[1].set_ylabel("SpliceAI-higher / OpenSpliceAI-higher")
    axes[1].set_title("Which predictor is the higher one, and by how much more often")
    style.clean(axes[1], grid="both")
    _save(fig, out_dir, "f03_difference_distributions.png", index,
          "Tail asymmetry of the signed score difference. Left: the probability that each "
          "predictor exceeds the other by at least x. Right: their ratio, where 1.0 would mean "
          "the two disagree in both directions equally often. Difference magnitudes are rounded "
          "to histogram centres, so these tail estimates are approximate.")


def fig_threshold_agreement(run: Run, out_dir: Path, index: FigureIndex,
                            thresholds: Sequence[float] = derive.DEFAULT_THRESHOLDS) -> None:
    """Agreement as a continuous function of the calling cutoff.

    The configured thresholds give four or five points; the joint histogram gives
    one per score bin, so this is a curve rather than a line between four dots.
    Overall agreement is deliberately absent: with 99.8 % of pairs below any cutoff
    in both predictors it is above 99 % everywhere and says nothing.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5))
    panels = [("jaccard", "Jaccard (positive class)"), ("kappa", "Cohen's kappa"), ("mcc", "Matthews correlation"),
              ("call_rate_ratio_right_over_left", "Call-rate ratio (OSAI / SpliceAI)")]
    curves = {label: derive.same_cutoff_curve(run, label) for label in SCORE_LABELS}
    for ax, (field, title) in zip(axes, panels):
        for label, colour, marker in zip(SCORE_LABELS, style.CATEGORICAL, style.MARKERS):
            curve = curves[label]
            values = [np.nan if p[field] is None else p[field] for p in curve["points"]]
            ax.plot(curve["cutoffs"], values, color=colour, linewidth=2.0, marker=marker,
                    markevery=max(1, len(curve["cutoffs"])//8), markersize=3,
                    label=style.EVENT_TITLES[label])
        if field == "call_rate_ratio_right_over_left":
            ax.axhline(1.0, color=style.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("delta-score cutoff, applied to both")
        ax.set_xlim(0, 1)
        style.clean(ax, grid="both")
    axes[0].set_ylabel("agreement")
    axes[0].legend(loc="upper left", fontsize=7.5)
    _save(fig, out_dir, "f04_threshold_agreement.png", index,
          "Chance-corrected and positive-class agreement, and the ratio of call volumes, as "
          "continuous functions of a cutoff applied to both predictors. The dashed line on the "
          "right marks equal call volume. Values below one mean fewer OpenSpliceAI calls, "
          "without implying greater specificity.")


def fig_operating_point(run: Run, out_dir: Path, index: FigureIndex,
                        thresholds: Sequence[float] = derive.DEFAULT_THRESHOLDS) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.5), sharey=True)
    for event, ax in zip(SCORE_LABELS, axes):
        rows = derive.operating_point_transfer(run, event)
        x = [r["spliceai_threshold"] for r in rows]
        for strategy, color, marker, label in (("rate_matched", style.SPLICEAI, "o", "match call volume"),
                ("agreement_optimal", style.OPENSPLICEAI, "s", "maximise MCC")):
            ax.plot(x, [r[strategy]["right_cutoff"] for r in rows], color=color,
                    marker=marker, label=label)
        ax.plot([0, 1], [0, 1], color=style.INK_MUTED, ls="--", lw=.8)
        ax.set_title(style.EVENT_TITLES[event])
        ax.set_xlabel("SpliceAI cutoff")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style.clean(ax)
    axes[0].set_ylabel("OpenSpliceAI cutoff")
    axes[0].legend(fontsize=7)
    _save(fig, out_dir, "f05_operating_point_transfer.png", index,
          "Per-event operating-point transfer at every configured threshold. Matching call volume "
          "and maximising MCC answer different questions; neither optimises biological accuracy. "
          "Candidate cutoffs use the histogram grid and exclude the unresolved endpoint at one.")


def fig_dominant_confusion(run: Run, out_dir: Path, index: FigureIndex) -> None:
    normalized = run.metrics["dominant_normalized"]["row_normalized"]
    labels = list(normalized)
    matrix = np.array([[normalized[left][right] for right in labels] for left in labels])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(matrix, cmap=style.SEQUENTIAL, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("OpenSpliceAI dominant event")
    ax.set_ylabel("SpliceAI dominant event")
    ax.set_title("Dominant-event agreement (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color=style.SURFACE if value > 0.55 else style.INK)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("share of SpliceAI row", color=style.INK_SECONDARY)
    cbar.outline.set_visible(False)
    _save(fig, out_dir, "f06_dominant_event.png", index,
          "Which event each predictor considers dominant, normalised within SpliceAI's "
          "category. TIE and NONE are explicit categories, not discarded.")


def fig_dp_agreement(run: Run, out_dir: Path, index: FigureIndex) -> None:
    tolerances = [0, 1, 2, 5, 10]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=True)
    for ax, event in zip(axes, EVENTS):
        for threshold, colour, marker in zip(run.raw["config"]["thresholds"], style.CATEGORICAL,
                                             style.MARKERS):
            entry = run.metrics["dp"][event][f"{threshold:.10g}"]
            rates = [np.nan if entry["within_rates"].get(str(t)) is None
                     else entry["within_rates"][str(t)] for t in tolerances]
            ax.plot(tolerances, rates, marker=marker, color=colour,
                    label=f"$\\geq$ {threshold:g}" if event == "AG" else None)
        ax.set_title(style.EVENT_TITLES[event])
        ax.set_xlabel("position tolerance (bp)")
        ax.set_xticks(tolerances)
        ax.set_ylim(0, 1.02)
        style.clean(ax)
    axes[0].set_ylabel("share of jointly-called pairs")
    axes[0].legend(title="both scores", loc="lower right")
    _save(fig, out_dir, "f07_dp_agreement.png", index,
          "Predicted-position agreement among variants both predictors call at a given "
          "score threshold. Every configured cutoff is shown. Denominators are jointly-called "
          "pairs; undefined groups are gaps and a measured zero remains zero.")


def fig_chromosome(run: Run, out_dir: Path, index: FigureIndex) -> None:
    """Where along the genome the two predictors diverge.

    Twenty-four bars said the correlation is flat everywhere. The 1-Mb blocks the
    reducer already computes say considerably more: the divergence is not uniform
    within a chromosome either, and it tracks how much splicing signal is present.
    """
    blocks = derive.genomic_landscape(run)
    if not blocks:
        return
    chroms, offsets, running = [], {}, 0.0
    for block in blocks:
        if block["chrom"] not in offsets:
            offsets[block["chrom"]] = running
            chroms.append(block["chrom"])
            running += 1.0
    xs, ys, sizes = [], [], []
    ranks = {chrom: 0 for chrom in chroms}
    for block in blocks:
        span = max(1, sum(1 for b in blocks if b["chrom"] == block["chrom"]))
        # Equal-width chromosome panels encode block order, not physical distance.
        # Using genomic start / occupied-block count spills across panels at gaps.
        xs.append(offsets[block["chrom"]] + (ranks[block["chrom"]] + 0.5) / span)
        ranks[block["chrom"]] += 1
        ys.append(block["bias"])
        sizes.append(block["n"])
    sizes = np.asarray(sizes, dtype=float)
    scaled = 4 + 26 * (sizes - sizes.min()) / max(1.0, sizes.max() - sizes.min())

    fig, ax = plt.subplots(figsize=(14.0, 3.6))
    for position, chrom in enumerate(chroms):
        if position % 2 == 0:
            ax.axvspan(position, position + 1, color=style.GRID, alpha=0.45, linewidth=0)
    ax.scatter(xs, ys, s=scaled, c=style.SPLICEAI, alpha=0.55, linewidths=0)
    ax.axhline(0.0, color=style.INK_MUTED, linewidth=1.0)
    genome = run.scores("MAX")["bias_right_minus_left"]
    ax.axhline(genome, color=style.OPENSPLICEAI, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(len(chroms) - 0.05, genome, " genome-wide", fontsize=7.5, va="center",
            color=style.OPENSPLICEAI)
    ax.set_xticks([offsets[c] + 0.5 for c in chroms])
    ax.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=7.5)
    ax.set_xlim(0, len(chroms))
    # A handful of extreme blocks otherwise compress the band every other block sits
    # in. Clip to the 0.5th percentile and say how many fall outside, rather than
    # letting three points set the scale for 2,600.
    values = np.asarray(ys, dtype=float)
    floor = float(np.percentile(values, 0.5))
    below = int((values < floor).sum())
    ax.set_ylim(floor, max(0.0005, float(values.max()) * 1.1))
    if below:
        ax.text(0.005, 0.04, f"{below} block(s) below {floor:.4f}, not shown",
                transform=ax.transAxes, fontsize=7.5, color=style.INK_SECONDARY)
    ax.set_xlabel("chromosome (1-Mb blocks, ordered along the genome)")
    ax.set_ylabel("mean MAX score: OpenSpliceAI - SpliceAI")
    ax.set_title("Divergence along the genome; point area is the number of paired annotations")
    style.clean(ax, grid="y")
    _save(fig, out_dir, "f08_chromosome.png", index,
          "Mean signed MAX difference in each 1-Mb block with at least 10,000 pairs. Blocks are "
          "ordered and evenly spaced within each chromosome; gaps are not drawn to scale. Point area scales "
          "with the number of paired annotations in the block. The dashed line is the genome-wide "
          "mean; blocks below it are where OpenSpliceAI scores relatively lower still.")


def fig_gene_divergence(run: Run, out_dir: Path, index: FigureIndex, top: int = 12,
                        minimum_n: int = 1000) -> None:
    """Every gene, not just the worst twenty.

    A top-20 bar chart cannot say whether those twenty are outliers or the tip of a
    continuum. Plotting all ~19,000 genes answers that, and naming the extremes
    keeps what the bar chart was for.
    """
    rows = [r for r in derive.stratum_rows(run, "gene") if r["n"] >= minimum_n]
    if not rows:
        raise ValueError("gene stratum is empty; cannot render the divergence figure")
    means = np.array([r["mean_spliceai"] for r in rows])
    biases = np.array([r["bias"] for r in rows])
    counts = np.array([r["n"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    scaled = 3 + 22 * (counts - counts.min()) / max(1.0, counts.max() - counts.min())
    axes[0].scatter(means, biases, s=scaled, c=style.SPLICEAI, alpha=0.35, linewidths=0)
    axes[0].axhline(0.0, color=style.INK_MUTED, linewidth=1.0)
    # The most divergent genes cluster tightly, so labelling the top N by bias alone
    # stacks them into an unreadable pile. Accept a label only when it clears the
    # ones already placed -- fewer names, all of them legible.
    span_x = max(1e-9, float(means.max() - means.min()))
    span_y = max(1e-9, float(biases.max() - biases.min()))
    placed: List[tuple] = []
    for row in sorted(rows, key=lambda r: r["bias"]):
        if len(placed) >= top:
            break
        x, y = row["mean_spliceai"], row["bias"]
        if any(abs(x - px) / span_x < 0.06 and abs(y - py) / span_y < 0.05
               for px, py in placed):
            continue
        placed.append((x, y))
        axes[0].annotate(row["stratum"], (x, y), textcoords="offset points",
                         xytext=(5, -1), fontsize=6.8, color=style.INK_SECONDARY)
    axes[0].set_xlabel("mean SpliceAI score in the gene")
    axes[0].set_ylabel("mean OpenSpliceAI - SpliceAI")
    axes[0].set_title(f"All {len(rows):,} genes with $\\geq$ {minimum_n:,} paired annotations")
    style.clean(axes[0], grid="both")

    axes[1].hist(biases, bins=60, color=style.NEUTRAL_FILL, linewidth=0)
    axes[1].axvline(0.0, color=style.INK_MUTED, linewidth=1.0)
    axes[1].axvline(float(np.median(biases)), color=style.OPENSPLICEAI, linewidth=1.6)
    axes[1].text(float(np.median(biases)), axes[1].get_ylim()[1] * 0.94, " median gene",
                 fontsize=7.5, color=style.OPENSPLICEAI)
    axes[1].set_xlabel("mean OpenSpliceAI - SpliceAI")
    axes[1].set_ylabel("genes")
    axes[1].set_title("The divergence is a continuum, not a few outliers")
    style.clean(axes[1])
    _save(fig, out_dir, "f09_gene_divergence.png", index,
          "Per-gene divergence against how much splice signal the gene carries. Point area is the "
          "number of paired annotations; the labelled genes are the most negative. The histogram "
          "shows the same values as a distribution.")


def _seed_of(arm: str) -> str:
    """The checkpoint an arm used, recovered from its name (``C_rs10_matched`` -> ``rs10``)."""
    for token in arm.replace("-", "_").split("_"):
        if token.startswith("rs") and token[2:].isdigit():
            return token
    return arm


def fig_seed_versus_model(seed_run: Run, model_runs: Sequence[Run], out_dir: Path,
                          index: FigureIndex) -> None:
    """Training-seed disagreement against method disagreement, per event.

    The aggregate MAX comparison hides the finding: the method gap exceeds the
    seed gap by very different factors on losses and on gains. Showing the two
    error levels side by side per event, and their ratio, is what makes that
    visible.
    """
    comparison = derive.seed_versus_model(seed_run, model_runs)
    events = comparison["events"]
    arms = sorted(events["MAX"]["models"])
    labels = list(SCORE_LABELS)
    xs = np.arange(len(labels))
    series = [("two training seeds", style.CATEGORICAL[2],
               [events[k]["seed"]["mae"] for k in labels])]
    for offset, arm in enumerate(arms):
        series.append((f"SpliceAI vs {_seed_of(arm)}", style.CATEGORICAL[offset],
                       [events[k]["models"][arm]["mae"] for k in labels]))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.6))
    width = 0.26
    for position, (name, colour, values) in enumerate(series):
        axes[0].bar(xs + (position - 1) * width, values, width=width * 0.92,
                    color=colour, label=name)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([style.EVENT_TITLES[k] for k in labels], fontsize=8)
    axes[0].set_ylabel("mean absolute difference")
    axes[0].set_title("How far apart are the scores?")
    axes[0].legend(loc="upper left")
    style.clean(axes[0])

    ratios = [events[k]["ratios"][arms[0]]["mae"] for k in labels]
    colours = [style.OPENSPLICEAI if value and value >= 2.0 else style.SEED_ALT for value in ratios]
    axes[1].bar(xs, ratios, width=0.6, color=colours)
    axes[1].axhline(1.0, color=style.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[1].text(len(labels) - 0.45, 1.06, "equal to seed variation", fontsize=7.5,
                 color=style.INK_SECONDARY, ha="right")
    for x, value in zip(xs, ratios):
        if value is not None:
            axes[1].text(x, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8.5,
                         color=style.INK)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([style.EVENT_TITLES[k] for k in labels], fontsize=8)
    axes[1].set_ylabel("method error / seed error")
    axes[1].set_ylim(0, max(v for v in ratios if v) * 1.22)
    axes[1].set_title("Is the method gap bigger than retraining?")
    style.clean(axes[1])
    _save(fig, out_dir, "f10_seed_versus_model.png", index,
          "Left: mean absolute difference between two training seeds of OpenSpliceAI, and between "
          "SpliceAI and each seed, on identical three-way variant/gene observations. Right: rs10 "
          "method MAE divided by seed MAE, per event. A value near 1 indicates similar average "
          "score differences for these contrasts; it does not establish interchangeability.")


def fig_site_distance(run: Run, out_dir: Path, index: FigureIndex,
                      threshold: float = 0.5) -> None:
    """Where each predictor's calls sit relative to real annotated splice sites.

    This is the one figure that brings in information from outside the two score
    files. The sites are derived from the same gene table both predictors were
    scored against, so the comparison is annotation-neutral. Note the strata carry
    the *maximum* delta score only, which is the quantity a caller thresholds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    pooled = derive.site_distance_table(run, threshold=threshold)
    if not pooled:
        raise ValueError("no site_distance stratum in this run")
    order = [row["distance"] for row in pooled]
    xs = np.arange(len(order))

    # (A) call rate against distance, both predictors
    axes[0].plot(xs, [r["spliceai_call_rate"] for r in pooled], marker="o",
                 color=style.SPLICEAI, label="SpliceAI")
    axes[0].plot(xs, [r["openspliceai_call_rate"] for r in pooled], marker="s",
                 color=style.OPENSPLICEAI, label="OpenSpliceAI")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"share of variants called at $\\geq$ {threshold:g}")
    axes[0].set_title("Call rate against distance to an annotated site")
    axes[0].legend(loc="upper right")

    # (B) the ratio, which is where the two part company
    ratios = [r["call_rate_ratio"] for r in pooled]
    axes[1].bar(xs, [np.nan if v is None else v for v in ratios], width=0.62,
                color=[style.OPENSPLICEAI if (v or 0) >= 1 else style.SPLICEAI for v in ratios])
    axes[1].axhline(1.0, color=style.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[1].set_ylabel("OpenSpliceAI calls / SpliceAI calls")
    axes[1].set_title("Which predictor calls more, and where")

    # (C) how much of the called set is shared
    axes[2].plot(xs, [np.nan if r["jaccard"] is None else r["jaccard"] for r in pooled],
                 marker="^", color=style.NEUTRAL_FILL)
    axes[2].set_ylabel("Jaccard of the two call sets")
    axes[2].set_title("Agreement among the calls that are made")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
        ax.set_xlabel("distance to nearest annotated splice site (bp)")
        style.clean(ax)
    _save(fig, out_dir, "f12_site_distance.png", index,
          "Maximum-delta-score behaviour as a function of distance to the nearest annotated "
          "splice site, derived from the same gene table both predictors were scored against. "
          "A call far from any annotated site is not necessarily wrong -- it may be a genuine "
          "unannotated site -- so this bounds the question rather than settling it.")


def fig_site_distance_by_type(run: Run, out_dir: Path, index: FigureIndex,
                              thresholds: Sequence[float] = (0.1, 0.5, 0.8)) -> None:
    """Call rate by distance at several stringencies, split by nearest-site type."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7), sharey=True)
    for ax, site_type in zip(axes, derive.SITE_TYPES):
        rows0 = derive.site_distance_table(run, threshold=thresholds[0], site_type=site_type)
        if not rows0:
            continue
        order = [r["distance"] for r in rows0]
        xs = np.arange(len(order))
        for threshold, alpha in zip(thresholds, (0.42, 0.72, 1.0)):
            rows = derive.site_distance_table(run, threshold=threshold, site_type=site_type)
            lookup = {r["distance"]: r for r in rows}
            ax.plot(xs, [lookup[d]["spliceai_call_rate"] for d in order], marker="o",
                    color=style.SPLICEAI, alpha=alpha,
                    label=f"SpliceAI $\\geq$ {threshold:g}")
            ax.plot(xs, [lookup[d]["openspliceai_call_rate"] for d in order], marker="s",
                    color=style.OPENSPLICEAI, alpha=alpha,
                    label=f"OpenSpliceAI $\\geq$ {threshold:g}")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
        ax.set_xlabel("distance to nearest site (bp)")
        ax.set_title(f"Nearest site is an {site_type}" if site_type == "acceptor"
                     else f"Nearest site is a {site_type}")
        style.clean(ax)
    axes[0].set_ylabel("share of variants called")
    axes[0].legend(loc="lower left", fontsize=6.8, ncol=2)
    _save(fig, out_dir, "f13_site_distance_by_type.png", index,
          "Call rate against distance at three stringencies, split by whether the nearest "
          "annotated site is an acceptor or a donor. Deeper colour is a stricter cutoff.")


def fig_quantization(run: Run, out_dir: Path, index: FigureIndex) -> None:
    rows = derive.quantization_profile(run)
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    width = 0.38
    axes[0].bar(xs - width / 2, [r["raw_exact_match_rate"] for r in rows], width=width,
                color=style.SPLICEAI, label="as published (5 dp vs 2 dp)")
    axes[0].bar(xs + width / 2, [r["rounded_exact_match_rate"] for r in rows], width=width,
                color=style.OPENSPLICEAI, label="OpenSpliceAI rounded to 2 dp")
    axes[0].set_ylabel("exact-match rate")
    axes[0].set_title("Agreement recovered by matching output precision")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylim(0, 1.3)
    axes[1].bar(xs, [r["share_of_mae_beyond_quantization"] for r in rows], width=0.6,
                color=style.NEUTRAL_FILL)
    for x, r in zip(xs, rows):
        axes[1].text(x, r["share_of_mae_beyond_quantization"], f"{r['share_of_mae_beyond_quantization']:.2f}",
                     ha="center", va="bottom", fontsize=8, color=style.INK)
    axes[1].set_ylabel("share of MAE")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Disagreement surviving the $\\pm$0.005 rounding band")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([r["label"] for r in rows])
        style.clean(ax)
    _save(fig, out_dir, "f11_quantization.png", index,
          "SpliceAI publishes two decimals. Left: how much exact agreement is recovered by "
          "rounding OpenSpliceAI to the same grid. Right: the fraction of mean absolute "
          "difference that cannot be explained by that grid.")


def render_all(runs: Mapping[str, Run], primary: str, seed_arm: str,
               model_arms: Sequence[str], out_dir: Path) -> FigureIndex:
    style.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    index: FigureIndex = []
    run = runs[primary]
    fig_coverage(run, out_dir, index)
    fig_joint_density(run, out_dir, index)
    fig_difference_distributions(run, out_dir, index)
    fig_threshold_agreement(run, out_dir, index)
    fig_operating_point(run, out_dir, index)
    fig_dominant_confusion(run, out_dir, index)
    fig_dp_agreement(run, out_dir, index)
    fig_chromosome(run, out_dir, index)
    fig_gene_divergence(run, out_dir, index)
    if seed_arm in runs and all(arm in runs for arm in model_arms):
        fig_seed_versus_model(runs[seed_arm], [runs[a] for a in model_arms], out_dir, index)
    fig_quantization(run, out_dir, index)
    # Only present when the run carried the opt-in site_distance stratum.
    if run.raw.get("depth"):
        from . import depth_figures
        depth_figures.render(run, runs.get(seed_arm), [runs[a] for a in model_arms], out_dir, index)
    elif run.metrics["strata"].get("site_distance"):
        fig_site_distance(run, out_dir, index)
        fig_site_distance_by_type(run, out_dir, index)
    return index
