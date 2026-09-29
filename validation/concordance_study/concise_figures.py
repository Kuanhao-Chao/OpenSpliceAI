"""Display-only revision figures; all scientific inputs remain frozen."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import style, concise_supplement
from .figures import _column_quantiles
from .loading import EVENTS, SCORE_LABELS, load_run
from .deeper import histograms

DISTANCES = ("at_site", "1-2", "3-10", "11-50", "51-500", ">500")
DISTANCE_LABELS = ("At site", "1–2", "3–10", "11–50", "51–500", ">500")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pool_site_rates(rows, event, threshold):
    """Pool numerators and denominators over site types, never average rates."""
    pooled = {}
    for row in rows:
        if row["event"] != event or row["threshold"] != threshold:
            continue
        n, both, left, right = (row[k] for k in
                               ("n", "both_positive", "left_only", "right_only"))
        if min(n, both, left, right) < 0 or both + left + right > n:
            raise ValueError("inconsistent site call counts")
        group = pooled.setdefault(row["distance"], [0, 0, 0])
        group[0] += n
        group[1] += both + left
        group[2] += both + right
    result = []
    for distance in DISTANCES:
        if distance in pooled:
            n, left, right = pooled[distance]
            result.append(dict(distance=distance, n=n, left=left, right=right,
                               left_rate=left / n if n else None,
                               right_rate=right / n if n else None,
                               ratio=right / left if left else None))
    return result


def display_arrays(study, out):
    """Extract only stored histograms once; bind the cache to the source digest."""
    source = study / "production/A_rs10_genomewide/summary.json"
    source_hash = digest(source)
    cache, receipt = out / "display_histograms.npz", out / "display_histograms.json"
    if cache.exists() and receipt.exists():
        metadata = json.loads(receipt.read_text())
        if metadata.get("schema_version") == 2 and metadata.get("source_sha256") == source_hash and metadata.get("cache_sha256") == digest(cache):
            with np.load(cache, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}
    print("Reading frozen primary histograms (no analysis rerun)", flush=True)
    run = load_run(source, "A_rs10_genomewide", "primary", expected_chunks=99283)
    arrays = {"bins": np.asarray(run.score_bins)}
    for event in SCORE_LABELS:
        arrays["joint_" + event] = np.asarray(run.joint_hist(event), dtype=np.int64).reshape(run.score_bins, run.score_bins)
        if arrays["joint_" + event].sum() != run.paired_n():
            raise ValueError("joint histogram does not cover the frozen primary domain")
        for label, distances in (("near", ("at_site", "1-2")), ("far", (">500",))):
            left, right, joint = histograms(run, distances, event)
            arrays[label + "_" + event] = joint.astype(np.int64)
            arrays[label + "_left_" + event] = left.astype(np.int64)
            arrays[label + "_right_" + event] = right.astype(np.int64)
            if not (np.array_equal(left, joint.sum(axis=1)) and np.array_equal(right, joint.sum(axis=0))):
                raise ValueError("stored context marginals disagree with joint histogram")
    np.savez_compressed(cache, **arrays)
    receipt.write_text(json.dumps({"schema_version": 2, "source": str(source), "source_sha256": source_hash,
                                  "cache_sha256": digest(cache)}, indent=2) + "\n")
    return arrays


def _save(fig, directory, name):
    for suffix in (".png", ".pdf"):
        fig.savefig(directory / (name + suffix), bbox_inches="tight", facecolor=style.SURFACE)
    plt.close(fig)


def _panels(axes):
    for letter, ax in zip("ABCDEF", np.asarray(axes).flat):
        style.panel_label(ax, letter, x=-0.10, y=1.02)
        style.clean(ax)


def progress_figure(facts, progress, directory):
    fig = plt.figure(figsize=(10, 5.5))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.6, 1))
    ax, info = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    colors = {"valid": style.SPLICEAI, "missing": "#d8d8d2",
              "invalid": "#bb493b", "unverified": style.NEUTRAL_FILL}
    labels = {"valid": "Audited valid, metadata unchanged", "missing": "Missing",
              "invalid": "Invalid", "unverified": "Changed / unverified"}
    for y, seed in enumerate(("rs10", "rs13")):
        data = progress["seeds"][seed]
        counts = data["current_counts"]
        if counts is None:
            raise ValueError("progress figure requires a filesystem check")
        start = 0
        for state, color in colors.items():
            percent = counts[state] / data["total_chunks"] * 100
            ax.barh(y, percent, left=start, height=.3, color=color,
                    label=labels[state] if y == 0 else None)
            start += percent
        valid = counts["valid"]
        ax.text(0, y-.23, f"{seed}    {valid:,} / {data['total_chunks']:,} valid "
                f"({valid / data['total_chunks']:.3%})", fontsize=10, fontweight="bold")
        jobs = data["jobs"]
        reasons = {"AssocGrpBillingMinutes": "account billing-minute limit", "Dependency": "waiting for dependency"}
        status = "; ".join(f"{j['state'].lower()}: {reasons.get(j['reason'].strip('()'), j['reason'])}" for j in jobs)
        if progress["scheduler"]["status"] != "available":
            status = "Scheduler status unavailable"
        elif not jobs:
            status = "No matching live scoring jobs"
        rest = ", ".join(f"{counts[s]:,} {s}" for s in ("missing", "invalid", "unverified"))
        ax.text(0, y+.26, rest + "  |  " + status, fontsize=8, va="top")
    ax.set_ylim(1.55, -.55)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of the 100,000 source chunks (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, fontsize=8)
    style.clean(ax, "x")
    info.axis("off")
    primary = facts["primary"]
    matched = facts["arms"]["B_seeds_rs10_rs13"]
    info.text(0, .75, "Frozen scientific analysis • September 9 snapshot", weight="bold", fontsize=11)
    info.text(0, .48, f"Primary: {primary['chunks']:,} chunks → "
              f"{primary['coverage']['paired_annotations']:,} variant–gene pairs", fontsize=10)
    info.text(0, .24, f"Three-way comparison: {matched['chunks']:,} shared chunks → "
              f"{matched['coverage']['paired_annotations']:,} common variant–gene pairs", fontsize=10)
    audits = sorted({v["audit_at_utc"][:10] for v in progress["seeds"].values()})
    info.text(0, -.01, "Content audit: " + ", ".join(audits) + "  |  Snapshot completed: "
              + progress["observed_at_utc"].replace("T", " "), fontsize=8, color=style.INK_SECONDARY)
    _save(fig, directory, "main01_progress")


def score_figure(facts, arrays, directory):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    bins = int(arrays["bins"])
    x = (np.arange(bins)+.5)/bins
    for ax, event in zip(axes.flat, EVENTS):
        q = _column_quantiles(arrays["joint_"+event], bins, (.05, .25, .5, .75, .95))
        ax.vlines(x, q[.05], q[.95], color=style.OPENSPLICEAI, alpha=.16, linewidth=2)
        ax.vlines(x, q[.25], q[.75], color=style.OPENSPLICEAI, alpha=.45, linewidth=2)
        ax.plot(x, q[.5], ".", color=style.OPENSPLICEAI, markersize=3, label="Conditional median")
        ax.plot([0, 1], [0, 1], "--", color=style.INK_MUTED, linewidth=1, label="Equal scores")
        r = facts["primary"]["agreement"][event]["pearson_r"]
        ax.set_title(f"{style.EVENT_TITLES[event]} ({event})   r = {r:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="SpliceAI delta score", ylabel="OpenSpliceAI delta score")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    _panels(axes)
    _save(fig, directory, "main02_score_agreement")


def threshold_figure(facts, directory, points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for event, color, marker in zip(EVENTS, style.CATEGORICAL, style.MARKERS):
        rows = [row for row in points if row["event"] == event]
        for ax, metric in zip(axes, ("jaccard", "call_rate_ratio_right_over_left")):
            ax.plot([r["cutoff"] for r in rows], [r[metric] for r in rows],
                    color=color, label=style.EVENT_TITLES[event], marker=marker,
                    markevery=25, markersize=4)
            exact = facts["primary"]["thresholds"][event]["0.5"]
            ax.scatter([.5], [exact[metric]], color=color, marker=marker, s=48, zorder=5)
        axes[0].annotate(f"{facts['primary']['thresholds'][event]['0.5']['jaccard']:.3f}",
                         (.5, facts["primary"]["thresholds"][event]["0.5"]["jaccard"]),
                         xytext=(7, -12 if event == "AL" else 5), textcoords="offset points",
                         fontsize=8, color=color)
    for ax in axes:
        ax.set(xlim=(0, 1), xlabel="Delta-score cutoff applied to both models")
        ax.axvline(.5, color=style.INK_MUTED, linewidth=.8, ls=":")
    axes[0].set(ylabel="Jaccard: shared / union of positive calls", ylim=(0, .8),
                title="Overlap of predicted effects")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].axhline(1, color=style.INK_MUTED, ls="--", linewidth=1)
    axes[1].set(yscale="log", ylabel="OpenSpliceAI / SpliceAI call count",
                title="Relative number of calls")
    _panels(axes)
    _save(fig, directory, "main03_call_agreement")


def context_figure(facts, arrays, directory):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for ax, event in zip(axes.flat, EVENTS):
        rows = pool_site_rates(facts["primary"]["site_event_table"], event, .5)
        for side, color, marker, name in (("left", style.SPLICEAI, "o", "SpliceAI"),
                                          ("right", style.OPENSPLICEAI, "s", "OpenSpliceAI")):
            rates = [r[side+"_rate"] for r in rows]
            ax.plot(range(len(rows)), [v if v else np.nan for v in rates],
                    color=color, marker=marker, label=name)
            for i, value in enumerate(rates):
                if value == 0:
                    ax.annotate("0", (i, 1.3e-8 if side == "left" else 4e-8),
                                color=color, ha="center", fontsize=8)
        ax.set(yscale="log", ylim=(1e-8, 1), title=style.EVENT_TITLES[event],
               ylabel="Fraction with event score ≥ 0.5",
               xlabel="Variant distance to an internal boundary (bp)")
        ax.set_xticks(range(len(rows)), [DISTANCE_LABELS[DISTANCES.index(r["distance"])] for r in rows])
    axes[0, 0].legend(fontsize=8)
    bins = int(arrays["bins"])
    x = (np.arange(bins)+.5)/bins
    for ax, event in zip(axes[2], ("AG", "DG")):
        for cohort, color, line, name in (("near", style.NEUTRAL_FILL, "-", "At / within 2 bp"),
                                          ("far", style.SEED_ALT, "--", "More than 500 bp")):
            q = _column_quantiles(arrays[cohort+"_"+event], bins, (.5,))
            ax.plot(x, q[.5], color=color, ls=line, marker="o" if cohort == "near" else "s",
                    markersize=2.5, label=name)
        ax.plot([0, 1], [0, 1], "--", color=style.INK_MUTED, linewidth=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=style.EVENT_TITLES[event]+": score agreement",
               xlabel="SpliceAI delta score", ylabel="Conditional median OpenSpliceAI score")
    axes[2, 0].legend(fontsize=8)
    _panels(axes)
    _save(fig, directory, "main04_boundary_context")


def position_figure(facts, directory):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    rows = []
    for i, event in enumerate(EVENTS):
        row = facts["primary"]["dp"][event]["0.5"]
        rows.append(row)
        rate = row["within_0bp"]
        ax.scatter(rate * 100, i, color=style.NEUTRAL_FILL, s=55, zorder=3)
        ax.text(rate*100-.15, i-.15, f"{rate:.6%}" if rate > .9999 and rate < 1 else f"{rate:.2%}", ha="right", fontsize=10)
    ax.set_yticks(range(4), [style.EVENT_TITLES[e] for e in EVENTS])
    ax.set(xlim=(90, 101), ylim=(3.55, -.65), xlabel="Identical predicted position among jointly called annotations (%)")
    ax.axvline(100, ls=":", color=style.INK_MUTED, lw=1)
    for i, row in enumerate(rows):
        ax.text(90.2, i+.22, f"Exact: {row['within_0bp_n']:,} / {row['eligible']:,}  |  Mismatches: {row['eligible']-row['within_0bp_n']:,}", fontsize=8, color=style.INK_SECONDARY)
    ax.set_title("Both event scores ≥ 0.5 • selected position, not score magnitude")
    style.clean(ax, "x")
    _save(fig, directory, "main05_positions")


def seed_figure(facts, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    events = facts["seed_versus_model"]["events"]
    arms = ("C_rs10_matched", "D_rs13_matched")
    xs = np.arange(4)
    series = [("rs10 vs rs13", style.SEED_ALT, [events[e]["seed"]["mae"] for e in EVENTS])]
    for arm, seed, color in zip(arms, ("rs10", "rs13"), (style.SPLICEAI, style.OPENSPLICEAI)):
        series.append(("SpliceAI vs "+seed, color, [events[e]["models"][arm]["mae"] for e in EVENTS]))
    for i, (label, color, values) in enumerate(series):
        axes[0].bar(xs+(i-1)*.25, values, width=.23, label=label, color=color)
    for i, (arm, seed, color) in enumerate(zip(arms, ("rs10", "rs13"), (style.SPLICEAI, style.OPENSPLICEAI))):
        values = [events[e]["ratios"][arm]["mae"] for e in EVENTS]
        axes[1].bar(xs+(i-.5)*.32, values, width=.3, label=seed, color=color)
        for x, y in zip(xs+(i-.5)*.32, values):
            axes[1].text(x, y+.045, f"{y:.2f}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(xs, EVENTS)
    axes[0].set(ylabel="Mean absolute score difference", ylim=(0, .0019),
                title="Three comparisons on identical observations")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color=style.INK_MUTED, ls="--", linewidth=1)
    axes[1].set(ylabel="Method difference / observed seed difference", ylim=(0, 4.6),
                title="Gain differences exceed the seed contrast")
    axes[1].legend(title="SpliceAI vs", fontsize=8, loc="upper right")
    _panels(axes)
    _save(fig, directory, "main06_seed_comparison")


def chromosome_figure(facts, directory, blocks):
    chroms = list(dict.fromkeys(b["chrom"] for b in blocks))
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, chrom in enumerate(chroms):
        group = [b for b in blocks if b["chrom"] == chrom]
        if i % 2 == 0:
            ax.axvspan(i, i+1, color=style.GRID, alpha=.4)
        ax.scatter(i+(np.arange(len(group))+.5)/len(group), [b["bias"] for b in group],
                   s=[4+25*b["n"]/max(c["n"] for c in blocks) for b in group],
                   color=style.SPLICEAI, alpha=.55, linewidths=0)
    ax.axhline(0, color=style.INK_MUTED, lw=1)
    ax.axhline(facts["primary"]["agreement"]["MAX"]["bias"], color=style.OPENSPLICEAI, ls="--",
               label="Primary mean")
    ax.set_xticks(np.arange(len(chroms))+.5, [c.replace("chr", "") for c in chroms])
    ax.set(xlim=(0, len(chroms)), xlabel="Chromosome (retained 1-Mb blocks ordered within each chromosome)",
           ylabel="Mean MAX difference: OpenSpliceAI − SpliceAI", title="Regional variation, including all retained extreme blocks")
    ax.legend()
    style.clean(ax)
    _save(fig, directory, "supp10_chromosome")


SUPPLEMENT_SOURCES = (
    ("supp01_coverage", "f01_coverage"), ("supp02_tails", "f03_difference_distributions"),
    ("supp03_threshold_transfer", "f05_operating_point_transfer"), ("supp04_precision", "f11_quantization"),
    ("supp05_dominant_event", "f06_dominant_event"), ("supp06_site_distributions", "f13_site_score_distributions"),
    ("supp07_site_score_agreement", "f14_site_conditioned_calibration"),
    ("supp08_site_positions", "f15_site_conditioned_dp"),
    ("supp09_seed_by_gene", "f16_seed_method_by_gene"),
    ("supp11_gene_divergence", "f09_gene_divergence"),
)


def render(facts, progress, study, out):
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    style.apply_style()
    arrays = display_arrays(study, out)
    progress_figure(facts, progress, directory)
    score_figure(facts, arrays, directory)
    with (study / "publication/tables/A_rs10_genomewide/agreement_curves.csv").open() as handle:
        points = list(csv.DictReader(handle))
    for row in points:
        for key in ("cutoff", "jaccard", "call_rate_ratio_right_over_left"):
            row[key] = float(row[key]) if row[key] else None
    threshold_figure(facts, directory, points)
    context_figure(facts, arrays, directory)
    position_figure(facts, directory)
    seed_figure(facts, directory)
    with (study / "publication/tables/A_rs10_genomewide/stratum_block_1mb.csv").open() as handle:
        blocks = [dict(chrom=r["stratum"].split(":")[0], start=int(r["stratum"].split(":")[1].split("-")[0]),
                       n=int(r["n"]), bias=float(r["bias"])) for r in csv.DictReader(handle) if int(r["n"]) >= 10000]
    order = {"chr" + c: i for i, c in enumerate([str(i) for i in range(1, 23)] + ["X", "Y", "M"])}
    blocks.sort(key=lambda r: (order.get(r["chrom"], 99), r["start"]))
    if len(blocks) != facts["primary"]["landscape"]["blocks"]:
        raise ValueError("block display domain differs from frozen facts")
    chromosome_figure(facts, directory, blocks)
    for target, source in SUPPLEMENT_SOURCES:
        for suffix in (".png", ".pdf"):
            shutil.copy2(study / "publication/figures" / (source+suffix), directory / (target+suffix))
    table_dir = out / "figure_data"
    table_dir.mkdir(exist_ok=True)
    rows = [dict(event=e, **row) for e in EVENTS for row in pool_site_rates(facts["primary"]["site_event_table"], e, .5)]
    with (table_dir / "boundary_call_rates.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    concise_supplement.render(facts, arrays, study, out, _save)
    return directory
