#!/usr/bin/env python
"""
make_report_figs.py — generate the figures for the OpenSpliceAI technical report.

Throwaway/reproducible plotting script (NOT part of the openspliceai package).
Renders 6 PNGs straight into the website's report-assets directory, in a clean
LiftOn-report-like aesthetic (sans-serif, green = OpenSpliceAI, grey = SpliceAI).

Run:
  /home/kchao10/miniconda3/envs/pytorch_cuda/bin/python validation/make_report_figs.py

Data sources (real, on-disk / live):
  - validation/benchmark_out/{osai,spliceai}.json  (per-position accuracy + wall time)
  - verified variant delta-score findings (delta-score-comparison analysis)
  - live campaign status (./campaign.sh status)  -> passed in as constants below
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark_out")
ASSETS = "/home/kchao10/data_ssalzbe1/khchao/Kuanhao-Chao.github.io/src/assets/reports/openspliceai-technical-report"
os.makedirs(ASSETS, exist_ok=True)

# ----------------------------------------------------------------------------- style
OSAI = "#2f9e44"   # green  = OpenSpliceAI
SAI  = "#868e96"   # grey   = SpliceAI / baseline
OSAI_D = "#1f7a32"
INK  = "#212529"
ACCENT = "#1c7ed6"  # blue accent for schematics / highlights
GAIN = "#e8590c"    # orange for "gain" side

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#495057",
    "axes.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})

def _clean(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#dee2e6", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

def _panel_label(ax, s, x=-0.085, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left")

def save(fig, name):
    out = os.path.join(ASSETS, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

# ============================================================================= data
with open(os.path.join(BENCH, "osai.json")) as fh:
    OS = json.load(fh)
with open(os.path.join(BENCH, "spliceai.json")) as fh:
    SP = json.load(fh)

# live campaign numbers (from ./campaign.sh status at write time)
RS10_DONE, RS13_DONE, NCHUNK = 11380, 1692, 100000
SEC_PER_CHUNK, CONC = 122.0, 4
REC_PER_CHUNK = 34334            # SNVs scored per chunk (mean)


# ============================================================================= FIG 2: accuracy
def fig_accuracy():
    metrics = [("top0.5L", "Top-0.5L"), ("top1L", "Top-1L"), ("top2L", "Top-2L"),
               ("top4L", "Top-4L"), ("auprc", "AUPRC"), ("f1", "F1")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for ax, cls, lab in zip(axes, ["donor", "acceptor"], ["Donor", "Acceptor"]):
        o = [OS[cls][k] for k, _ in metrics]
        s = [SP[cls][k] for k, _ in metrics]
        x = np.arange(len(metrics)); w = 0.38
        b1 = ax.bar(x - w/2, o, w, label="OpenSpliceAI", color=OSAI, zorder=3)
        b2 = ax.bar(x + w/2, s, w, label="SpliceAI", color=SAI, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], rotation=30, ha="right")
        ax.set_ylim(0.84, 1.005)
        ax.set_ylabel("score")
        ax.set_title(f"{lab}  (n$_{{true}}$ = {OS[cls]['n_true']:,})")
        _clean(ax)
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.3f}", (r.get_x()+r.get_width()/2, r.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.6, color="#343a40")
    axes[0].legend(frameon=False, loc="lower left", fontsize=10)
    _panel_label(axes[0], "A"); _panel_label(axes[1], "B")
    fig.suptitle("Per-position accuracy on the human MANE test set (10,000 nt, 5-model ensembles)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "rfig_accuracy.png")


# ============================================================================= FIG 3: speed
def fig_speed():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    # (A) per-position benchmark wall time
    ax = axes[0]
    tools = ["OpenSpliceAI", "SpliceAI\n(Keras)"]
    secs = [OS["seconds"], SP["seconds"]]
    cols = [OSAI, SAI]
    b = ax.bar(tools, secs, color=cols, width=0.6, zorder=3)
    for r in b:
        ax.annotate(f"{r.get_height():.0f} s", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_ylabel("wall-clock (s)")
    ax.set_ylim(0, max(secs)*1.18)
    ax.set_title("Whole-test-set scoring\n(420.58 M positions, same GPU)")
    sp = SP["seconds"]/OS["seconds"]
    ax.text(0.5, 0.92, f"{sp:.2f}× faster", transform=ax.transAxes, ha="center",
            fontsize=11, color=OSAI_D, fontweight="bold")
    _clean(ax)
    # (B) variant scorer throughput: per-variant vs batched
    ax = axes[1]
    labels = ["per-variant\n(original)", "batched  -b 128\n(+ ref dedup)"]
    rel = [1.0, 6.7]
    b = ax.bar(labels, rel, color=[SAI, OSAI], width=0.6, zorder=3)
    for r, v in zip(b, rel):
        ax.annotate(f"{v:.1f}×", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_ylabel("relative throughput")
    ax.set_ylim(0, 7.7)
    ax.set_title("Variant delta-score engine\n(≈122 s / 34,334-SNV chunk, A100)")
    _clean(ax)
    _panel_label(axes[0], "A"); _panel_label(axes[1], "B")
    fig.suptitle("Inference speed", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "rfig_speed.png")


# ============================================================================= FIG 4: variant concordance
def fig_variant():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    # (A) concordance of delta scores
    ax = axes[0]
    names = ["Pearson\n(max-DS)", "Spearman\n(max-DS)", "κ @0.5", "Jaccard\n(pos. class)",
             "loss agree\n(AL/DL)", "gain agree\n(AG/DG)"]
    vals  = [0.763, 0.47, 0.67, 0.50, 0.84, 0.64]
    cols  = ["#495057", "#495057", "#495057", "#495057", OSAI, GAIN]
    b = ax.bar(names, vals, color=cols, width=0.66, zorder=3)
    for r, v in zip(b, vals):
        ax.annotate(f"{v:.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("agreement")
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.8)
    ax.set_title("Delta-score concordance with SpliceAI")
    _clean(ax)
    # (B) discrimination: AUROC / AUPRC, OSAI vs SAI
    ax = axes[1]
    groups = ["AUROC\n(splice-loss)", "AUPRC\n(max-DS)"]
    o = [0.95, 0.458]; s = [0.81, 0.417]
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w/2, o, w, label="OpenSpliceAI", color=OSAI, zorder=3)
    b2 = ax.bar(x + w/2, s, w, label="SpliceAI", color=SAI, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x()+r.get_width()/2, r.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.02); ax.set_ylabel("score")
    ax.set_title("Discrimination: tied power, better specificity")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    _clean(ax)
    _panel_label(axes[0], "A"); _panel_label(axes[1], "B")
    fig.suptitle("Variant delta-score comparison vs SpliceAI  (452 K paired SNVs; mask caveat in text)",
                 fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "rfig_variant_concordance.png")


# ============================================================================= helpers for schematics
def box(ax, x, y, w, h, text, fc, ec="#343a40", fs=10, tc=INK, lw=1.1, rounded=0.02, fw="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={rounded}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, fontweight=fw)

def arrow(ax, x0, y0, x1, y1, color="#343a40", lw=1.4, style="-|>", ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=2))


# ============================================================================= FIG 1: architecture
def fig_architecture():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11.0, 6.6),
                                   gridspec_kw={"height_ratios": [1.55, 1.0]})
    # ---- (A) the network ----
    ax = axA; ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    _panel_label(ax, "A", x=0.0, y=1.02)
    ax.text(50, 29, "SpliceAI residual dilated 1-D CNN", ha="center", fontsize=12.5,
            fontweight="bold")
    box(ax, 1, 11, 11, 8, "one-hot DNA\n(4 ch)\nSL+CL$_{max}$\n= 15,000", "#e7f5ff", ec=ACCENT, fs=8.5)
    arrow(ax, 12, 15, 16, 15)
    box(ax, 16, 11.5, 8, 7, "Conv1d\nstem\n(L=32)", "#f1f3f5", fs=8.5)
    arrow(ax, 24, 15, 27, 15)
    # residual stack: 4 groups of (ResUnit x4 + Skip)
    gx = 27
    for i in range(4):
        box(ax, gx, 9.5, 8.6, 11, f"ResidualUnit\n× 4\n(W,AR group {i+1})", "#ebfbee",
            ec=OSAI, fs=7.6)
        # skip tap
        box(ax, gx+1.8, 21.2, 5, 3.0, "Skip 1×1", "#fff9db", ec="#f08c00", fs=7.2)
        arrow(ax, gx+4.3, 20.5, gx+4.3, 21.2, color="#f08c00", lw=1.0, ms=6)
        if i < 3:
            arrow(ax, gx+8.6, 15, gx+9.6, 15)
        gx += 9.6
    # skip merge line
    ax.plot([28.8, 28.8+3*9.6+3.2], [24.2, 24.2], color="#f08c00", lw=1.4, zorder=2)
    arrow(ax, 28.8+3*9.6+3.2, 24.2, 28.8+3*9.6+3.2, 16.5, color="#f08c00", lw=1.4)
    arrow(ax, gx, 15, gx+1.5, 15)
    box(ax, gx+1.5, 11.5, 8.5, 7, "Cropping1D\ntrim CL/2\n→ 5,000", "#f1f3f5", fs=8.0)
    arrow(ax, gx+10, 15, gx+11.5, 15)
    box(ax, gx+11.5, 10.5, 9.5, 9, "1×1 conv\n+ softmax\n3 ch\n(·/A/D)", "#ffe3e3",
        ec="#e03131", fs=8.0)
    ax.text(50, 5.6, "context-length identity:  CL = 2·Σ AR$_i$(W$_i$−1)  =  flanking size  "
            "∈ {80, 400, 2000, 10000}", ha="center", fontsize=9.2, style="italic",
            color="#343a40")

    # ---- (B) the pipeline ----
    ax = axB; ax.set_xlim(0, 100); ax.set_ylim(0, 16); ax.axis("off")
    _panel_label(ax, "B", x=0.0, y=1.04)
    ax.text(50, 15.2, "The six-subcommand pipeline", ha="center", fontsize=12.5, fontweight="bold")
    stages = [("create-data", "#e7f5ff", ACCENT),
              ("train / transfer", "#ebfbee", OSAI),
              ("calibrate", "#fff9db", "#f08c00"),
              ("predict", "#f3f0ff", "#7048e8"),
              ("variant", "#ffe3e3", "#e03131")]
    w = 16.5; gap = 2.6; x = 4
    ys = 5.5; h = 5.5
    for i, (name, fc, ec) in enumerate(stages):
        box(ax, x, ys, w, h, name, fc, ec=ec, fs=10.5, fw="bold")
        if i < len(stages) - 1:
            arrow(ax, x + w, ys + h/2, x + w + gap, ys + h/2)
        x += w + gap
    ax.text(4, 2.4, "model checkpoints (state_dict, 5 random seeds / setting; ensembled by directory)  •  "
            "create→train flow shares one SpliceAI class keyed on flanking size",
            ha="left", fontsize=8.6, color="#495057")
    fig.tight_layout()
    save(fig, "rfig_architecture.png")


# ============================================================================= FIG 5: campaign design
def fig_campaign_design():
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    ax.text(50, 38.5, "Genome-wide SNV re-scoring: how the campaign caps the shared GPU account",
            ha="center", fontsize=12.5, fontweight="bold")
    # chunk grid
    box(ax, 1, 20, 17, 12, "100,000 chunks\n× ~34,334 SNVs\n= ~3.43 B SNVs / seed", "#e7f5ff",
        ec=ACCENT, fs=8.6)
    arrow(ax, 18, 26, 22, 26)
    # PER_TASK
    box(ax, 22, 20, 19, 12, "array task\nPER_TASK = 20 chunks\nrun SEQUENTIALLY\non one GPU (-b 128)",
        "#ebfbee", ec=OSAI, fs=8.6)
    arrow(ax, 41, 26, 45, 26)
    # serialized seeds
    box(ax, 45, 27, 24, 6.5, "seed rs10  —  array %4", "#fff9db", ec="#f08c00", fs=9.2, fw="bold")
    box(ax, 45, 18.5, 24, 6.5, "seed rs13  —  array %4", "#fff4e6", ec="#f08c00", fs=9.2)
    ax.add_patch(FancyArrowPatch((57, 27), (57, 25.1), arrowstyle="-|>", mutation_scale=9,
                                 color="#e8590c", lw=1.6, zorder=2))
    ax.text(55.4, 26.0, "afterany\ndependency", fontsize=7.4, color="#e8590c",
            ha="right", va="center", linespacing=0.95)
    arrow(ax, 69, 22.8, 78, 22.8)
    # total cap
    box(ax, 78, 19.5, 20, 9, "total concurrent\nGPUs ≡ CONC = 4\n(≥6 left free)", "#d3f9d8",
        ec=OSAI_D, fs=9.0, fw="bold")
    # constraints strip
    ax.text(50, 11.5, "Cluster caps that drove the design", ha="center", fontsize=10.5, fontweight="bold")
    caps = ["qos_gpu = 10 GPUs / account (shared)",
            "MaxJobCount ≈ 100,000 tasks  →  PER_TASK=20 (5,000 tasks/seed)",
            "MaxArraySize = 15,000",
            "idempotent, record-count-verified chunks  →  lossless pause/resume"]
    for i, c in enumerate(caps):
        yy = 7.5 - i*2.3
        ax.add_patch(Rectangle((9.5, yy-0.5), 1.0, 1.0, color=OSAI, zorder=3))
        ax.text(11.5, yy, c, fontsize=8.8, va="center", ha="left", color="#343a40")
    fig.tight_layout()
    save(fig, "rfig_campaign_design.png")


# ============================================================================= FIG 6: campaign progress
def fig_campaign_progress():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), gridspec_kw={"width_ratios": [1.45, 1]})
    # (A) progress bars
    ax = axes[0]
    seeds = ["rs10", "rs13"]
    done = [RS10_DONE, RS13_DONE]
    y = np.arange(len(seeds))[::-1]
    for yi, d in zip(y, done):
        ax.barh(yi, NCHUNK, color="#e9ecef", height=0.46, zorder=2)
        ax.barh(yi, d, color=OSAI, height=0.46, zorder=3)
        ax.text(NCHUNK*0.55, yi, f"{d:,} / {NCHUNK:,}   ({100*d/NCHUNK:.1f}%)",
                ha="center", va="center", fontsize=10, color="#343a40")
    ax.set_ylim(-0.62, 1.62)
    ax.set_yticks(y); ax.set_yticklabels(seeds, fontsize=11, fontweight="bold")
    ax.set_xlim(0, NCHUNK*1.02); ax.set_xlabel("chunks scored")
    ax.set_title("Campaign progress (live)")
    ax.set_xticks([0, 25000, 50000, 75000, 100000])
    ax.set_xticklabels(["0", "25k", "50k", "75k", "100k"])
    _clean(ax, grid="x")
    # (B) throughput / ETA summary
    ax = axes[1]; ax.axis("off")
    ax.set_title("Throughput & projection", loc="center")
    eta = lambda d: (NCHUNK - d)*SEC_PER_CHUNK/CONC/86400.0
    lines = [
        ("scorer", f"{SEC_PER_CHUNK:.0f} s / chunk (A100, -b 128)"),
        ("GPU cap", f"CONC = {CONC} (serialized seeds)"),
        ("per seed", "~3,400 GPU-h  →  ~34 d ideal"),
        ("rs10 ETA", f"~{eta(RS10_DONE):.0f} d remaining"),
        ("rs13 ETA", f"~{eta(RS13_DONE):.0f} d remaining"),
        ("both seeds", "~68 d ideal / ~85 d realistic"),
        ("output", "OpenSpliceAI= beside SpliceAI= at every SNV"),
    ]
    yy = 0.92
    for k, v in lines:
        ax.text(0.02, yy, k, fontsize=9.0, fontweight="bold", color=OSAI_D, transform=ax.transAxes)
        ax.text(0.30, yy, v, fontsize=9.0, color="#343a40", transform=ax.transAxes)
        yy -= 0.135
    _panel_label(axes[0], "A");
    fig.suptitle("Genome-wide SNV scoring — current status", fontsize=12.5,
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "rfig_campaign_progress.png")


if __name__ == "__main__":
    fig_architecture()
    fig_accuracy()
    fig_speed()
    fig_variant()
    fig_campaign_design()
    fig_campaign_progress()
    print("\nAll figures written to:", ASSETS)
