"""Shared visual system for the study figures.

The palette is the validated reference instance: categorical slots 1-3 (blue,
orange, aqua), a single-hue blue sequential ramp, and a blue<->red diverging pair
with a neutral gray midpoint. The three categorical slots clear every all-pairs
gate in both light and dark modes; aqua sits below 3:1 on the light surface, so
the relief rule applies and every figure that uses it ships direct labels and a
companion table in the report.

Figures are rendered on the light surface. The HTML report and the artifact place
them on a light figure card in dark mode rather than inverting them, because an
automatic flip is not a selected dark palette.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
GRID = "#e4e3de"

# The documented slot ORDER is the colour-blind-safety mechanism, not decoration:
# only orderings clearing every adjacent gate were kept upstream. Series are assigned
# CATEGORICAL[:n] in order and never cycled or cherry-picked out of sequence.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948")
# Marker shapes give every multi-series line chart a second, non-colour channel.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SPLICEAI = CATEGORICAL[0]      # slot 1, blue
OPENSPLICEAI = CATEGORICAL[1]  # slot 2, orange
SEED_ALT = CATEGORICAL[2]      # slot 3, aqua
NEUTRAL_FILL = "#4a3aa7"       # single-series fills, where no identity is encoded
SERIES = (SPLICEAI, OPENSPLICEAI, SEED_ALT)

# Single-hue sequential ramp (blue 100 -> 700), for magnitude/density only.
SEQUENTIAL_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                    "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("study_sequential", SEQUENTIAL_STEPS)
# Diverging: blue <-> red through a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "study_diverging", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"]
)

EVENT_TITLES = {
    "AG": "Acceptor gain",
    "AL": "Acceptor loss",
    "DG": "Donor gain",
    "DL": "Donor loss",
    "MAX": "Maximum delta score",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "figure.constrained_layout.use": True,
    })


def clean(ax, grid: str = "y") -> None:
    """Recessive axes: no box, one grid direction."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid == "both":
        ax.grid(axis="both", color=GRID, linewidth=0.7)
    else:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
        ax.grid(axis="x" if grid == "y" else "y", visible=False)


def panel_label(ax, text: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="left", va="bottom")
