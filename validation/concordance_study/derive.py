"""Derived analyses computed from reduced aggregates.

Nothing here re-reads a VCF. Every quantity is a function of the additive
counters, histograms and strata that the reducer already emitted, so the
derived numbers inherit the reducer's provenance and are exactly reproducible
from the summary JSONs alone.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .loading import EVENTS, SCORE_LABELS, Run, _key

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 0.8)


def _row_identity(run: Run) -> dict:
    """Labels carried by every comparison-agnostic exported row."""
    return {"left_label": run.left_label, "right_label": run.right_label}


def _mean_aliases(run: Run, left: float, right: float) -> dict:
    """Keep historical tool aliases only where they describe the actual tools."""
    if not run.is_concordance:
        return {}
    return {"mean_spliceai": left, "mean_openspliceai": right}


# --------------------------------------------------------------------------
# Continuous agreement
# --------------------------------------------------------------------------
def agreement_table(run: Run) -> List[dict]:
    """Per-event continuous agreement, exactly as the reducer computed it."""
    rows = []
    for label in SCORE_LABELS:
        s = run.scores(label)
        rows.append(
            {
                "label": label,
                **_row_identity(run),
                "n": s["n"],
                "mean_left": s["mean_left"],
                "mean_right": s["mean_right"],
                **_mean_aliases(run, s["mean_left"], s["mean_right"]),
                "bias": s["bias_right_minus_left"],
                "mae": s["mae"],
                "rmse": s["rmse"],
                "pearson_r": s["pearson_r"],
                "lin_ccc": s["lin_ccc"],
                "spearman_r_binned": s["distribution"]["spearman_r_binned"],
                "exact_match_rate": s["exact_match_rate"],
                "left_zero_rate": s["left_zero_rate"],
                "right_zero_rate": s["right_zero_rate"],
                "equivalence_0.01": s["practical_equivalence"][_key(0.01)],
                "equivalence_0.05": s["practical_equivalence"][_key(0.05)],
                "equivalence_0.10": s["practical_equivalence"][_key(0.10)],
                "quantization_adjusted_mae": s["mean_quantization_adjusted_abs_diff"],
                "ks_binned": s["distribution"]["ks_distance_binned"],
                "wasserstein_1_binned": s["distribution"]["wasserstein_1_binned"],
                "js_divergence_base2": s["distribution"]["jensen_shannon_divergence_base2"],
            }
        )
    return rows


def threshold_table(run: Run, labels: Sequence[str] = SCORE_LABELS,
                    thresholds: Optional[Sequence[float]] = None) -> List[dict]:
    thresholds = run.raw["config"]["thresholds"] if thresholds is None else thresholds
    rows = []
    for label in labels:
        for threshold in thresholds:
            t = run.thresholds(label, threshold)
            rows.append({"label": label, **_row_identity(run),
                         "threshold": threshold, **t})
    return rows


def signal_subset_table(run: Run, labels: Sequence[str] = SCORE_LABELS) -> List[dict]:
    rows = []
    for label in labels:
        for subset, stats in sorted(run.metrics["signal_subsets"][label].items()):
            rows.append(
                {
                    "label": label,
                    **_row_identity(run),
                    "subset": subset,
                    "n": stats["n"],
                    "bias": stats["bias_right_minus_left"],
                    "mae": stats["mae"],
                    "rmse": stats["rmse"],
                    "pearson_r": stats["pearson_r"],
                    "lin_ccc": stats["lin_ccc"],
                    "exact_match_rate": stats["exact_match_rate"],
                }
            )
    return rows


def dp_table(run: Run, thresholds: Optional[Sequence[float]] = None) -> List[dict]:
    thresholds = run.raw["config"]["thresholds"] if thresholds is None else thresholds
    rows = []
    for event in EVENTS:
        for threshold in thresholds:
            entry = run.metrics["dp"][event][_key(threshold)]
            row = {
                "event": event,
                **_row_identity(run),
                "threshold": threshold,
                "both_above": entry["both_above"],
                "eligible": entry["eligible"],
            }
            for tolerance, rate in sorted(entry["within_rates"].items(), key=lambda kv: int(kv[0])):
                row[f"within_{tolerance}bp"] = rate
                row[f"within_{tolerance}bp_n"] = entry["within"][tolerance]
            rows.append(row)
    return rows


def dominant_table(run: Run) -> Tuple[List[dict], dict]:
    matrix = run.metrics["dominant"]
    labels = list(matrix)
    rows = []
    for left in labels:
        row = {**_row_identity(run), "left_dominant": left}
        row.update({f"right_{right}": matrix[left][right] for right in matrix[left]})
        if run.is_concordance:
            row["spliceai_dominant"] = left
            row.update({f"osai_{right}": matrix[left][right] for right in matrix[left]})
        rows.append(row)
    summary = {
        "overall_exact_rate": run.metrics["dominant_normalized"]["overall_exact_rate"],
        "union_signal_n": run.metrics["dominant_normalized"]["union_signal_n"],
        "union_signal_exact_rate": run.metrics["dominant_normalized"]["union_signal_exact_rate"],
        "n": run.metrics["dominant_normalized"]["n"],
    }
    return rows, summary


# --------------------------------------------------------------------------
# Operating-point transfer
# --------------------------------------------------------------------------
def _joint_matrix(run: Run, label: str) -> np.ndarray:
    bins = run.score_bins
    return np.asarray(run.joint_hist(label), dtype=np.float64).reshape(bins, bins)


def _defined(numerator: float, denominator: float) -> Optional[float]:
    """``None`` where a ratio is undefined.

    The reducer represents an undefined statistic as JSON ``null`` rather than
    NaN; these derived tables must use the same representation so the two can be
    compared field-by-field and serialised to strict JSON.
    """
    return (numerator / denominator) if denominator else None


def _call_metrics(tp: float, fp: float, fn: float, tn: float) -> dict:
    n = tp + fp + fn + tn
    predicted_pos = tp + fp
    actual_pos = tp + fn
    agreement = _defined(tp + tn, n)
    kappa: Optional[float] = None
    if n:
        pe = ((actual_pos * predicted_pos) + ((n - actual_pos) * (n - predicted_pos))) / (n * n)
        kappa = _defined((agreement - pe), (1.0 - pe))
    denominator = float(np.sqrt(max(0.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    return {
        "both_positive": tp,
        "left_only": fn,
        "right_only": fp,
        "both_negative": tn,
        "n": n,
        "overall_agreement": agreement,
        "jaccard": _defined(tp, tp + fp + fn),
        "dice": _defined(2 * tp, 2 * tp + fp + fn),
        "kappa": kappa,
        "mcc": _defined((tp * tn) - (fp * fn), denominator),
        "left_positive_rate": _defined(actual_pos, n),
        "right_positive_rate": _defined(predicted_pos, n),
        "call_rate_ratio_right_over_left": _defined(predicted_pos, actual_pos),
    }


def operating_point_transfer(run: Run, label: str = "MAX",
                             left_thresholds: Optional[Sequence[float]] = None) -> List[dict]:
    """Which OpenSpliceAI cutoff reproduces a given SpliceAI cutoff?

    Two different questions, answered separately because they have different
    answers and different uses:

    * ``rate_matched`` -- the cutoff at which OpenSpliceAI flags *as many*
      variants as SpliceAI does. This preserves review workload.
    * ``agreement_optimal`` -- the cutoff that maximises Matthews correlation
      against SpliceAI's own call set. This preserves *which* variants are
      flagged.

    Both are read off the additive joint histogram at the run's bin resolution.
    The upper endpoint cannot distinguish scores exactly one from its final bin,
    so candidate cutoffs stop at the last bin's lower edge.
    """
    bins = run.score_bins
    left_thresholds = run.raw["config"]["thresholds"] if left_thresholds is None else left_thresholds
    joint = _joint_matrix(run, label)
    total = joint.sum()
    # right_ge[b] = number of pairs whose OpenSpliceAI bin is >= b, per left bin
    right_suffix = np.cumsum(joint[:, ::-1], axis=1)[:, ::-1]  # (left_bin, b)
    right_positive_by_cut = right_suffix.sum(axis=0)           # (b,)

    rows = []
    for threshold in left_thresholds:
        if not 0 <= threshold < 1 or not np.isclose(threshold*bins, round(threshold*bins)):
            raise ValueError("operating-point cutoffs must be histogram edges below one")
        cut_left = int(round(threshold * bins))
        left_positive = joint[cut_left:, :].sum()
        # counts as a function of the right cutoff b
        tp = right_suffix[cut_left:, :].sum(axis=0)
        fp = right_positive_by_cut - tp
        fn = left_positive - tp
        tn = total - tp - fp - fn

        candidates = []
        for b in range(bins):
            stats = _call_metrics(tp[b], fp[b], fn[b], tn[b])
            stats["right_cutoff"] = b / bins
            candidates.append(stats)

        identical = candidates[cut_left]
        target_rate = identical["left_positive_rate"] or 0.0
        rate_matched = min(candidates,
                           key=lambda c: (abs((c["right_positive_rate"] or 0.0) - target_rate),
                                          c["right_cutoff"]))
        finite = [c for c in candidates if c["mcc"] is not None]
        agreement_optimal = max(finite, key=lambda c: (c["mcc"], -c["right_cutoff"])) if finite else identical

        rows.append(
            {
                "label": label,
                "spliceai_threshold": threshold,
                "spliceai_positive_calls": left_positive,
                "spliceai_positive_rate": identical["left_positive_rate"],
                "identical_cutoff": identical,
                "rate_matched": rate_matched,
                "agreement_optimal": agreement_optimal,
                "n": total,
            }
        )
    return rows


def transfer_curve(run: Run, threshold: float, label: str = "MAX") -> dict:
    """MCC/Jaccard as a function of the OpenSpliceAI cutoff, at one SpliceAI cutoff."""
    bins = run.score_bins
    joint = _joint_matrix(run, label)
    total = joint.sum()
    right_suffix = np.cumsum(joint[:, ::-1], axis=1)[:, ::-1]
    right_positive_by_cut = right_suffix.sum(axis=0)
    cut_left = int(round(threshold * bins))
    left_positive = joint[cut_left:, :].sum()
    tp = right_suffix[cut_left:, :].sum(axis=0)
    fp = right_positive_by_cut - tp
    fn = left_positive - tp
    tn = total - tp - fp - fn
    cutoffs, mcc, jaccard = [], [], []
    for b in range(bins):
        stats = _call_metrics(tp[b], fp[b], fn[b], tn[b])
        cutoffs.append(b / bins)
        mcc.append(stats["mcc"])
        jaccard.append(stats["jaccard"])
    return {"cutoffs": cutoffs, "mcc": mcc, "jaccard": jaccard, "spliceai_threshold": threshold}


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------
def quantization_profile(run: Run) -> List[dict]:
    """How much apparent disagreement is SpliceAI's two-decimal output grid?"""
    rows = []
    impact = run.metrics["right_rounded_2dp"]["impact_vs_raw"]
    for label in SCORE_LABELS:
        entry = impact[label]
        scores = run.scores(label)
        mae = scores["mae"]
        adjusted = scores["mean_quantization_adjusted_abs_diff"]
        rows.append(
            {
                "label": label,
                **_row_identity(run),
                "raw_mae": entry["raw_mae"],
                "rounded_mae": entry["right_rounded_2dp_mae"],
                "mae_change": entry["mae_change_rounded_minus_raw"],
                "raw_rmse": entry["raw_rmse"],
                "rounded_rmse": entry["right_rounded_2dp_rmse"],
                "raw_exact_match_rate": entry["raw_exact_match_rate"],
                "rounded_exact_match_rate": entry["right_rounded_2dp_exact_match_rate"],
                "exact_match_gain": entry["exact_match_rate_gain"],
                "quantization_adjusted_mae": adjusted,
                # Share of the mean absolute difference that survives discarding
                # every discrepancy attributable to the +/-0.005 rounding band.
                "share_of_mae_beyond_quantization": (adjusted / mae) if mae else float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------
def stratum_rows(run: Run, dimension: str, minimum_n: int = 0) -> List[dict]:
    rows = []
    for key, entry in run.metrics["strata"][dimension].items():
        stats = entry["metrics"]
        if stats["n"] < minimum_n:
            continue
        rows.append(
            {
                "stratum": key,
                **_row_identity(run),
                "n": stats["n"],
                "mean_left": stats["mean_left"],
                "mean_right": stats["mean_right"],
                **_mean_aliases(run, stats["mean_left"], stats["mean_right"]),
                "bias": stats["bias_right_minus_left"],
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "pearson_r": stats["pearson_r"],
                "lin_ccc": stats["lin_ccc"],
                "exact_match_rate": stats["exact_match_rate"],
            }
        )
    rows.sort(key=lambda r: -r["n"])
    return rows


def event_stratum_rows(run: Run, dimension: str) -> List[dict]:
    """One row per event and stratum, retaining threshold maps as JSON-ready data."""
    rows = []
    groups = run.metrics.get("event_strata", {}).get(dimension, {})
    for stratum, events in groups.items():
        for event, entry in events.items():
            rows.append({
                "dimension": dimension,
                "stratum": stratum,
                "event": event,
                **_row_identity(run),
                **entry["metrics"],
                "thresholds": entry["thresholds"],
            })
    rows.sort(key=lambda row: (row["stratum"], row["event"]))
    return rows


def stratum_dispersion(rows: Sequence[Mapping], field: str = "mae",
                       weight: str = "n") -> dict:
    """Spread of a per-stratum statistic, weighted by stratum size."""
    if not rows:
        return {}
    values = np.array([r[field] for r in rows], dtype=float)
    weights = np.array([r[weight] for r in rows], dtype=float)
    keep = np.isfinite(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return {}
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()

    def q(p: float) -> float:
        return float(values[np.searchsorted(cumulative, p, side="left").clip(0, values.size - 1)])

    return {
        "strata": int(values.size),
        "observations": float(weights.sum()),
        "weighted_mean": float(np.average(values, weights=weights)),
        "p05": q(0.05),
        "median": q(0.50),
        "p95": q(0.95),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------
# Seed versus model, and generalization
# --------------------------------------------------------------------------
_COMPARISON_FIELDS = (
    ("n", "paired annotations"),
    ("bias", "bias (right - left)"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("pearson_r", "Pearson r"),
    ("lin_ccc", "Lin CCC"),
    ("spearman_r_binned", "Spearman (binned)"),
    ("exact_match_rate", "exact-match rate"),
)


def _agreement_row(run: Run, label: str) -> dict:
    return {name: value for name, value in
            zip([f[0] for f in _COMPARISON_FIELDS],
                [run.scores(label)["n"],
                 run.scores(label)["bias_right_minus_left"],
                 run.scores(label)["mae"],
                 run.scores(label)["rmse"],
                 run.scores(label)["pearson_r"],
                 run.scores(label)["lin_ccc"],
                 run.scores(label)["distribution"]["spearman_r_binned"],
                 run.scores(label)["exact_match_rate"]])}


def seed_versus_model(seed_run: Run, model_runs: Sequence[Run], label: str = "MAX",
                      thresholds: Optional[Sequence[float]] = None) -> dict:
    """Contrast training-seed disagreement with SpliceAI-vs-OpenSpliceAI disagreement.

    Only legitimate because all arms are reduced over an identical chunk domain;
    the caller is responsible for having built them that way, and this function
    refuses to proceed if the paired counts imply otherwise.
    """
    thresholds = seed_run.raw["config"]["thresholds"] if thresholds is None else thresholds
    all_runs = [seed_run, *model_runs]
    if any(set(r.summary.get("chunk_ids", [])) != set(seed_run.summary.get("chunk_ids", []))
           for r in model_runs):
        raise ValueError("seed-versus-model comparisons require identical chunk IDs")
    domains = [r.summary.get("matched_domain") for r in all_runs]
    if not all(domains):
        raise ValueError("seed-versus-model comparisons require matched-domain metadata")
    if any(domain != domains[0] for domain in domains[1:]):
        raise ValueError("seed-versus-model comparisons require the identical three-way domain")
    if len({r.paired_n() for r in all_runs}) != 1:
        raise ValueError("three-way paired counts differ")
    seed = _agreement_row(seed_run, label)
    models = {run.arm: _agreement_row(run, label) for run in model_runs}

    ratios = {}
    for arm, row in models.items():
        ratios[arm] = {
            field: _defined(row[field], seed[field]) for field in ("mae", "rmse")
        }

    seed_thresholds = {
        _key(t): seed_run.thresholds(label, t) for t in thresholds
    }
    model_thresholds = {
        run.arm: {_key(t): run.thresholds(label, t) for t in thresholds}
        for run in model_runs
    }
    # Per-event decomposition: the interesting question is not whether the method
    # gap exceeds the seed gap overall, but whether it does so uniformly across
    # the four events. It does not.
    events = {}
    for event in SCORE_LABELS:
        seed_event = _agreement_row(seed_run, event)
        entry = {"seed": seed_event, "models": {}, "ratios": {}}
        for run in model_runs:
            model_event = _agreement_row(run, event)
            entry["models"][run.arm] = model_event
            entry["ratios"][run.arm] = {
                field: _defined(model_event[field], seed_event[field])
                for field in ("mae", "rmse")
            }
        events[event] = entry

    return {
        "label": label,
        "events": events,
        "seed": {"arm": seed_run.arm, "comparison": seed_run.comparison, **seed},
        "models": {arm: {"comparison": next(r.comparison for r in model_runs if r.arm == arm), **row}
                   for arm, row in models.items()},
        "error_ratios_model_over_seed": ratios,
        "seed_thresholds": seed_thresholds,
        "model_thresholds": model_thresholds,
        "chunk_domain": {
            "seed_chunks": seed_run.chunk_count,
            **{run.arm: run.chunk_count for run in model_runs},
        },
    }


def generalization(genomewide: Run, matched: Run, label: str = "MAX") -> dict:
    """Does the restricted-region arm reproduce the genome-wide arm?"""
    wide = _agreement_row(genomewide, label)
    part = _agreement_row(matched, label)
    return {
        "label": label,
        "genomewide": {"arm": genomewide.arm, "chunks": genomewide.chunk_count, **wide},
        "matched": {"arm": matched.arm, "chunks": matched.chunk_count, **part},
        "absolute_differences": {
            field: part[field] - wide[field]
            for field, _ in _COMPARISON_FIELDS
            if field != "n"
        },
    }


COMPARISON_FIELD_NAMES = dict(_COMPARISON_FIELDS)


# --------------------------------------------------------------------------
# Splice-site distance
# --------------------------------------------------------------------------
#: Bin order as emitted by `full_snv_concordance.sites`, nearest first.
SITE_DISTANCE_ORDER: Tuple[str, ...] = (
    "at_site", "1-2", "3-10", "11-50", "51-500", ">500", "no_site",
)
SITE_TYPES: Tuple[str, ...] = ("acceptor", "donor", "ambiguous")


def _split_site_key(key: str) -> Tuple[str, str]:
    distance, _, site_type = key.partition(":")
    return distance, site_type or "unknown"


def site_distance_table(run: Run, threshold: float = 0.5,
                        site_type: Optional[str] = None) -> List[dict]:
    """Agreement and call rates as a function of distance to an annotated site.

    ``site_type`` selects acceptor- or donor-nearest variants; ``None`` pools them.
    Pooling is the default because that is what the scorer's mask keys on -- it
    takes the nearest annotated boundary of either type.
    """
    strata = run.metrics["strata"].get("site_distance", {})
    buckets: Dict[str, dict] = {}
    for key, entry in strata.items():
        distance, kind = _split_site_key(key)
        if site_type is not None and kind != site_type:
            continue
        stats = entry["metrics"]
        table = entry["thresholds"].get(_key(threshold), {})
        bucket = buckets.setdefault(distance, {
            "distance": distance, "n": 0, "sum_left": 0.0, "sum_right": 0.0,
            "sum_abs": 0.0, "left_calls": 0, "right_calls": 0, "both_calls": 0,
        })
        n = stats["n"]
        bucket["n"] += n
        bucket["sum_left"] += stats["mean_left"] * n
        bucket["sum_right"] += stats["mean_right"] * n
        bucket["sum_abs"] += stats["mae"] * n
        bucket["both_calls"] += table.get("both_positive", 0)
        bucket["left_calls"] += table.get("both_positive", 0) + table.get("left_only", 0)
        bucket["right_calls"] += table.get("both_positive", 0) + table.get("right_only", 0)

    rows = []
    for distance in SITE_DISTANCE_ORDER:
        bucket = buckets.get(distance)
        if not bucket or bucket["n"] == 0:
            continue
        n = bucket["n"]
        rows.append({
            "distance": distance,
            "site_type": site_type or "either",
            **_row_identity(run),
            "n": n,
            "mean_left": bucket["sum_left"] / n,
            "mean_right": bucket["sum_right"] / n,
            **_mean_aliases(run, bucket["sum_left"] / n, bucket["sum_right"] / n),
            "mae": bucket["sum_abs"] / n,
            "left_calls": bucket["left_calls"],
            "right_calls": bucket["right_calls"],
            "shared_calls": bucket["both_calls"],
            "left_call_rate": bucket["left_calls"] / n,
            "right_call_rate": bucket["right_calls"] / n,
            "call_rate_ratio": _defined(bucket["right_calls"], bucket["left_calls"]),
            "jaccard": _defined(
                bucket["both_calls"],
                bucket["left_calls"] + bucket["right_calls"] - bucket["both_calls"],
            ),
        } | ({
            "spliceai_calls": bucket["left_calls"],
            "openspliceai_calls": bucket["right_calls"],
            "spliceai_call_rate": bucket["left_calls"] / n,
            "openspliceai_call_rate": bucket["right_calls"] / n,
        } if run.is_concordance else {}))
    return rows


# --------------------------------------------------------------------------
# Continuous agreement curves, read off the joint histogram
# --------------------------------------------------------------------------
def same_cutoff_curve(run: Run, label: str = "MAX") -> dict:
    """Agreement when *both* predictors use the same cutoff, over the whole grid.

    The configured thresholds give four or five points; the joint histogram gives
    one point per bin, which is what turns a threshold plot into a threshold curve.
    """
    bins = run.score_bins
    joint = _joint_matrix(run, label)
    total = float(joint.sum())
    # suffix sums in both directions: reversed cumsum on each axis
    suffix2d = np.cumsum(np.cumsum(joint[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    left_positive = joint[::-1].cumsum(axis=0)[::-1].sum(axis=1)
    right_positive = joint[:, ::-1].cumsum(axis=1)[:, ::-1].sum(axis=0)

    cutoffs, rows = [], []
    for b in range(1, bins):
        tp = float(suffix2d[b, b])
        fn = float(left_positive[b]) - tp
        fp = float(right_positive[b]) - tp
        tn = total - tp - fn - fp
        stats = _call_metrics(tp, fp, fn, tn)
        stats["cutoff"] = b / bins
        cutoffs.append(b / bins)
        rows.append(stats)
    return {"label": label, "cutoffs": cutoffs, "points": rows}


def tail_asymmetry(run: Run, label: str = "MAX") -> dict:
    """``P(delta <= -x)`` against ``P(delta >= +x)`` -- the gain skew as a curve.

    The signed-difference histogram is symmetric-looking on a log count axis even
    when the two tails differ by an order of magnitude; comparing the tails
    directly states the asymmetry instead of hiding it.
    """
    bins = run.score_bins
    counts = np.asarray(run.raw["diff_hist"][label], dtype=float)
    total = counts.sum()
    centre = bins  # diff_hist spans [-1, 1] in 2*bins + 1 slots
    magnitudes, negative, positive = [], [], []
    for step in range(1, bins + 1):
        magnitudes.append(step / bins)
        negative.append(float(counts[: centre - step + 1].sum()) / total if total else None)
        positive.append(float(counts[centre + step:].sum()) / total if total else None)
    ratios = [
        (n / p) if (p not in (None, 0.0) and n is not None) else None
        for n, p in zip(negative, positive)
    ]
    return {
        "label": label,
        "approximate": True,
        "bin_width": 1 / bins,
        "difference_rounding": "nearest histogram centre",
        "magnitudes": magnitudes,
        "p_left_tail": negative,
        "p_right_tail": positive,
        "left_over_right": ratios,
    }


# --------------------------------------------------------------------------
# Record-level discordance
# --------------------------------------------------------------------------
def discordance_taxonomy(run: Run, top: int = 25) -> dict:
    """What the largest disagreements actually are, from the top-discrepancy records."""
    records = run.raw.get("top_discrepancies", [])
    by_event: Dict[str, dict] = {
        event: {"event": event, "n": 0, "spliceai_higher": 0, "openspliceai_higher": 0,
                "sum_difference": 0.0}
        for event in SCORE_LABELS[:-1]
    }
    genes: Dict[str, int] = {}
    for record in records:
        left = record.get("left_scores") or []
        right = record.get("right_scores") or []
        if len(left) != 4 or len(right) != 4:
            continue
        deltas = [r - lft for lft, r in zip(left, right)]
        index = max(range(4), key=lambda i: abs(deltas[i]))
        event = SCORE_LABELS[index]
        bucket = by_event[event]
        bucket["n"] += 1
        bucket["sum_difference"] += deltas[index]
        if deltas[index] < 0:
            bucket["spliceai_higher"] += 1
        else:
            bucket["openspliceai_higher"] += 1
        gene = record.get("gene")
        if gene:
            genes[gene] = genes.get(gene, 0) + 1
    for bucket in by_event.values():
        bucket["mean_difference"] = _defined(bucket["sum_difference"], bucket["n"])
        bucket["share_spliceai_higher"] = _defined(bucket["spliceai_higher"], bucket["n"])
    return {
        "records": len(records),
        "by_event": [by_event[e] for e in SCORE_LABELS[:-1]],
        "top_genes": sorted(
            ({"gene": g, "records": c} for g, c in genes.items()),
            key=lambda r: -r["records"],
        )[:top],
    }


# --------------------------------------------------------------------------
# Genomic landscape
# --------------------------------------------------------------------------
def genomic_landscape(run: Run, minimum_n: int = 10_000) -> List[dict]:
    """Per-1-Mb-block statistics ordered along the genome."""
    order = {f"chr{name}": i for i, name in enumerate(
        [str(v) for v in range(1, 23)] + ["X", "Y", "M"])}
    rows = []
    for key, entry in run.metrics["strata"].get("block_1mb", {}).items():
        stats = entry["metrics"]
        if stats["n"] < minimum_n:
            continue
        chrom, _, span = key.partition(":")
        start = int(span.split("-")[0]) if "-" in span else 0
        rows.append({
            "block": key, "chrom": chrom, "start": start, "n": stats["n"],
            **_row_identity(run),
            "bias": stats["bias_right_minus_left"], "mae": stats["mae"],
            "pearson_r": stats["pearson_r"],
            "mean_left": stats["mean_left"], "mean_right": stats["mean_right"],
            **_mean_aliases(run, stats["mean_left"], stats["mean_right"]),
        })
    rows.sort(key=lambda r: (order.get(r["chrom"], 99), r["start"]))
    return rows
