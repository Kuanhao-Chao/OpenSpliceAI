"""Tables derived from the additional depth aggregates, with explicit labels."""
from __future__ import annotations

from collections import Counter
from .derive import _defined, operating_point_transfer
from .loading import EVENTS, SCORE_LABELS


def site_event_table(run):
    rows = []
    for key, events in run.metrics.get("event_strata", {}).get("site_distance", {}).items():
        distance, kind = key.split(":", 1)
        entries = dict(events)
        entries["MAX"] = run.metrics["strata"]["site_distance"][key]
        for label in SCORE_LABELS:
            entry = entries[label]
            for threshold, calls in entry["thresholds"].items():
                rows.append({"distance": distance, "site_type": kind, "event": label,
                             "left_label": run.left_label, "right_label": run.right_label,
                             "threshold": float(threshold), **entry["metrics"], **calls})
    return sorted(rows, key=lambda r: (r["distance"], r["site_type"], r["event"], r["threshold"]))


def site_dp_table(run):
    rows = []
    for key, events in run.raw.get("depth", {}).get("site_dp", {}).items():
        distance, kind = key.split(":", 1)
        for event, thresholds in events.items():
            for threshold, values in thresholds.items():
                for tolerance, n in values["within"].items():
                    rows.append({"distance": distance, "site_type": kind, "event": event,
                        "left_label": run.left_label, "right_label": run.right_label,
                        "threshold": float(threshold), "tolerance_bp": int(tolerance),
                        "eligible": values["eligible"], "within": n,
                        "rate": _defined(n, values["eligible"])})
    return rows


def operating_points(run):
    return {label: {f"{r['spliceai_threshold']:g}": r
                    for r in operating_point_transfer(run, label)} for label in SCORE_LABELS}


def matched_strata(seed, models):
    domain = seed.summary.get("matched_domain")
    if not domain:
        return []  # historical pairwise summaries cannot support this estimand
    if any(r.summary.get("matched_domain") != domain for r in models):
        raise ValueError("stratum comparison requires a shared three-way domain")
    rows = []
    # Dominant-pair membership is comparison-dependent, so compare it
    # descriptively within each arm rather than treating those cells as shared.
    dimensions = ("chrom", "gene", "block_1mb", "substitution", "site_distance")
    for dimension in dimensions:
        groups = seed.metrics["event_strata"][dimension]
        for key, events in groups.items():
            for event in EVENTS:
                base = events[event]["metrics"]
                for model in models:
                    model_entry = model.metrics["event_strata"][dimension][key][event]
                    other = model_entry["metrics"]
                    if base["n"] != other["n"]:
                        raise ValueError(f"three-way stratum counts differ: {dimension}/{key}/{event}")
                    rows.append({"dimension": dimension, "stratum": key, "event": event,
                        "model_arm": model.arm, "n": base["n"], "seed_mae": base["mae"],
                        "method_mae": other["mae"], "mae_ratio": _defined(other["mae"], base["mae"]),
                        "seed_rmse": base["rmse"], "method_rmse": other["rmse"],
                        "rmse_ratio": _defined(other["rmse"], base["rmse"]),
                        "method_bias": other["bias_right_minus_left"],
                        "threshold_disagreement": {t: {
                            "seed": seed_calls["left_only"]+seed_calls["right_only"],
                            "method": model_entry["thresholds"][t]["left_only"]+model_entry["thresholds"][t]["right_only"]}
                            for t, seed_calls in events[event]["thresholds"].items()}})
    return rows


def discordance(run, sites):
    if sites is None:
        return {}
    if sites.digest != run.raw["config"]["sites_digest"]:
        raise ValueError("discordance taxonomy annotation differs from the mapper annotation")
    strata, ties, genes = Counter(), Counter(), Counter()
    records = run.raw.get("top_discrepancies", [])
    details = []
    for r in records:
        chrom, pos, ref, alt = r["variant"].rsplit(":", 3)
        distance, kind = sites.stratum_key(chrom, int(pos)).split(":", 1)
        diffs = [y-x for x, y in zip(r["left_scores"], r["right_scores"])]
        magnitude = max(abs(d) for d in diffs)
        winners = [i for i, d in enumerate(diffs) if abs(d) == magnitude]
        event = EVENTS[winners[0]] if len(winners) == 1 else "TIE"
        directions = {"left_higher" if diffs[i] < 0 else "right_higher" if diffs[i] > 0 else "equal" for i in winners}
        direction = next(iter(directions)) if len(directions) == 1 else "mixed_tie"
        strata[(event, direction, distance, kind)] += 1
        ties[len(winners)] += 1
        genes[r["gene"]] += 1
        details.append({"variant": r["variant"], "gene": r["gene"], "event": event,
                        "direction": direction, "distance": distance, "site_type": kind,
                        "max_abs_difference": magnitude})
    return {"records": len(records), "population": "selected top absolute discrepancies; not a prevalence estimate",
            "left_label": run.left_label, "right_label": run.right_label,
            "ties": dict(ties), "strata": [dict(event=e, direction=d, distance=s, site_type=k, n=n)
                 for (e, d, s, k), n in sorted(strata.items())],
            "top_genes": [{"gene": g, "n": n} for g, n in genes.most_common(25)],
            "examples": details[:25], "record_table": details}


def collapsed_table(run):
    view = run.metrics["variant_collapsed_view"]
    return [{"event": e, "left_label": run.left_label, "right_label": run.right_label,
             **view["scores"][e], "primary_n": run.scores(e)["n"],
             "primary_mae": run.scores(e)["mae"]} for e in SCORE_LABELS]


def histograms(run, distances, event):
    """Pool stored sparse joint histograms; used for distributions/calibration."""
    import numpy as np
    bins = run.score_bins
    left, right, joint = np.zeros(bins), np.zeros(bins), np.zeros((bins, bins))
    for key, events in run.raw.get("depth", {}).get("site_histograms", {}).items():
        if key.split(":", 1)[0] not in distances:
            continue
        state = events[event]
        left += state["left"]
        right += state["right"]
        for index, n in state["joint"].items():
            x, y = divmod(int(index), bins)
            joint[x, y] += n
    return left, right, joint


def stratum_summary(rows):
    import numpy as np
    result = []
    grouped = {}
    for r in rows:
        grouped.setdefault((r["dimension"], r["event"], r["model_arm"]), []).append(r)
    for (dimension, event, arm), group in sorted(grouped.items()):
        finite = [r["mae_ratio"] for r in group if r["mae_ratio"] is not None]
        result.append({"dimension": dimension, "event": event, "model_arm": arm,
                       "strata": len(group), "defined_ratios": len(finite),
                       "median_mae_ratio": float(np.median(finite)) if finite else None,
                       "fraction_method_above_seed": _defined(sum(x>1 for x in finite), len(finite))})
    return result
