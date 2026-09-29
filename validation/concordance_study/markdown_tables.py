"""Markdown table builders driven entirely by the fact base.

The narrative template refers to a table as ``{{table:name}}`` and never contains
table rows itself, so a table cannot fall out of step with the run that produced
it, and adding a chromosome or an event never means editing prose.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from .loading import EVENTS, EVENT_NAMES, SCORE_LABELS

BUILDERS: Dict[str, Callable[[Mapping], str]] = {}


def builder(name: str):
    def register(function):
        BUILDERS[name] = function
        return function
    return register


# --------------------------------------------------------------------------
def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if spec == "sig":
            return f"{value:,.4g}"
        return format(value, spec)
    if isinstance(value, int):
        return f"{value:,d}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           align: Sequence[str] | None = None) -> str:
    align = align or (["left"] + ["right"] * (len(headers) - 1))
    bar = {"left": ":---", "right": "---:", "center": ":---:"}
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(bar[a] for a in align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _seed_of(arm: str) -> str:
    """The checkpoint an arm used, recovered from its name (``C_rs10_matched`` -> ``rs10``)."""
    for token in arm.replace("-", "_").split("_"):
        if token.startswith("rs") and token[2:].isdigit():
            return token
    return arm


def _label_name(label: str) -> str:
    return f"{EVENT_NAMES[label]} ({label})" if label != "MAX" else "maximum (MAX)"


# --------------------------------------------------------------------------
@builder("runs_overview")
def _runs_overview(facts: Mapping) -> str:
    rows = []
    for arm, data in sorted(facts["arms"].items()):
        rows.append([
            f"**{arm}**",
            data["comparison"],
            data["role"],
            _fmt(data["chunks"]),
            _fmt(data["coverage"]["paired_annotations"]),
            data["finality"]["status"],
        ])
    return _table(["Arm", "Comparison", "Role", "Chunks", "Paired annotations", "Status"], rows)


@builder("coverage")
def _coverage(facts: Mapping) -> str:
    coverage = facts["primary"]["coverage"]
    rates = facts["primary"]["coverage_rates"]
    rows = [
        ["Retained VCF rows", _fmt(coverage["source_rows"]), "rows entering aggregation after incomplete edge groups are excluded"],
        ["Variant groups", _fmt(coverage["variant_groups"]), "distinct (CHROM, POS, REF, ALT)"],
        ["SpliceAI annotations", _fmt(coverage["left_valid_annotations"]), "valid, deduplicated, allele-matched"],
        ["OpenSpliceAI annotations", _fmt(coverage["right_valid_annotations"]), "valid, deduplicated, allele-matched"],
        ["**Exact-gene paired**", f"**{_fmt(coverage['paired_annotations'])}**", "**the denominator of every paired statistic**"],
        ["SpliceAI-only annotations", _fmt(coverage["left_only_annotations"]), "gene absent from the OpenSpliceAI output"],
        ["OpenSpliceAI-only annotations", _fmt(coverage["right_only_annotations"]), "gene absent from the SpliceAI source"],
        ["Groups without any prediction", _fmt(coverage["groups_without_right_prediction"]), "no OpenSpliceAI value on any row"],
        ["Conflicting SpliceAI values", _fmt(coverage["left_conflicts"]), "excluded from paired metrics"],
        ["Conflicting OpenSpliceAI values", _fmt(coverage["right_conflicts"]), "excluded from paired metrics"],
        ["Duplicate annotations", _fmt(coverage["duplicate_annotations"]), "identical repeats, collapsed"],
        ["Malformed annotations", _fmt(coverage["invalid_annotations"]), "rejected"],
        ["Edge fragments excluded", _fmt(coverage["excluded_incomplete_edge_fragments"]), "groups touching a missing chunk"],
    ]
    table = _table(["Quantity", "Count", "Meaning"], rows,
                   align=["left", "right", "left"])
    return table + (
        f"\n\nOf the SpliceAI annotations, {rates['paired_share_of_spliceai_annotations'] * 100:.2f}% "
        f"could be paired with an OpenSpliceAI annotation for the same gene."
    )


@builder("agreement")
def _agreement(facts: Mapping) -> str:
    rows = []
    for label in SCORE_LABELS:
        a = facts["primary"]["agreement"][label]
        rows.append([
            _label_name(label), _fmt(a["n"]), _fmt(a["mean_spliceai"], ".5f"),
            _fmt(a["mean_openspliceai"], ".5f"), _fmt(a["bias"], "+.5f"),
            _fmt(a["mae"], ".5f"), _fmt(a["rmse"], ".5f"),
            _fmt(a["pearson_r"], ".4f"), _fmt(a["lin_ccc"], ".4f"),
            _fmt(a["spearman_r_binned"], ".4f"),
        ])
    return _table(["Event", "n", "Mean SpliceAI", "Mean OpenSpliceAI", "Bias",
                   "MAE", "RMSE", "Pearson r", "Lin CCC", "Spearman*"], rows)


@builder("equivalence")
def _equivalence(facts: Mapping) -> str:
    rows = []
    for label in SCORE_LABELS:
        a = facts["primary"]["agreement"][label]
        rows.append([
            _label_name(label),
            f"{a['left_zero_rate'] * 100:.2f}%", f"{a['right_zero_rate'] * 100:.2f}%",
            f"{a['exact_match_rate'] * 100:.2f}%",
            f"{a['equivalence_0.01'] * 100:.2f}%", f"{a['equivalence_0.05'] * 100:.2f}%",
            f"{a['equivalence_0.10'] * 100:.2f}%",
        ])
    return _table(["Event", "SpliceAI = 0", "OpenSpliceAI = 0", "Exact match",
                   "Absolute difference ≤ 0.01", "Absolute difference ≤ 0.05", "Absolute difference ≤ 0.10"], rows)


@builder("thresholds_max")
def _thresholds_max(facts: Mapping) -> str:
    rows = []
    for key, entry in sorted(facts["primary"]["thresholds"]["MAX"].items(), key=lambda kv: float(kv[0])):
        rows.append([
            key, _fmt(entry["both_positive"]), _fmt(entry["left_only"]),
            _fmt(entry["right_only"]), _fmt(entry["both_negative"]),
            f"{entry['overall_agreement'] * 100:.3f}%",
            _fmt(entry["jaccard"], ".4f"), _fmt(entry["kappa"], ".4f"),
            _fmt(entry["mcc"], ".4f"), _fmt(entry["call_rate_ratio_right_over_left"], ".3f"),
        ])
    return _table(["Threshold", "Both called", "SpliceAI only", "OpenSpliceAI only",
                   "Neither", "Agreement", "Jaccard", "Kappa", "MCC", "Call-rate ratio"], rows)


@builder("thresholds_by_event")
def _thresholds_by_event(facts: Mapping) -> str:
    rows = []
    for label in SCORE_LABELS:
        entry = facts["primary"]["thresholds"][label]["0.5"]
        rows.append([
            _label_name(label), _fmt(entry["both_positive"]), _fmt(entry["left_only"]),
            _fmt(entry["right_only"]), _fmt(entry["jaccard"], ".4f"),
            _fmt(entry["kappa"], ".4f"), _fmt(entry["mcc"], ".4f"),
            _fmt(entry["call_rate_ratio_right_over_left"], ".3f"),
        ])
    return _table(["Event", "Both called", "SpliceAI only", "OpenSpliceAI only",
                   "Jaccard", "Kappa", "MCC", "Call-rate ratio"], rows)


@builder("operating_point")
def _operating_point(facts: Mapping) -> str:
    rows = []
    for key, entry in sorted(facts["primary"]["operating_point"].items(), key=lambda kv: float(kv[0])):
        identical, matched, optimal = entry["identical_cutoff"], entry["rate_matched"], entry["agreement_optimal"]
        rows.append([
            f"SpliceAI ≥ {key}",
            _fmt(int(entry["spliceai_positive_calls"])),
            _fmt(int(identical["both_positive"] + identical["right_only"])),
            _fmt(identical["mcc"], ".4f"),
            f"{matched['right_cutoff']:.3f}",
            _fmt(matched["mcc"], ".4f"),
            f"{optimal['right_cutoff']:.3f}",
            _fmt(optimal["mcc"], ".4f"),
        ])
    return _table(["SpliceAI operating point", "SpliceAI calls", "OpenSpliceAI calls at the same cutoff",
                   "MCC at same cutoff", "Rate-matched cutoff", "MCC", "Agreement-optimal cutoff", "MCC"], rows)


@builder("quantization")
def _quantization(facts: Mapping) -> str:
    rows = []
    for label in SCORE_LABELS:
        q = facts["primary"]["quantization"][label]
        rows.append([
            _label_name(label), _fmt(q["raw_mae"], ".6f"), _fmt(q["rounded_mae"], ".6f"),
            f"{q['raw_exact_match_rate'] * 100:.2f}%",
            f"{q['rounded_exact_match_rate'] * 100:.2f}%",
            f"{q['exact_match_gain'] * 100:+.2f} pp",
            _fmt(q["quantization_adjusted_mae"], ".6f"),
            f"{q['share_of_mae_beyond_quantization'] * 100:.1f}%",
        ])
    return _table(["Event", "MAE", "MAE (OSAI at 2 dp)", "Exact match", "Exact match (2 dp)",
                   "Change", "MAE beyond +/-0.005", "Share of MAE"], rows)


@builder("signal_subsets")
def _signal_subsets(facts: Mapping) -> str:
    rows = []
    subsets = facts["primary"]["signal_subsets"]["MAX"]
    order = sorted(subsets, key=lambda k: -1 if k == "either_gt_0" else float(k.removeprefix("either_ge_")))
    names = {key: "either score > 0" if key == "either_gt_0" else
             "either score ≥ " + key.removeprefix("either_ge_") for key in order}
    total = facts["primary"]["agreement"]["MAX"]["n"]
    rows.append(["all paired annotations", _fmt(total), "100.00%",
                 _fmt(facts["primary"]["agreement"]["MAX"]["bias"], "+.5f"),
                 _fmt(facts["primary"]["agreement"]["MAX"]["mae"], ".5f"),
                 _fmt(facts["primary"]["agreement"]["MAX"]["pearson_r"], ".4f"),
                 _fmt(facts["primary"]["agreement"]["MAX"]["lin_ccc"], ".4f")])
    for key in order:
        if key not in subsets:
            continue
        entry = subsets[key]
        rows.append([names.get(key, key), _fmt(entry["n"]),
                     f"{entry['n'] / total * 100:.3f}%",
                     _fmt(entry["bias"], "+.5f"), _fmt(entry["mae"], ".5f"),
                     _fmt(entry["pearson_r"], ".4f"), _fmt(entry["lin_ccc"], ".4f")])
    return _table(["Subset", "n", "Share", "Bias", "MAE", "Pearson r", "Lin CCC"], rows)


@builder("dp_agreement")
def _dp(facts: Mapping) -> str:
    rows = []
    for event in EVENTS:
        entry = facts["primary"]["dp"][event]["0.5"]
        rows.append([
            _label_name(event), _fmt(entry["both_above"]),
            _fmt(entry.get('within_0bp'), ".2%"),
            _fmt(entry.get('within_1bp'), ".2%"),
            _fmt(entry.get('within_2bp'), ".2%"),
            _fmt(entry.get('within_5bp'), ".2%"),
            _fmt(entry.get('within_10bp'), ".2%"),
        ])
    return _table(["Event", "Jointly called (≥ 0.5)", "Exact", "≤ 1 bp", "≤ 2 bp",
                   "≤ 5 bp", "≤ 10 bp"], rows)


@builder("dp_agreement_counts")
def _dp_counts(facts: Mapping) -> str:
    """Exact integers avoid rounding rare mismatches into apparent perfection."""
    rows = []
    for event in EVENTS:
        entry = facts["primary"]["dp"][event]["0.5"]
        eligible, exact = entry["eligible"], entry["within_0bp_n"]
        if not 0 <= exact <= eligible:
            raise ValueError("invalid exact-position counts")
        rows.append([_label_name(event), _fmt(eligible), _fmt(exact), _fmt(eligible-exact)])
    return _table(["Event", "Jointly called (≥0.5)", "Exact positions", "Mismatches"], rows)


@builder("dominant")
def _dominant(facts: Mapping) -> str:
    normalized = facts["primary"]["dominant"]["row_normalized"]
    labels = list(normalized)
    rows = []
    for left in labels:
        rows.append([left] + [f"{normalized[left][right] * 100:.1f}%" for right in labels])
    return _table(["SpliceAI \\ OpenSpliceAI"] + labels, rows)


@builder("chromosomes")
def _chromosomes(facts: Mapping) -> str:
    order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    rank = {name: i for i, name in enumerate(order)}
    entries = sorted(facts["primary"]["strata"]["chrom"]["largest"],
                     key=lambda r: rank.get(r["stratum"], 999))
    rows = [[e["stratum"], _fmt(e["n"]), _fmt(e["bias"], "+.5f"), _fmt(e["mae"], ".5f"),
             _fmt(e["pearson_r"], ".4f"), _fmt(e["lin_ccc"], ".4f")] for e in entries]
    return _table(["Chromosome", "n", "Bias", "MAE", "Pearson r", "Lin CCC"], rows)


@builder("top_genes")
def _top_genes(facts: Mapping) -> str:
    entries = facts["primary"]["strata"]["gene"]["top_by_mae"][:15]
    rows = [[e["stratum"], _fmt(e["n"]), _fmt(e["mean_spliceai"], ".4f"),
             _fmt(e["mean_openspliceai"], ".4f"), _fmt(e["bias"], "+.4f"),
             _fmt(e["mae"], ".4f"), _fmt(e["pearson_r"], ".3f")] for e in entries]
    return _table(["Gene", "n", "Mean SpliceAI", "Mean OpenSpliceAI", "Bias", "MAE", "Pearson r"], rows)


@builder("substitutions")
def _substitutions(facts: Mapping) -> str:
    entries = sorted(facts["primary"]["strata"]["substitution"]["largest"],
                     key=lambda r: r["stratum"])
    rows = [[e["stratum"], _fmt(e["n"]), _fmt(e["bias"], "+.5f"), _fmt(e["mae"], ".5f"),
             _fmt(e["pearson_r"], ".4f")] for e in entries]
    return _table(["Substitution", "n", "Bias", "MAE", "Pearson r"], rows)


@builder("stratum_dispersion")
def _stratum_dispersion(facts: Mapping) -> str:
    names = {"chrom": "Chromosome", "gene": "Gene", "block_1mb": "1-Mb block",
             "substitution": "REF>ALT substitution"}
    rows = []
    for dimension, name in names.items():
        entry = facts["primary"]["strata"].get(dimension)
        if not entry or not entry["dispersion_mae"]:
            continue
        mae, bias = entry["dispersion_mae"], entry["dispersion_bias"]
        rows.append([name, _fmt(entry["count"]),
                     _fmt(mae["p05"], ".5f"), _fmt(mae["median"], ".5f"), _fmt(mae["p95"], ".5f"),
                     _fmt(bias["p05"], "+.5f"), _fmt(bias["median"], "+.5f"), _fmt(bias["p95"], "+.5f")])
    return _table(["Stratum", "Count", "MAE p05", "MAE median", "MAE p95",
                   "Bias p05", "Bias median", "Bias p95"], rows)


@builder("bootstrap")
def _bootstrap(facts: Mapping) -> str:
    payload = facts["primary"]["bootstrap"]
    rows = []
    for dimension, entry in sorted(payload["dimensions"].items()):
        for metric, values in sorted(entry["metrics"].items()):
            rows.append([dimension, metric, _fmt(entry["cluster_count"]),
                         _fmt(values["estimate"], ".5f"),
                         f"{_fmt(values['lower'], '.5f')} to {_fmt(values['upper'], '.5f')}"])
    return _table(["Cluster", "Statistic", "Clusters", "Estimate", "95% interval"], rows)


@builder("seed_versus_model")
def _seed_versus_model(facts: Mapping) -> str:
    if "seed_versus_model" not in facts:
        return "_Seed comparison not available in this run set._"
    payload = facts["seed_versus_model"]
    rows = [["**Training seed** — OpenSpliceAI rs10 vs rs13", _fmt(payload["seed"]["n"]),
             _fmt(payload["seed"]["bias"], "+.5f"), _fmt(payload["seed"]["mae"], ".5f"),
             _fmt(payload["seed"]["rmse"], ".5f"), _fmt(payload["seed"]["pearson_r"], ".4f"),
             _fmt(payload["seed"]["lin_ccc"], ".4f"),
             f"{payload['seed']['exact_match_rate'] * 100:.2f}%"]]
    for arm, row in sorted(payload["models"].items()):
        rows.append([f"**Method** — SpliceAI vs OpenSpliceAI {_seed_of(arm)}", _fmt(row["n"]),
                     _fmt(row["bias"], "+.5f"), _fmt(row["mae"], ".5f"),
                     _fmt(row["rmse"], ".5f"), _fmt(row["pearson_r"], ".4f"),
                     _fmt(row["lin_ccc"], ".4f"),
                     f"{row['exact_match_rate'] * 100:.2f}%"])
    table = _table(["Contrast", "n", "Bias", "MAE", "RMSE", "Pearson r", "Lin CCC", "Exact match"], rows)
    ratios = "; ".join(
        f"{_seed_of(arm)} MAE x{_fmt(values['mae'], '.2f')}, RMSE x{_fmt(values['rmse'], '.2f')}"
        for arm, values in sorted(payload["error_ratios_model_over_seed"].items())
    )
    return table + f"\n\nMethod error relative to training-seed error — {ratios}."


@builder("seed_versus_model_events")
def _seed_versus_model_events(facts: Mapping) -> str:
    if "seed_versus_model" not in facts:
        return "_Seed comparison not available in this run set._"
    events = facts["seed_versus_model"]["events"]
    arms = sorted(next(iter(events.values()))["models"])
    rows = []
    for label in SCORE_LABELS:
        entry = events[label]
        row = [_label_name(label),
               _fmt(entry["seed"]["pearson_r"], ".4f"), _fmt(entry["seed"]["mae"], ".5f")]
        for arm in arms:
            row += [_fmt(entry["models"][arm]["pearson_r"], ".4f"),
                    _fmt(entry["models"][arm]["mae"], ".5f")]
        row.append(" / ".join(f"x{_fmt(entry['ratios'][arm]['mae'], '.2f')}" for arm in arms))
        rows.append(row)
    headers = ["Event", "Seed r", "Seed MAE"]
    for arm in arms:
        seed_label = "rs10" if "rs10" in arm else "rs13"
        headers += [f"Method r ({seed_label})", f"Method MAE ({seed_label})"]
    headers.append("MAE ratio method/seed")
    return _table(headers, rows)


@builder("generalization")
def _generalization(facts: Mapping) -> str:
    if "generalization" not in facts:
        return "_Restricted-region comparison not available in this run set._"
    payload = facts["generalization"]
    fields = [("bias", "Bias"), ("mae", "MAE"), ("rmse", "RMSE"),
              ("pearson_r", "Pearson r"), ("lin_ccc", "Lin CCC"),
              ("exact_match_rate", "Exact-match rate")]
    rows = []
    for key, name in fields:
        wide, part = payload["genomewide"][key], payload["matched"][key]
        rows.append([name, _fmt(wide, ".5f"), _fmt(part, ".5f"), _fmt(part - wide, "+.5f")])
    return _table(["Statistic", "Genome-wide (chr1-chrY)", "Restricted (chr1-chr7)", "Difference"], rows)


@builder("site_distance")
def _site_distance(facts: Mapping) -> str:
    if "site_distance" not in facts["primary"]:
        return "_This run did not carry the splice-site distance stratum._"
    payload = facts["primary"]["site_distance"]
    rows = []
    for distance in payload["order"]:
        row = payload["pooled"][distance]
        rows.append([
            distance, _fmt(row["n"]),
            f"{row['left_call_rate'] * 100:.4f}%",
            f"{row['right_call_rate'] * 100:.4f}%",
            _fmt(row["call_rate_ratio"], ".3f"),
            _fmt(row["jaccard"], ".4f"),
            _fmt(row["mean_left"], ".5f"),
            _fmt(row["mean_right"], ".5f"),
        ])
    return _table(["Distance to nearest site", "n", "SpliceAI calls", "OpenSpliceAI calls",
                   "Call-rate ratio", "Jaccard", "Mean SpliceAI", "Mean OpenSpliceAI"], rows)


@builder("tail_asymmetry")
def _tail_asymmetry(facts: Mapping) -> str:
    rows = []
    for label in SCORE_LABELS:
        tails = facts["primary"]["tail_asymmetry"][label]
        entries = []
        for magnitude in (0.1, 0.2, 0.5):
            index = int(round(magnitude * len(tails["magnitudes"]))) - 1
            index = max(0, min(index, len(tails["magnitudes"]) - 1))
            entries.append(tails["left_over_right"][index])
        rows.append([_label_name(label)] + [_fmt(v, ".2f") for v in entries])
    return _table(["Event", "ratio at absolute difference ≥ 0.1", "≥ 0.2", "≥ 0.5"], rows) + (
        "\n\nValues above 1 mean SpliceAI is the higher scorer more often at that magnitude; "
        "below 1 means OpenSpliceAI is."
    )


@builder("discordance")
def _discordance(facts: Mapping) -> str:
    payload = facts["primary"]["discordance"]
    rows = []
    for entry in payload["by_event"]:
        rows.append([
            _label_name(entry["event"]), _fmt(entry["n"]),
            _fmt(entry["share_spliceai_higher"], ".1%"),
            _fmt(entry["mean_difference"], "+.4f"),
        ])
    table = _table(["Event driving the discrepancy", "records", "SpliceAI higher",
                    "Mean signed difference"], rows)
    genes = ", ".join(f"{g['gene']} ({g['records']})" for g in payload["top_genes"][:10])
    return table + f"\n\nAcross {payload['records']:,} largest-discrepancy records. Most frequent genes: {genes}."


@builder("provenance")
def _provenance(facts: Mapping) -> str:
    rows = []
    for arm, data in sorted(facts["arms"].items()):
        provenance = data["provenance"]
        classes = ", ".join(f"{k}: {v:,}" for k, v in sorted(
            provenance.get("output_provenance_class_counts", {}).items()))
        rows.append([arm, data["finality"]["status"],
                     _fmt(data["finality"]["observed_pair_count"]),
                     _fmt(data["finality"]["unselected_chunk_count"]),
                     _fmt(data["finality"]["excluded_incomplete_edge_fragments"]),
                     classes])
    return _table(["Arm", "Status", "Chunks analysed", "Chunks absent",
                   "Edge fragments excluded", "Output provenance"], rows)


@builder("operating_points_by_event")
def _operating_points_by_event(facts):
    rows = []
    for event in SCORE_LABELS:
        entries = facts["primary"]["operating_point_by_event"][event]
        for threshold, r in sorted(entries.items(), key=lambda kv: float(kv[0])):
            rows.append([event, threshold, _fmt(r["identical_cutoff"]["mcc"], ".3f"),
                         _fmt(r["rate_matched"]["right_cutoff"], ".3f"),
                         _fmt(r["agreement_optimal"]["right_cutoff"], ".3f"),
                         _fmt(r["agreement_optimal"]["mcc"], ".3f")])
    return _table(["Event", "SpliceAI cutoff", "MCC at same cutoff", "Rate-matched OSAI cutoff",
                   "MCC-optimal OSAI cutoff", "Optimised MCC"], rows)


@builder("site_events_depth")
def _site_events_depth(facts):
    groups = {}
    for r in facts["primary"]["site_event_table"]:
        if r["threshold"] != 0.5:
            continue
        g = groups.setdefault((r["distance"], r["event"]), {"n":0, "bias":0., "mae":0., "left":0, "right":0})
        g["n"] += r["n"]
        g["bias"] += r["bias_right_minus_left"]*r["n"]
        g["mae"] += r["mae"]*r["n"]
        g["left"] += r["both_positive"]+r["left_only"]
        g["right"] += r["both_positive"]+r["right_only"]
    rows = []
    for distance in ("at_site","1-2","3-10","11-50","51-500",">500","no_site"):
        for event in SCORE_LABELS:
            if (distance,event) not in groups:
                continue
            r = groups[(distance,event)]
            rows.append([distance,event,_fmt(r["n"]),_fmt(r["bias"]/r["n"],"+.5f"),
                         _fmt(r["mae"]/r["n"],".5f"),_fmt(r["left"]),_fmt(r["right"]),
                         _fmt(r["right"]/r["left"] if r["left"] else None,".3f")])
    return _table(["Distance (bp)","Event","Paired n","Bias","MAE","SpliceAI ≥0.5","OSAI ≥0.5","Call ratio"],rows)


@builder("collapsed_view")
def _collapsed_view(facts):
    return _table(["Event","Exact-gene n","Collapsed allele n","Exact-gene MAE","Collapsed MAE","Collapsed bias"],
        [[r["event"],_fmt(r["primary_n"]),_fmt(r["n"]),_fmt(r["primary_mae"],".6f"),
          _fmt(r["mae"],".6f"),_fmt(r["bias_right_minus_left"],"+.6f")] for r in facts["primary"]["collapsed"]])


@builder("dominant_pair_strata")
def _dominant_pair_strata(facts):
    return _table(["Dominant-event pair","n","MAX bias","MAX MAE","Pearson r"],
        [[r["stratum"],_fmt(r["n"]),_fmt(r["bias"],"+.5f"),_fmt(r["mae"],".5f"),_fmt(r["pearson_r"],".3f")]
         for r in facts["primary"]["dominant_pair_strata"]])


@builder("discordance_depth")
def _discordance_depth(facts):
    taxonomy = facts["primary"]["discordance_depth"]
    rows = sorted(taxonomy["strata"],key=lambda r:-r["n"])[:25]
    return _table(["Event of largest difference","Direction","Distance (bp)","Nearest-site type","Top-k records"],
        [[r["event"],r["direction"].replace("left_higher","SpliceAI higher").replace("right_higher","OSAI higher"),
          r["distance"],r["site_type"],_fmt(r["n"])] for r in rows])


@builder("seed_method_strata")
def _seed_method_strata(facts):
    return _table(["Stratum dimension","Event","Method comparison","Strata","Defined ratios",
                   "Median MAE ratio","Fraction of defined ratios > 1"],
        [[r["dimension"],r["event"],r["model_arm"],_fmt(r["strata"]),_fmt(r["defined_ratios"]),
          _fmt(r["median_mae_ratio"],".3f"),_fmt(r["fraction_method_above_seed"],".3f")]
         for r in facts["seed_versus_model"]["stratum_summary"]])


def render(name: str, facts: Mapping) -> str:
    if name not in BUILDERS:
        raise KeyError(f"unknown table: {name} (known: {', '.join(sorted(BUILDERS))})")
    return BUILDERS[name](facts)


__all__ = ["render", "BUILDERS"]
