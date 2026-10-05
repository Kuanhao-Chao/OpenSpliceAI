"""Independent real-VCF check of depth statistics (standard library only).

No imports from the mapper, reducer, parser, site-index or statistics packages.
Requires a contiguous chunk window, and removes its incomplete outer groups.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

EVENTS = ("AG", "AL", "DG", "DL", "MAX")
THRESHOLDS = (0.05, 0.1, 0.2, 0.5, 0.8)


def annotation_sites(path):
    positions = defaultdict(lambda: defaultdict(set))
    with open(path) as h:
        for r in csv.DictReader(h, delimiter="\t"):
            starts = [int(x) + 1 for x in r["EXON_START"].split(",") if x]
            ends = [int(x) for x in r["EXON_END"].split(",") if x]
            exons = sorted(zip(starts, ends))
            if len(starts) != len(ends) or not exons:
                raise ValueError("invalid exon arrays")
            for _, end in exons[:-1]:
                positions[r["CHROM"]][end].add("donor" if r["STRAND"] == "+" else "acceptor")
            for start, _ in exons[1:]:
                positions[r["CHROM"]][start].add("acceptor" if r["STRAND"] == "+" else "donor")
    return {c: (sorted(v), v) for c, v in positions.items()}


def site_key(index, chrom, pos):
    if chrom not in index:
        return "no_site:no_site"
    positions, roles = index[chrom]
    i = bisect_left(positions, pos)
    candidates = positions[max(0, i - 1) : i + 1]
    distance = min(abs(p - pos) for p in candidates)
    kinds = set().union(*(roles[p] for p in candidates if abs(p - pos) == distance))
    kind = next(iter(kinds)) if len(kinds) == 1 else "ambiguous"
    label = next(
        name
        for maximum, name in (
            (0, "at_site"),
            (2, "1-2"),
            (10, "3-10"),
            (50, "11-50"),
            (500, "51-500"),
            (math.inf, ">500"),
        )
        if distance <= maximum
    )
    return label + ":" + kind


def parse_annotations(value, alt):
    genes = defaultdict(set)
    for text in value.split(","):
        fields = text.split("|")
        if len(fields) != 10 or fields[0] != alt or not fields[1]:
            continue
        try:
            scores = tuple(float(x) for x in fields[2:6])
            dps = tuple(int(x) for x in fields[6:10])
        except ValueError:
            continue
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            continue
        genes[fields[1]].add((scores, dps))
    return genes


def groups(paths, fields):
    key, sides = None, None
    for path in paths:
        with open(path) as h:
            for line in h:
                if line.startswith("#"):
                    continue
                row = line.rstrip("\n").split("\t")
                current = (row[0], int(row[1]), row[3], row[4])
                if key is not None and key != current:
                    yield key, sides
                    key = None
                if key is None:
                    key, sides = current, {s: defaultdict(set) for s in fields}
                info = dict(v.split("=", 1) for v in row[7].split(";") if "=" in v)
                for side, name in fields.items():
                    for gene, values in parse_annotations(info.get(name, ""), current[3]).items():
                        sides[side][gene].update(values)
    if key is not None:
        yield key, sides


VERSION = 2
BINS = 200
DOMAIN_SIZE = 100000
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-10
EQUIVALENCE = (0.01, 0.05, 0.1)
DIMENSIONS = ("chrom", "gene", "block_1mb", "substitution", "dominant_pair", "site_distance")
ARMS = {
    "primary": {"A_rs10_genomewide": ("left", "right")},
    "matched": {
        "B_seeds_rs10_rs13": ("left", "right"),
        "C_rs10_matched": ("reference", "left"),
        "D_rs13_matched": ("reference", "right"),
    },
}
FLOAT_FIELDS = (
    "sum_left",
    "sum_right",
    "sum_left2",
    "sum_right2",
    "sum_cross",
    "sum_abs_diff",
    "sum_sq_diff",
    "sum_quantized_abs_diff",
)
INTEGER_FIELDS = ("n", "left_zero", "right_zero", "exact_match")
CELLS = ("both_positive", "left_only", "right_only", "both_negative")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def contract():
    return {
        "version": VERSION,
        "score_bins": BINS,
        "thresholds": list(THRESHOLDS),
        "equivalence_tolerances": list(EQUIVALENCE),
        "expected_total_chunks": DOMAIN_SIZE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "verifier_sha256": digest(__file__),
    }


def input_rows(pairs):
    with open(pairs) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = [int(r["chunk_id"]) for r in rows]
    if not ids or ids != list(range(ids[0], ids[-1] + 1)) or not 1 <= ids[0] <= ids[-1] <= DOMAIN_SIZE:
        raise ValueError("independent check requires a nonempty contiguous ordered chunk window")
    return rows, ids


def validate_cache(data, pairs, sites, kind):
    _, ids = input_rows(pairs)
    expected = {
        "kind": kind,
        "pairs_sha256": digest(pairs),
        "sites_sha256": digest(sites),
        "chunk_ids": ids,
        "contract": contract(),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError("independent cache differs from requested " + key)
    data.pop("verification", None)


def interior(stream, include_first, include_last):
    previous, count = None, 0
    for current in stream:
        if count and (count > 1 or include_first):
            yield previous
        previous, count = current, count + 1
    if count and include_last and (count > 1 or include_first):
        yield previous


def fresh(histogram=False):
    result = {"moments": {k: 0.0 for k in FLOAT_FIELDS}, "tables": {str(t): [0, 0, 0, 0] for t in THRESHOLDS}}
    result["moments"].update({k: 0 for k in INTEGER_FIELDS})
    result["moments"]["equivalence"] = {str(t): 0 for t in EQUIVALENCE}
    if histogram:
        result.update(
            left_hist=[0] * BINS,
            right_hist=[0] * BINS,
            joint={},
            dp={
                str(t): {"both_above": 0, "eligible": 0, "within": {str(d): 0 for d in (0, 1, 2, 5, 10)}}
                for t in THRESHOLDS
            },
        )
    return result


def add(state, x, y, dx=None, dy=None):
    if x == 0 and y == 0:
        state["_zeros"] = state.get("_zeros", 0) + 1
        return
    m = state["moments"]
    delta = y - x
    distance = abs(delta)
    m["n"] += 1
    m["sum_left"] += x
    m["sum_right"] += y
    m["sum_left2"] += x * x
    m["sum_right2"] += y * y
    m["sum_cross"] += x * y
    m["sum_abs_diff"] += distance
    m["sum_sq_diff"] += delta * delta
    m["sum_quantized_abs_diff"] += max(0.0, distance - 0.005)
    m["left_zero"] += int(x == 0)
    m["right_zero"] += int(y == 0)
    m["exact_match"] += int(x == y)
    for t in EQUIVALENCE:
        m["equivalence"][str(t)] += int(distance <= t)
    if "joint" in state:
        # Independent decimal-grid binning, without production's float epsilon.
        xb, yb = min(BINS - 1, round(x * 100000) // 500), min(BINS - 1, round(y * 100000) // 500)
        state["left_hist"][xb] += 1
        state["right_hist"][yb] += 1
        key = str(xb * BINS + yb)
        state["joint"][key] = state["joint"].get(key, 0) + 1
    for t in THRESHOLDS:
        cell = 0 if x >= t and y >= t else 1 if x >= t else 2 if y >= t else 3
        state["tables"][str(t)][cell] += 1
        if "dp" in state and dx is not None and cell == 0:
            dp = state["dp"][str(t)]
            dp["eligible"] += 1
            dp["both_above"] += 1
            for tolerance in dp["within"]:
                dp["within"][tolerance] += int(abs(dx - dy) <= int(tolerance))


def dominant(scores):
    highest = max(scores)
    if highest == 0:
        return "NONE"
    winners = [e for e, v in zip(EVENTS, scores) if v == highest]
    return winners[0] if len(winners) == 1 else "TIE"


def flush_zeros(state):
    count = state.pop("_zeros", 0)
    if not count:
        return
    m = state["moments"]
    for field in INTEGER_FIELDS:
        m[field] += count
    for t in m["equivalence"]:
        m["equivalence"][t] += count
    for cells in state["tables"].values():
        cells[3] += count
    if "joint" in state:
        state["left_hist"][0] += count
        state["right_hist"][0] += count
        state["joint"]["0"] = state["joint"].get("0", 0) + count


def recompute(pairs, sites, kind):
    rows, ids = input_rows(pairs)
    index = annotation_sites(sites)
    if kind == "primary":
        stream = groups([r["prediction_vcf"] for r in rows], {"left": "SpliceAI", "right": "OpenSpliceAI"})
    else:
        left = groups([r["left_vcf"] for r in rows], {"reference": "SpliceAI", "left": "OpenSpliceAI"})
        right = groups([r["right_vcf"] for r in rows], {"right": "OpenSpliceAI"})

        def joined():
            from itertools import zip_longest

            for a, b in zip_longest(left, right):
                if a is None or b is None or a[0] != b[0]:
                    raise ValueError("input variant streams differ")
                yield a[0], {**a[1], **b[1]}

        stream = joined()
    tallies = {
        a: {"all": {e: fresh(True) for e in EVENTS}, "sites": {}, "strata": {d: {} for d in DIMENSIONS}}
        for a in ARMS[kind]
    }
    paired = 0
    for key, sides in interior(stream, ids[0] == 1, ids[-1] == DOMAIN_SIZE):
        valid = {s: {g: next(iter(v)) for g, v in genes.items() if len(v) == 1} for s, genes in sides.items()}
        common = set.intersection(*(set(v) for v in valid.values()))
        chrom, pos, ref, alt = key
        site = site_key(index, chrom, pos)
        start = ((pos - 1) // 1000000) * 1000000 + 1
        for gene in sorted(common):
            paired += 1
            for arm, (left, right) in ARMS[kind].items():
                (xs, dxs), (ys, dys) = valid[left][gene], valid[right][gene]
                a = tallies[arm]
                if site not in a["sites"]:
                    a["sites"][site] = {e: fresh(True) for e in EVENTS}
                keys = (
                    chrom,
                    gene,
                    f"{chrom}:{start}-{start + 999999}",
                    ref.upper() + ">" + alt.upper(),
                    dominant(xs) + ">" + dominant(ys),
                    site,
                )
                states = [a["all"], a["sites"][site]]
                for dimension, stratum in zip(DIMENSIONS, keys):
                    if stratum not in a["strata"][dimension]:
                        a["strata"][dimension][stratum] = {e: fresh() for e in EVENTS}
                    states.append(a["strata"][dimension][stratum])
                for e, x, y, dx, dy in zip(EVENTS, (*xs, max(xs)), (*ys, max(ys)), (*dxs, None), (*dys, None)):
                    for state in states:
                        add(state[e], x, y, dx, dy)
    for arm in tallies.values():
        groups_to_flush = [arm["all"], *arm["sites"].values()]
        for dimension in arm["strata"].values():
            groups_to_flush.extend(dimension.values())
        for events in groups_to_flush:
            for state in events.values():
                flush_zeros(state)
    return {
        "kind": kind,
        "chunks": len(ids),
        "chunk_ids": ids,
        "first_chunk": ids[0],
        "last_chunk": ids[-1],
        "paired_annotations": paired,
        "sites_sha256": digest(sites),
        "pairs_sha256": digest(pairs),
        "contract": contract(),
        "arms": tallies,
    }


def compare(independent, summary):
    try:
        return _compare(independent, summary)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("incomplete independent or mapper verification schema: " + str(exc)) from exc


def _compare(independent, summary):
    failures, checks, max_error = [], 0, 0.0
    kind = independent["kind"]
    arms = ARMS[kind]
    if independent.get("contract") != contract():
        raise ValueError("independent verifier contract differs")
    if set(independent["arms"]) != set(arms):
        raise ValueError("independent arms are incomplete")
    if independent["pairs_sha256"] != summary["pairs_sha256"]:
        raise ValueError("independent and mapper pair-list digests differ")
    config = summary["raw"]["config"]
    if independent["sites_sha256"] != config["sites_digest"]:
        raise ValueError("independent and mapper annotation digests differ")
    for name, value in [
        ("score_bins", BINS),
        ("thresholds", list(THRESHOLDS)),
        ("equivalence_tolerances", list(EQUIVALENCE)),
    ]:
        if config[name] != value:
            raise ValueError("mapper configuration differs: " + name)
    if summary["kind"] != ("reduced-concordance" if kind == "primary" else "reduced-seeds"):
        raise ValueError("comparison kind differs")
    if (
        sorted(map(int, summary["chunk_ids"])) != independent["chunk_ids"]
        or summary["finality"]["expected_total_chunks"] != DOMAIN_SIZE
    ):
        raise ValueError("comparison chunk domain differs")

    def equal(a, b, name, exact=False):
        nonlocal checks, max_error
        checks += 1
        if exact:
            ok = a == b
        else:
            error = abs(a - b)
            max_error = max(max_error, error)
            ok = math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)
        if not ok:
            failures.append(name)

    def moments(state, raw, name):
        for field in INTEGER_FIELDS:
            equal(state["moments"][field], raw["moments"][field], name + "/" + field, True)
        for field in FLOAT_FIELDS:
            equal(state["moments"][field], raw["moments"][field], name + "/" + field)
        equal(state["moments"]["equivalence"], raw["moments"]["equivalence"], name + "/equivalence", True)
        if set(raw["thresholds"]) != set(state["tables"]):
            raise ValueError(name + "/threshold schema differs")
        for t, cells in state["tables"].items():
            for cell, n in zip(CELLS, cells):
                equal(n, raw["thresholds"][t][cell], name + "/" + t + "/" + cell, True)

    def histograms(state, hist, joint, dp, name, event):
        equal(state["left_hist"], hist["left"], name + "/left histogram", True)
        equal(state["right_hist"], hist["right"], name + "/right histogram", True)
        if isinstance(joint, list):
            joint = {str(i): n for i, n in enumerate(joint) if n}
        equal(state["joint"], joint, name + "/joint histogram", True)
        if event != "MAX":
            if dp is None or set(dp) != set(state["dp"]):
                raise ValueError(name + "/missing DP state")
            equal(state["dp"], dp, name + "/DP counts", True)

    for arm in arms:
        a = independent["arms"][arm]
        raw = summary["raw"]["comparisons"][arm] if kind == "matched" else summary["raw"]
        if set(a["all"]) != set(EVENTS) or set(a["strata"]) != set(DIMENSIONS):
            raise ValueError(arm + "/incomplete event or dimension schema")
        equal(independent["paired_annotations"], raw["moments"]["MAX"]["n"], arm + "/paired count", True)
        for event in EVENTS:
            state = a["all"][event]
            name = arm + "/all/" + event
            moments(state, {"moments": raw["moments"][event], "thresholds": raw["thresholds"][event]}, name)
            histograms(
                state,
                raw["score_hist"][event],
                raw["joint_hist"][event],
                raw["dp"][event] if event != "MAX" else None,
                name,
                event,
            )
        if set(a["sites"]) != set(raw["depth"]["site_histograms"]):
            failures.append(arm + "/site keys")
        for site, events in a["sites"].items():
            if set(events) != set(EVENTS):
                raise ValueError(arm + "/incomplete site events")
            for event, state in events.items():
                h = raw["depth"]["site_histograms"][site][event]
                dp = raw["depth"]["site_dp"][site][event] if event != "MAX" else None
                histograms(state, h, h["joint"], dp, arm + "/" + site + "/" + event, event)
        for dimension in DIMENSIONS:
            if set(a["strata"][dimension]) != set(raw["depth"]["event_strata"][dimension]):
                failures.append(arm + "/" + dimension + "/stratum keys")
            for key, events in a["strata"][dimension].items():
                if set(events) != set(EVENTS):
                    raise ValueError(arm + "/" + dimension + "/incomplete events")
                for event, state in events.items():
                    entry = (
                        raw["strata"][dimension][key]
                        if event == "MAX"
                        else raw["depth"]["event_strata"][dimension][key][event]
                    )
                    moments(state, entry, arm + "/" + dimension + "/" + key + "/" + event)
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "max_continuous_absolute_error": max_error,
        "failures": failures,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "scope": "all moments and exact counts; global/site joint+marginal histograms and DP; six dimensions with AG/AL/DG/DL/MAX",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--sites-file", required=True)
    p.add_argument("--kind", choices=tuple(ARMS), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--compare-summary")
    p.add_argument("--reuse-independent")
    a = p.parse_args()
    if a.reuse_independent:
        result = json.loads(Path(a.reuse_independent).read_text())
        validate_cache(result, a.pairs_file, a.sites_file, a.kind)
    else:
        result = recompute(a.pairs_file, a.sites_file, a.kind)
    if a.compare_summary:
        result["verification"] = compare(result, json.loads(Path(a.compare_summary).read_text()))
        result["verification"].update(
            summary_path=str(Path(a.compare_summary).resolve()), summary_sha256=digest(a.compare_summary)
        )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
    return int(result.get("verification", {}).get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
