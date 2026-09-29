"""Event-level strata and a shared-observation three-way comparison.

The existing aggregate remains the source of global statistics. Only small
site-distance strata retain joint histograms; genes/blocks retain moments and
exact threshold tables, avoiding a quadratic histogram for every gene.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from validation.full_snv_concordance.aggregate import (
    Aggregate, AnalysisConfig, DP_TOLERANCES, EVENTS, SCORE_LABELS, Moments,
    _add_dp_to_state, _add_table, _bin_index, _dominant, _empty_table,
    _merge_dp, _number_key, derive_threshold_metrics,
)
from validation.full_snv_concordance.vcf import VariantGroup

DEPTH_VERSION = 1
DIMENSIONS = ("chrom", "gene", "block_1mb", "substitution", "dominant_pair", "site_distance")
COMPARISONS = {
    "B_seeds_rs10_rs13": ("left", "right"),
    "C_rs10_matched": ("reference", "left"),
    "D_rs13_matched": ("reference", "right"),
}


def _new_dp(config):
    return {e: {_number_key(t): {"both_above": 0, "eligible": 0,
                "within": {str(d): 0 for d in DP_TOLERANCES}}
                for t in config.thresholds} for e in EVENTS}


class DepthAggregate(Aggregate):
    def __init__(self, config=None, sites=None):
        super().__init__(config, sites)
        self.event_strata = {d: {} for d in DIMENSIONS}
        self.zero_pairs = {d: Counter() for d in DIMENSIONS}
        self.site_histograms = {}
        self.site_dp = {}

    def _new_entry(self):
        return {e: {"moments": Moments(),
                    "thresholds": {_number_key(t): _empty_table() for t in self.config.thresholds}}
                for e in EVENTS}

    def add_pair(self, variant_token, chrom, gene, left, right, pos=None, ref=None, alt=None):
        super().add_pair(variant_token, chrom, gene, left, right, pos, ref, alt)
        if pos is None:
            _, pos, ref, alt = variant_token.rsplit(":", 3)
            pos = int(pos)
        if self.sites is None or self.sites.digest != self.config.sites_digest:
            raise ValueError("depth analysis requires the digest-matched site index")
        site = self.sites.stratum_key(chrom, pos)
        start = ((pos - 1) // 1_000_000) * 1_000_000 + 1
        keys = {"chrom": chrom, "gene": gene,
                "block_1mb": f"{chrom}:{start}-{start + 999_999}",
                "substitution": f"{ref.upper()}>{alt.upper()}",
                "dominant_pair": f"{_dominant(left)}>{_dominant(right)}", "site_distance": site}
        all_zero = not any(left.scores) and not any(right.scores)
        for dimension, key in keys.items():
            if all_zero:
                # Almost every observation is zero on both sides. Bulk-add its
                # exact moments/tables at serialization, instead of repeating
                # 24 floating-point and threshold updates for every such row.
                self.zero_pairs[dimension][key] += 1
                continue
            groups = self.event_strata[dimension]
            if key not in groups:
                groups[key] = self._new_entry()
            for event, x, y in zip(EVENTS, left.scores, right.scores):
                state = groups[key][event]
                state["moments"].add(x, y, self.config.equivalence_tolerances)
                for t in self.config.thresholds:
                    _add_table(state["thresholds"][_number_key(t)], x, y, t)
        bins = self.config.score_bins
        if site not in self.site_histograms:
            self.site_histograms[site] = {e: {"left": [0]*bins, "right": [0]*bins, "joint": Counter()}
                                          for e in SCORE_LABELS}
            self.site_dp[site] = _new_dp(self.config)
        xs, ys = (*left.scores, max(left.scores)), (*right.scores, max(right.scores))
        for event, x, y in zip(SCORE_LABELS, xs, ys):
            state = self.site_histograms[site][event]
            xb, yb = _bin_index(x, bins), _bin_index(y, bins)
            state["left"][xb] += 1
            state["right"][yb] += 1
            state["joint"][str(xb*bins + yb)] += 1
        if not all_zero or 0.0 in self.config.thresholds:
            _add_dp_to_state({"dp": self.site_dp[site]}, self.config, left, right)

    def _flush_zeros(self):
        for dimension, groups in self.zero_pairs.items():
            for key, count in groups.items():
                if key not in self.event_strata[dimension]:
                    self.event_strata[dimension][key] = self._new_entry()
                for state in self.event_strata[dimension][key].values():
                    m = state["moments"]
                    m.n += count
                    m.left_zero += count
                    m.right_zero += count
                    m.exact_match += count
                    for tolerance in self.config.equivalence_tolerances:
                        key_t = _number_key(tolerance)
                        m.equivalence[key_t] = m.equivalence.get(key_t, 0) + count
                    for threshold, table in state["thresholds"].items():
                        table["both_positive" if float(threshold) == 0 else "both_negative"] += count
            groups.clear()

    def merge(self, other):
        if not isinstance(other, DepthAggregate):
            raise ValueError("cannot merge a legacy aggregate into a depth analysis")
        self._flush_zeros()
        other._flush_zeros()
        super().merge(other)
        for dimension, groups in other.event_strata.items():
            for key, events in groups.items():
                if key not in self.event_strata[dimension]:
                    self.event_strata[dimension][key] = deepcopy(events)
                    continue
                for event, state in events.items():
                    target = self.event_strata[dimension][key][event]
                    target["moments"].merge(state["moments"])
                    for t, table in state["thresholds"].items():
                        for cell, n in table.items():
                            target["thresholds"][t][cell] += n
        for key, events in other.site_histograms.items():
            if key not in self.site_histograms:
                self.site_histograms[key] = deepcopy(events)
                self.site_dp[key] = deepcopy(other.site_dp[key])
                continue
            for event, state in events.items():
                target = self.site_histograms[key][event]
                for side in ("left", "right"):
                    target[side] = [a+b for a, b in zip(target[side], state[side])]
                target["joint"].update(state["joint"])
            _merge_dp(self.site_dp[key], other.site_dp[key])

    def to_dict(self):
        self._flush_zeros()
        data = super().to_dict()
        data["depth"] = {
            "version": DEPTH_VERSION,
            "event_strata": {d: {k: {e: {"moments": s["moments"].to_dict(),
                                           "thresholds": s["thresholds"]} for e, s in events.items()}
                                  for k, events in groups.items()} for d, groups in self.event_strata.items()},
            "site_histograms": self.site_histograms, "site_dp": self.site_dp,
        }
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get("depth", {}).get("version") != DEPTH_VERSION:
            raise ValueError("missing or incompatible depth aggregate version")
        result = super().from_dict(data)
        depth = data["depth"]
        result.event_strata = {d: {k: {e: {"moments": Moments.from_dict(s["moments"]),
                                           "thresholds": deepcopy(s["thresholds"])} for e, s in events.items()}
                                  for k, events in groups.items()} for d, groups in depth["event_strata"].items()}
        result.site_histograms = deepcopy(depth["site_histograms"])
        for events in result.site_histograms.values():
            for state in events.values():
                state["joint"] = Counter(state["joint"])
        result.site_dp = deepcopy(depth["site_dp"])
        return result

    def derived(self):
        self._flush_zeros()
        data = super().derived()
        data["event_strata"] = {d: {k: {e: {"metrics": s["moments"].derived(),
                 "thresholds": {t: derive_threshold_metrics(c) for t, c in s["thresholds"].items()}}
                 for e, s in events.items()} for k, events in groups.items()}
                 for d, groups in self.event_strata.items()}
        data["depth_version"] = DEPTH_VERSION
        return data


class MatchedAggregate:
    """Three comparisons fed exactly the same non-conflicting gene triples.

    Filtering occurs after boundary fragments have been rejoined. Filtering a
    fragment before reduction could conceal a conflict in its neighbour.
    """
    def __init__(self, config=None, sites=None):
        self.config = config or AnalysisConfig()
        self.sites = sites
        self.coverage = Counter()
        self.comparisons = {name: DepthAggregate(self.config, sites) for name in COMPARISONS}

    def add_group(self, group):
        valid = {side: {gene: next(iter(values)) for gene, values in group.sides.get(side, {}).items()
                        if len(values) == 1} for side in ("reference", "left", "right")}
        genes = set(valid["reference"]) & set(valid["left"]) & set(valid["right"])
        union = set().union(*(set(v) for v in valid.values()))
        self.coverage["variant_groups"] += 1
        self.coverage["source_rows"] += group.rows
        self.coverage["shared_annotations"] += len(genes)
        self.coverage["excluded_nonshared_annotations"] += len(union-genes)
        for name, (left, right) in COMPARISONS.items():
            child = self.comparisons[name]
            child.sites = self.sites
            paired = VariantGroup(group.key, rows=group.rows)
            for gene in sorted(genes):
                paired.add("left", valid[left][gene])
                paired.add("right", valid[right][gene])
            child.add_group(paired)

    def merge(self, other):
        if not isinstance(other, MatchedAggregate) or self.config != other.config:
            raise ValueError("cannot merge different matched configurations")
        self.coverage.update(other.coverage)
        for name in COMPARISONS:
            self.comparisons[name].merge(other.comparisons[name])

    def to_dict(self):
        return {"depth_version": DEPTH_VERSION, "config": self.config.to_dict(),
                "domain": "exact-three-way-variant-gene-intersection",
                "coverage": dict(self.coverage),
                "comparisons": {name: a.to_dict() for name, a in self.comparisons.items()}}

    @classmethod
    def from_dict(cls, data):
        if data.get("depth_version") != DEPTH_VERSION or data.get("domain") != "exact-three-way-variant-gene-intersection":
            raise ValueError("missing or incompatible matched domain")
        result = cls(AnalysisConfig.from_dict(data["config"]))
        result.coverage = Counter(data["coverage"])
        result.comparisons = {name: DepthAggregate.from_dict(data["comparisons"][name]) for name in COMPARISONS}
        return result

    def derived(self):
        return {"coverage": dict(self.coverage),
                "comparisons": {name: a.derived() for name, a in self.comparisons.items()}}
