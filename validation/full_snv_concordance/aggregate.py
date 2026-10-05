"""Mergeable statistics for score concordance and seed reproducibility."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import heapq
import math
import random
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from .sites import SiteIndex
from .vcf import Annotation, EVENTS, VariantGroup


SCORE_LABELS: Tuple[str, ...] = EVENTS + ("MAX",)
DOMINANT_LABELS: Tuple[str, ...] = EVENTS + ("TIE", "NONE")
DP_TOLERANCES: Tuple[int, ...] = (0, 1, 2, 5, 10)
SIGNAL_THRESHOLDS: Tuple[float, ...] = (0.1, 0.2, 0.5)
STANDARD_STRATA: Tuple[str, ...] = (
    "chrom",
    "gene",
    "block_1mb",
    "substitution",
    "dominant_pair",
    "site_event",
)
#: `site_distance` is supported but deliberately not in STANDARD_STRATA: it is the
#: one stratum that needs an annotation from outside the two score files, so it is
#: opt-in via `--sites-file` rather than something every caller must provide.
SUPPORTED_STRATA: Tuple[str, ...] = STANDARD_STRATA + ("site_distance",)


@dataclass(frozen=True)
class AnalysisConfig:
    thresholds: Tuple[float, ...] = (0.1, 0.2, 0.5, 0.8)
    score_bins: int = 100
    sample_size: int = 10_000
    top_k: int = 1_000
    equivalence_tolerances: Tuple[float, ...] = (0.01, 0.05, 0.10)
    strata: Tuple[str, ...] = STANDARD_STRATA
    bootstrap_replicates: int = 200
    bootstrap_seed: int = 20260801
    #: SHA-256 of the annotation the site_distance stratum was derived from. It
    #: lives in the config because `Aggregate.merge` refuses to merge aggregates
    #: whose configs differ -- so shards built against different (or missing)
    #: annotations fail closed through the mechanism that already exists.
    sites_digest: str = ""

    def __post_init__(self) -> None:
        if self.score_bins < 2:
            raise ValueError("score_bins must be at least 2")
        if self.sample_size < 0 or self.top_k < 0:
            raise ValueError("sample_size and top_k must be nonnegative")
        if not self.thresholds:
            raise ValueError("at least one threshold is required")
        if any(t < 0.0 or t > 1.0 for t in self.thresholds):
            raise ValueError("thresholds must be in [0, 1]")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be nonnegative")
        unknown_strata = set(self.strata) - set(SUPPORTED_STRATA)
        if unknown_strata:
            raise ValueError(f"unsupported strata: {', '.join(sorted(unknown_strata))}")
        if "site_distance" in self.strata and not self.sites_digest:
            raise ValueError("the site_distance stratum requires a sites file (sites_digest)")

    def to_dict(self) -> dict:
        return {
            "thresholds": list(self.thresholds),
            "score_bins": self.score_bins,
            "sample_size": self.sample_size,
            "top_k": self.top_k,
            "equivalence_tolerances": list(self.equivalence_tolerances),
            "strata": list(self.strata),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "sites_digest": self.sites_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "AnalysisConfig":
        return cls(
            thresholds=tuple(float(x) for x in data.get("thresholds", (0.1, 0.2, 0.5, 0.8))),
            score_bins=int(data.get("score_bins", 100)),
            sample_size=int(data.get("sample_size", 10_000)),
            top_k=int(data.get("top_k", 1_000)),
            equivalence_tolerances=tuple(float(x) for x in data.get("equivalence_tolerances", (0.01, 0.05, 0.10))),
            strata=tuple(str(x) for x in data.get("strata", STANDARD_STRATA)),
            bootstrap_replicates=int(data.get("bootstrap_replicates", 200)),
            bootstrap_seed=int(data.get("bootstrap_seed", 20260801)),
            sites_digest=str(data.get("sites_digest", "")),
        )


@dataclass
class Moments:
    n: int = 0
    sum_left: float = 0.0
    sum_right: float = 0.0
    sum_left2: float = 0.0
    sum_right2: float = 0.0
    sum_cross: float = 0.0
    sum_abs_diff: float = 0.0
    sum_sq_diff: float = 0.0
    sum_quantized_abs_diff: float = 0.0
    left_zero: int = 0
    right_zero: int = 0
    exact_match: int = 0
    equivalence: MutableMapping[str, int] = field(default_factory=dict)

    def add(self, left: float, right: float, tolerances: Sequence[float]) -> None:
        diff = right - left
        abs_diff = abs(diff)
        self.n += 1
        self.sum_left += left
        self.sum_right += right
        self.sum_left2 += left * left
        self.sum_right2 += right * right
        self.sum_cross += left * right
        self.sum_abs_diff += abs_diff
        self.sum_sq_diff += diff * diff
        # SpliceAI is published to two decimals: discrepancies <= 0.005 are
        # indistinguishable from output quantization.
        self.sum_quantized_abs_diff += max(0.0, abs_diff - 0.005)
        self.left_zero += int(left == 0.0)
        self.right_zero += int(right == 0.0)
        self.exact_match += int(left == right)
        for tolerance in tolerances:
            key = _number_key(tolerance)
            self.equivalence[key] = self.equivalence.get(key, 0) + int(abs_diff <= tolerance)

    def merge(self, other: "Moments") -> None:
        for name in (
            "n",
            "sum_left",
            "sum_right",
            "sum_left2",
            "sum_right2",
            "sum_cross",
            "sum_abs_diff",
            "sum_sq_diff",
            "sum_quantized_abs_diff",
            "left_zero",
            "right_zero",
            "exact_match",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key, value in other.equivalence.items():
            self.equivalence[key] = self.equivalence.get(key, 0) + value

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "sum_left": self.sum_left,
            "sum_right": self.sum_right,
            "sum_left2": self.sum_left2,
            "sum_right2": self.sum_right2,
            "sum_cross": self.sum_cross,
            "sum_abs_diff": self.sum_abs_diff,
            "sum_sq_diff": self.sum_sq_diff,
            "sum_quantized_abs_diff": self.sum_quantized_abs_diff,
            "left_zero": self.left_zero,
            "right_zero": self.right_zero,
            "exact_match": self.exact_match,
            "equivalence": dict(self.equivalence),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Moments":
        return cls(
            n=int(data.get("n", 0)),
            sum_left=float(data.get("sum_left", 0.0)),
            sum_right=float(data.get("sum_right", 0.0)),
            sum_left2=float(data.get("sum_left2", 0.0)),
            sum_right2=float(data.get("sum_right2", 0.0)),
            sum_cross=float(data.get("sum_cross", 0.0)),
            sum_abs_diff=float(data.get("sum_abs_diff", 0.0)),
            sum_sq_diff=float(data.get("sum_sq_diff", 0.0)),
            sum_quantized_abs_diff=float(data.get("sum_quantized_abs_diff", 0.0)),
            left_zero=int(data.get("left_zero", 0)),
            right_zero=int(data.get("right_zero", 0)),
            exact_match=int(data.get("exact_match", 0)),
            equivalence={str(k): int(v) for k, v in data.get("equivalence", {}).items()},
        )

    def derived(self) -> dict:
        if self.n == 0:
            return {
                "n": 0,
                "mean_left": None,
                "mean_right": None,
                "bias_right_minus_left": None,
                "pearson_r": None,
                "lin_ccc": None,
                "mae": None,
                "rmse": None,
                "exact_match_rate": None,
            }
        n = float(self.n)
        mean_left = self.sum_left / n
        mean_right = self.sum_right / n
        var_left = max(0.0, self.sum_left2 / n - mean_left * mean_left)
        var_right = max(0.0, self.sum_right2 / n - mean_right * mean_right)
        covariance = self.sum_cross / n - mean_left * mean_right
        pearson_denominator = math.sqrt(var_left * var_right)
        ccc_denominator = var_left + var_right + (mean_left - mean_right) ** 2
        return {
            "n": self.n,
            "mean_left": mean_left,
            "mean_right": mean_right,
            "bias_right_minus_left": mean_right - mean_left,
            "pearson_r": covariance / pearson_denominator if pearson_denominator else None,
            "lin_ccc": 2.0 * covariance / ccc_denominator if ccc_denominator else None,
            "mae": self.sum_abs_diff / n,
            "rmse": math.sqrt(self.sum_sq_diff / n),
            "mean_quantization_adjusted_abs_diff": self.sum_quantized_abs_diff / n,
            "left_zero_rate": self.left_zero / n,
            "right_zero_rate": self.right_zero / n,
            "exact_match_rate": self.exact_match / n,
            "practical_equivalence": {key: value / n for key, value in sorted(self.equivalence.items())},
        }


@lru_cache(maxsize=None)
def _bin_index(value: float, bins: int) -> int:
    """Bin a score in [0, 1] without losing the exact grid points to float error.

    ``int(value * bins)`` looks right and is not: a two-decimal value whose binary
    representation sits a hair below the exact multiple -- 0.29, 0.57, 0.58 at 100
    bins -- lands one bin low, which leaves bins empty and merges their contents
    into the neighbour below. The comparator publishes two decimals, so those are
    exactly the values that matter. A small epsilon pulls them back onto the grid
    without moving any value that is genuinely below it.
    """
    return min(bins - 1, max(0, int(value * bins + 1e-9)))


def _number_key(value: float) -> str:
    return f"{value:.10g}"


def _empty_table() -> Dict[str, int]:
    return {"both_positive": 0, "left_only": 0, "right_only": 0, "both_negative": 0}


def _dominant(annotation: Annotation) -> str:
    maximum = max(annotation.scores)
    if maximum == 0.0:
        return "NONE"
    indices = [i for i, value in enumerate(annotation.scores) if value == maximum]
    return "TIE" if len(indices) != 1 else EVENTS[indices[0]]


@lru_cache(maxsize=None)
def _signal_keys_for(thresholds: Tuple[float, ...]) -> Tuple[Tuple[str, float], ...]:
    values = sorted(set(SIGNAL_THRESHOLDS).union(thresholds))
    return (("either_gt_0", 0.0),) + tuple((f"either_ge_{_number_key(value)}", value) for value in values)


def _signal_keys(config: AnalysisConfig) -> Tuple[Tuple[str, float], ...]:
    return _signal_keys_for(tuple(config.thresholds))


def _new_score_state(config: AnalysisConfig) -> dict:
    """Create a fully additive, bounded-memory score-comparison view."""

    bins = config.score_bins
    return {
        "moments": {label: Moments() for label in SCORE_LABELS},
        "score_hist": {label: {"left": [0] * bins, "right": [0] * bins} for label in SCORE_LABELS},
        "joint_hist": {label: [0] * (bins * bins) for label in SCORE_LABELS},
        "diff_hist": {label: [0] * (2 * bins + 1) for label in SCORE_LABELS},
        "thresholds": {label: {_number_key(t): _empty_table() for t in config.thresholds} for label in SCORE_LABELS},
        "dominant": {left: {right: 0 for right in DOMINANT_LABELS} for left in DOMINANT_LABELS},
        "dp": {
            event: {
                _number_key(t): {
                    "both_above": 0,
                    "eligible": 0,
                    "within": {_number_key(float(distance)): 0 for distance in DP_TOLERANCES},
                }
                for t in config.thresholds
            }
            for event in EVENTS
        },
    }


def _score_state_to_dict(state: Mapping) -> dict:
    return {
        "moments": {label: state["moments"][label].to_dict() for label in SCORE_LABELS},
        "score_hist": state["score_hist"],
        "joint_hist": state["joint_hist"],
        "diff_hist": state["diff_hist"],
        "thresholds": state["thresholds"],
        "dominant": state["dominant"],
        "dp": state["dp"],
    }


def _score_state_from_dict(config: AnalysisConfig, data: Mapping) -> dict:
    state = _new_score_state(config)
    state["moments"] = {label: Moments.from_dict(data.get("moments", {}).get(label, {})) for label in SCORE_LABELS}
    for name in ("score_hist", "joint_hist", "diff_hist", "thresholds", "dominant", "dp"):
        if name in data:
            state[name] = data[name]
    return state


def _merge_count_list(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("cannot merge histograms with different lengths")
    return [int(a) + int(b) for a, b in zip(left, right)]


def _merge_score_state(target: dict, incoming: Mapping) -> None:
    for label in SCORE_LABELS:
        target["moments"][label].merge(incoming["moments"][label])
        for side in ("left", "right"):
            target["score_hist"][label][side] = _merge_count_list(
                target["score_hist"][label][side], incoming["score_hist"][label][side]
            )
        target["joint_hist"][label] = _merge_count_list(target["joint_hist"][label], incoming["joint_hist"][label])
        target["diff_hist"][label] = _merge_count_list(target["diff_hist"][label], incoming["diff_hist"][label])
        for threshold, table in target["thresholds"][label].items():
            for cell in table:
                table[cell] += int(incoming["thresholds"][label][threshold][cell])
    for left_label in DOMINANT_LABELS:
        for right_label in DOMINANT_LABELS:
            target["dominant"][left_label][right_label] += int(incoming["dominant"][left_label][right_label])
    _merge_dp(target["dp"], incoming["dp"])


def _merge_dp(target: dict, incoming: Mapping) -> None:
    for event in EVENTS:
        for threshold, values in target[event].items():
            source = incoming[event][threshold]
            for metric_name in ("both_above", "eligible"):
                values[metric_name] += int(source[metric_name])
            for tolerance in values["within"]:
                values["within"][tolerance] += int(source["within"][tolerance])


def _add_table(table: MutableMapping[str, int], left: float, right: float, threshold: float) -> None:
    if left >= threshold and right >= threshold:
        table["both_positive"] += 1
    elif left >= threshold:
        table["left_only"] += 1
    elif right >= threshold:
        table["right_only"] += 1
    else:
        table["both_negative"] += 1


def _add_score_to_state(
    state: dict,
    config: AnalysisConfig,
    label: str,
    left: float,
    right: float,
) -> None:
    state["moments"][label].add(left, right, config.equivalence_tolerances)
    bins = config.score_bins
    left_bin = _bin_index(left, bins)
    right_bin = _bin_index(right, bins)
    state["score_hist"][label]["left"][left_bin] += 1
    state["score_hist"][label]["right"][right_bin] += 1
    state["joint_hist"][label][left_bin * bins + right_bin] += 1
    diff_bin = min(2 * bins, max(0, int(round((right - left + 1.0) * bins))))
    state["diff_hist"][label][diff_bin] += 1
    for threshold in config.thresholds:
        _add_table(
            state["thresholds"][label][_number_key(threshold)],
            left,
            right,
            threshold,
        )


def _add_dp_to_state(state: dict, config: AnalysisConfig, left: Annotation, right: Annotation) -> None:
    for index, event in enumerate(EVENTS):
        for threshold in config.thresholds:
            if left.scores[index] >= threshold and right.scores[index] >= threshold:
                values = state["dp"][event][_number_key(threshold)]
                values["both_above"] += 1
                # DP=0 denotes the variant itself, so it remains eligible.
                values["eligible"] += 1
                distance = abs(left.dps[index] - right.dps[index])
                for tolerance in DP_TOLERANCES:
                    values["within"][_number_key(float(tolerance))] += int(distance <= tolerance)


def _collapsed_annotation(annotations: Mapping[str, Annotation]) -> Annotation:
    """Collapse genes independently per event, retaining a deterministic DP."""

    scores = []
    dps = []
    ordered = sorted(annotations.items())
    for index in range(len(EVENTS)):
        # Prefer the highest score; ties use nearest site, signed DP, then gene.
        gene, annotation = min(
            ordered,
            key=lambda item: (
                -item[1].scores[index],
                abs(item[1].dps[index]),
                item[1].dps[index],
                item[0],
            ),
        )
        del gene
        scores.append(annotation.scores[index])
        dps.append(annotation.dps[index])
    return Annotation("*", "__VARIANT_MAX_ACROSS_GENES__", tuple(scores), tuple(dps))


class Aggregate:
    """Additive analysis state that can be merged without row-level storage."""

    schema_version = 2

    def __init__(self, config: Optional[AnalysisConfig] = None,
                 sites: Optional["SiteIndex"] = None) -> None:
        self.config = config or AnalysisConfig()
        # The index itself is data, not configuration: only its digest travels in
        # the config (and therefore into the merge guard and the shard payload).
        self.sites = sites
        self.coverage: Counter[str] = Counter()
        self.moments: Dict[str, Moments] = {label: Moments() for label in SCORE_LABELS}
        bins = self.config.score_bins
        self.score_hist = {label: {"left": [0] * bins, "right": [0] * bins} for label in SCORE_LABELS}
        self.joint_hist = {label: [0] * (bins * bins) for label in SCORE_LABELS}
        self.diff_hist = {label: [0] * (2 * bins + 1) for label in SCORE_LABELS}
        self.thresholds = {
            label: {_number_key(t): _empty_table() for t in self.config.thresholds} for label in SCORE_LABELS
        }
        self.dominant = {left: {right: 0 for right in DOMINANT_LABELS} for left in DOMINANT_LABELS}
        self.dp = {
            event: {
                _number_key(t): {
                    "both_above": 0,
                    "eligible": 0,
                    "within": {_number_key(float(d)): 0 for d in DP_TOLERANCES},
                }
                for t in self.config.thresholds
            }
            for event in EVENTS
        }
        # A second exact-gene comparison rounds the right score to the same
        # two-decimal precision as published SpliceAI output.  It is kept
        # separately so the raw comparison remains the primary estimand.
        self.rounded_right_moments: Dict[str, Moments] = {label: Moments() for label in SCORE_LABELS}
        self.rounded_right_thresholds = {
            label: {_number_key(t): _empty_table() for t in self.config.thresholds} for label in SCORE_LABELS
        }
        self.signal_subset_moments = {
            label: {key: Moments() for key, _threshold in _signal_keys(self.config)} for label in SCORE_LABELS
        }
        # Annotation-agnostic sensitivity analysis: one pair per genomic
        # allele after event-wise maximum collapse across every valid gene.
        self.variant_view = _new_score_state(self.config)
        # Only MAX moments and threshold tables are retained for strata.
        self.strata: Dict[str, Dict[str, dict]] = {name: {} for name in self.config.strata}
        self._sample_heap: list[tuple[int, str, dict]] = []
        self._top_heap: list[tuple[float, str, dict]] = []

    def add_group(self, group: VariantGroup) -> None:
        self.coverage["variant_groups"] += 1
        self.coverage["source_rows"] += group.rows
        self.coverage["invalid_annotations"] += group.invalid_annotations
        self.coverage["annotation_observations"] += group.annotation_observations
        self.coverage["duplicate_annotations"] += group.duplicate_annotations

        left, left_conflicts = self._valid_side(group, "left")
        right, right_conflicts = self._valid_side(group, "right")
        self.coverage["left_valid_annotations"] += len(left)
        self.coverage["right_valid_annotations"] += len(right)
        self.coverage["left_conflicts"] += left_conflicts
        self.coverage["right_conflicts"] += right_conflicts

        common = left.keys() & right.keys()
        left_only = left.keys() - right.keys()
        right_only = right.keys() - left.keys()
        self.coverage["paired_annotations"] += len(common)
        self.coverage["left_only_annotations"] += len(left_only)
        self.coverage["right_only_annotations"] += len(right_only)
        if left and right and not common:
            self.coverage["gene_mismatch_variants"] += 1
        if not right:
            self.coverage["groups_without_right_prediction"] += 1
        if not left:
            self.coverage["groups_without_left_prediction"] += 1

        if left and right:
            self.coverage["variant_collapsed_pairs"] += 1
            self._add_variant_pair(_collapsed_annotation(left), _collapsed_annotation(right))
        elif left:
            self.coverage["variant_collapsed_left_only"] += 1
        elif right:
            self.coverage["variant_collapsed_right_only"] += 1

        for gene in sorted(common):
            self.add_pair(
                group.key.token(),
                group.key.chrom,
                gene,
                left[gene],
                right[gene],
                pos=group.key.pos,
                ref=group.key.ref,
                alt=group.key.alt,
            )

    @staticmethod
    def _valid_side(group: VariantGroup, side: str) -> tuple[Dict[str, Annotation], int]:
        valid: Dict[str, Annotation] = {}
        conflicts = 0
        for gene, values in group.sides.get(side, {}).items():
            if len(values) == 1:
                valid[gene] = next(iter(values))
            elif len(values) > 1:
                conflicts += 1
        return valid, conflicts

    def add_pair(
        self,
        variant_token: str,
        chrom: str,
        gene: str,
        left: Annotation,
        right: Annotation,
        pos: Optional[int] = None,
        ref: Optional[str] = None,
        alt: Optional[str] = None,
    ) -> None:
        left_values = tuple(left.scores) + (max(left.scores),)
        right_values = tuple(right.scores) + (max(right.scores),)
        for label, x, y in zip(SCORE_LABELS, left_values, right_values):
            self._add_score(label, x, y)
        self.dominant[_dominant(left)][_dominant(right)] += 1
        _add_dp_to_state({"dp": self.dp}, self.config, left, right)

        payload = {
            "variant": variant_token,
            "chrom": chrom,
            "gene": gene,
            "left_scores": list(left.scores),
            "right_scores": list(right.scores),
            "left_dps": list(left.dps),
            "right_dps": list(right.dps),
            "max_abs_difference": max(abs(x - y) for x, y in zip(left.scores, right.scores)),
        }
        self._consider_sample(f"{variant_token}|{gene}", payload)
        self._consider_top(f"{variant_token}|{gene}", payload)
        if "chrom" in self.strata:
            self._add_stratum("chrom", chrom, left_values[-1], right_values[-1])
        if "gene" in self.strata:
            self._add_stratum("gene", gene, left_values[-1], right_values[-1])
        if pos is None or ref is None or alt is None:
            try:
                _token_chrom, token_pos, token_ref, token_alt = variant_token.rsplit(":", 3)
                pos = int(token_pos)
                ref = token_ref
                alt = token_alt
            except (TypeError, ValueError):
                pos = None
        if "block_1mb" in self.strata and pos is not None:
            start = ((pos - 1) // 1_000_000) * 1_000_000 + 1
            block = f"{chrom}:{start}-{start + 999_999}"
            self._add_stratum("block_1mb", block, left_values[-1], right_values[-1])
        if "substitution" in self.strata and ref is not None and alt is not None:
            substitution = f"{str(ref).upper()}>{str(alt).upper()}"
            self._add_stratum("substitution", substitution, left_values[-1], right_values[-1])
        left_dominant = _dominant(left)
        right_dominant = _dominant(right)
        if "dominant_pair" in self.strata:
            self._add_stratum(
                "dominant_pair",
                f"{left_dominant}>{right_dominant}",
                left_values[-1],
                right_values[-1],
            )
        if "site_event" in self.strata:
            for index, event in enumerate(EVENTS):
                self._add_stratum("site_event", event, left.scores[index], right.scores[index])
        if "site_distance" in self.strata:
            if self.sites is None:
                # Raised here rather than in __init__ because the reducer rebuilds
                # populated aggregates from JSON and never adds a row; this fires
                # exactly when a mapper would otherwise drop the stratum silently.
                raise ValueError("the site_distance stratum requires a SiteIndex")
        if "site_distance" in self.strata and pos is not None:
            # Distance to the nearest annotated splice site, keyed as
            # "<distance bin>:<site type>". This is the only stratum that brings in
            # information from outside the two score files, which is what lets the
            # analysis say where a disagreement sits relative to real splice sites.
            self._add_stratum(
                "site_distance",
                self.sites.stratum_key(chrom, pos),
                left_values[-1],
                right_values[-1],
            )

    def _add_variant_pair(self, left: Annotation, right: Annotation) -> None:
        left_values = tuple(left.scores) + (max(left.scores),)
        right_values = tuple(right.scores) + (max(right.scores),)
        for label, x, y in zip(SCORE_LABELS, left_values, right_values):
            _add_score_to_state(self.variant_view, self.config, label, x, y)
        self.variant_view["dominant"][_dominant(left)][_dominant(right)] += 1
        _add_dp_to_state(self.variant_view, self.config, left, right)

    def _add_score(self, label: str, left: float, right: float) -> None:
        self.moments[label].add(left, right, self.config.equivalence_tolerances)
        bins = self.config.score_bins
        left_bin = _bin_index(left, bins)
        right_bin = _bin_index(right, bins)
        self.score_hist[label]["left"][left_bin] += 1
        self.score_hist[label]["right"][right_bin] += 1
        self.joint_hist[label][left_bin * bins + right_bin] += 1
        diff_bin = min(2 * bins, max(0, int(round((right - left + 1.0) * bins))))
        self.diff_hist[label][diff_bin] += 1
        for threshold in self.config.thresholds:
            _add_table(self.thresholds[label][_number_key(threshold)], left, right, threshold)

        rounded_right = float(f"{right:.2f}")
        self.rounded_right_moments[label].add(left, rounded_right, self.config.equivalence_tolerances)
        for threshold in self.config.thresholds:
            _add_table(
                self.rounded_right_thresholds[label][_number_key(threshold)],
                left,
                rounded_right,
                threshold,
            )

        for key, signal_threshold in _signal_keys(self.config):
            include = max(left, right) > 0.0 if key == "either_gt_0" else max(left, right) >= signal_threshold
            if include:
                self.signal_subset_moments[label][key].add(left, right, self.config.equivalence_tolerances)

    def _add_stratum(self, dimension: str, key: str, left: float, right: float) -> None:
        stats = self.strata[dimension].setdefault(
            key,
            {
                "moments": Moments(),
                "thresholds": {_number_key(t): _empty_table() for t in self.config.thresholds},
            },
        )
        stats["moments"].add(left, right, self.config.equivalence_tolerances)
        for threshold in self.config.thresholds:
            table = stats["thresholds"][_number_key(threshold)]
            _add_table(table, left, right, threshold)

    def _consider_sample(self, token: str, payload: dict) -> None:
        if self.config.sample_size == 0:
            return
        rank = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        item = (-rank, token, payload)
        if len(self._sample_heap) < self.config.sample_size:
            heapq.heappush(self._sample_heap, item)
        elif item > self._sample_heap[0]:
            heapq.heapreplace(self._sample_heap, item)

    def _consider_top(self, token: str, payload: dict) -> None:
        if self.config.top_k == 0:
            return
        item = (float(payload["max_abs_difference"]), token, payload)
        if len(self._top_heap) < self.config.top_k:
            heapq.heappush(self._top_heap, item)
        elif item > self._top_heap[0]:
            heapq.heapreplace(self._top_heap, item)

    def merge(self, other: "Aggregate") -> None:
        if self.config != other.config:
            raise ValueError("cannot merge aggregates with different configurations")
        self.coverage.update(other.coverage)
        for label in SCORE_LABELS:
            self.moments[label].merge(other.moments[label])
            for side in ("left", "right"):
                self.score_hist[label][side] = [
                    a + b for a, b in zip(self.score_hist[label][side], other.score_hist[label][side])
                ]
            self.joint_hist[label] = [a + b for a, b in zip(self.joint_hist[label], other.joint_hist[label])]
            self.diff_hist[label] = [a + b for a, b in zip(self.diff_hist[label], other.diff_hist[label])]
            for threshold in self.thresholds[label]:
                for cell in self.thresholds[label][threshold]:
                    self.thresholds[label][threshold][cell] += other.thresholds[label][threshold][cell]
        for left in DOMINANT_LABELS:
            for right in DOMINANT_LABELS:
                self.dominant[left][right] += other.dominant[left][right]
        for event in EVENTS:
            for threshold, values in self.dp[event].items():
                incoming = other.dp[event][threshold]
                for metric_name in ("both_above", "eligible"):
                    values[metric_name] += incoming[metric_name]
                for tolerance in values["within"]:
                    values["within"][tolerance] += incoming["within"][tolerance]
        for label in SCORE_LABELS:
            self.rounded_right_moments[label].merge(other.rounded_right_moments[label])
            for threshold, table in self.rounded_right_thresholds[label].items():
                for cell in table:
                    table[cell] += other.rounded_right_thresholds[label][threshold][cell]
            for subset, moments in self.signal_subset_moments[label].items():
                moments.merge(other.signal_subset_moments[label][subset])
        _merge_score_state(self.variant_view, other.variant_view)
        self._merge_strata(other)
        for _rank, token, payload in other._sample_heap:
            self._consider_sample(token, payload)
        for _difference, token, payload in other._top_heap:
            self._consider_top(token, payload)

    def _merge_strata(self, other: "Aggregate") -> None:
        for dimension, groups in other.strata.items():
            target_groups = self.strata[dimension]
            for key, incoming in groups.items():
                if key not in target_groups:
                    target_groups[key] = incoming
                    continue
                target = target_groups[key]
                target["moments"].merge(incoming["moments"])
                for threshold, table in incoming["thresholds"].items():
                    for cell, value in table.items():
                        target["thresholds"][threshold][cell] += value

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "coverage": dict(self.coverage),
            "moments": {label: value.to_dict() for label, value in self.moments.items()},
            "score_hist": self.score_hist,
            "joint_hist": self.joint_hist,
            "diff_hist": self.diff_hist,
            "thresholds": self.thresholds,
            "dominant": self.dominant,
            "dp": self.dp,
            "rounded_right_2dp": {
                "moments": {label: value.to_dict() for label, value in self.rounded_right_moments.items()},
                "thresholds": self.rounded_right_thresholds,
            },
            "signal_subsets": {
                label: {subset: moments.to_dict() for subset, moments in subsets.items()}
                for label, subsets in self.signal_subset_moments.items()
            },
            "variant_collapsed_view": _score_state_to_dict(self.variant_view),
            "strata": {
                dimension: {
                    key: {
                        "moments": values["moments"].to_dict(),
                        "thresholds": values["thresholds"],
                    }
                    for key, values in groups.items()
                }
                for dimension, groups in self.strata.items()
            },
            "sample": [
                {"hash_rank": -rank, "token": token, **payload}
                for rank, token, payload in sorted(self._sample_heap, reverse=True)
            ],
            "top_discrepancies": [
                {"token": token, **payload} for _difference, token, payload in sorted(self._top_heap, reverse=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Aggregate":
        serialized_version = int(data.get("schema_version", 1))
        if serialized_version != cls.schema_version:
            raise ValueError(
                "aggregate schema version mismatch: "
                f"found {serialized_version}, require {cls.schema_version}; rerun mappers"
            )
        config = AnalysisConfig.from_dict(data["config"])
        result = cls(config)
        result.coverage = Counter({str(k): int(v) for k, v in data.get("coverage", {}).items()})
        result.moments = {label: Moments.from_dict(data.get("moments", {}).get(label, {})) for label in SCORE_LABELS}
        for attribute in ("score_hist", "joint_hist", "diff_hist", "thresholds", "dominant", "dp"):
            if attribute in data:
                setattr(result, attribute, data[attribute])
        rounded = data.get("rounded_right_2dp", {})
        result.rounded_right_moments = {
            label: Moments.from_dict(rounded.get("moments", {}).get(label, {})) for label in SCORE_LABELS
        }
        if "thresholds" in rounded:
            result.rounded_right_thresholds = rounded["thresholds"]
        subsets = data.get("signal_subsets", {})
        result.signal_subset_moments = {
            label: {
                key: Moments.from_dict(subsets.get(label, {}).get(key, {})) for key, _threshold in _signal_keys(config)
            }
            for label in SCORE_LABELS
        }
        if "variant_collapsed_view" in data:
            result.variant_view = _score_state_from_dict(config, data["variant_collapsed_view"])
        result.strata = {name: {} for name in config.strata}
        result.strata.update(
            {
                dimension: {
                    key: {
                        "moments": Moments.from_dict(values["moments"]),
                        "thresholds": values["thresholds"],
                    }
                    for key, values in groups.items()
                }
                for dimension, groups in data.get("strata", {}).items()
            }
        )
        result._sample_heap = []
        for item in data.get("sample", []):
            payload = {k: v for k, v in item.items() if k not in {"hash_rank", "token"}}
            result._consider_sample(str(item["token"]), payload)
        result._top_heap = []
        for item in data.get("top_discrepancies", []):
            payload = {k: v for k, v in item.items() if k != "token"}
            result._consider_top(str(item["token"]), payload)
        return result

    def derived(self) -> dict:
        scores = {}
        for label in SCORE_LABELS:
            scores[label] = {
                **self.moments[label].derived(),
                "distribution": derive_histogram_metrics(
                    self.score_hist[label]["left"],
                    self.score_hist[label]["right"],
                    self.joint_hist[label],
                ),
                "difference_distribution": derive_difference_histogram_metrics(
                    self.diff_hist[label], self.config.score_bins
                ),
            }
        rounded_scores = {label: self.rounded_right_moments[label].derived() for label in SCORE_LABELS}
        quantization_impact = {}
        for label in SCORE_LABELS:
            raw_metrics = scores[label]
            rounded_metrics = rounded_scores[label]
            quantization_impact[label] = {
                "raw_mae": raw_metrics.get("mae"),
                "right_rounded_2dp_mae": rounded_metrics.get("mae"),
                "mae_change_rounded_minus_raw": _optional_difference(
                    rounded_metrics.get("mae"), raw_metrics.get("mae")
                ),
                "raw_rmse": raw_metrics.get("rmse"),
                "right_rounded_2dp_rmse": rounded_metrics.get("rmse"),
                "raw_exact_match_rate": raw_metrics.get("exact_match_rate"),
                "right_rounded_2dp_exact_match_rate": rounded_metrics.get("exact_match_rate"),
                "exact_match_rate_gain": _optional_difference(
                    rounded_metrics.get("exact_match_rate"),
                    raw_metrics.get("exact_match_rate"),
                ),
            }
        main_state = {
            "moments": self.moments,
            "score_hist": self.score_hist,
            "joint_hist": self.joint_hist,
            "diff_hist": self.diff_hist,
            "thresholds": self.thresholds,
            "dominant": self.dominant,
            "dp": self.dp,
        }
        return {
            "coverage": dict(self.coverage),
            "estimand": {
                "primary": "exact normalized gene match within genomic allele",
                "variant_collapsed": (
                    "one genomic-allele observation after independent event-wise "
                    "maximum collapse across valid genes on each predictor"
                ),
                "right_rounded_2dp": ("primary pairs with only the right score rounded to two decimals"),
            },
            "scores": scores,
            "thresholds": {
                label: {threshold: derive_threshold_metrics(table) for threshold, table in values.items()}
                for label, values in self.thresholds.items()
            },
            "dominant": self.dominant,
            "dominant_normalized": derive_dominant_metrics(self.dominant),
            "dp": derive_dp_metrics(self.dp),
            "right_rounded_2dp": {
                "scores": rounded_scores,
                "thresholds": {
                    label: {threshold: derive_threshold_metrics(table) for threshold, table in tables.items()}
                    for label, tables in self.rounded_right_thresholds.items()
                },
                "impact_vs_raw": quantization_impact,
            },
            "signal_subsets": {
                label: {subset: moments.derived() for subset, moments in subsets.items()}
                for label, subsets in self.signal_subset_moments.items()
            },
            "variant_collapsed_view": derive_score_state(self.variant_view, self.config),
            "strata": {
                dimension: {
                    key: {
                        "metrics": values["moments"].derived(),
                        "thresholds": {
                            threshold: derive_threshold_metrics(table)
                            for threshold, table in values["thresholds"].items()
                        },
                    }
                    for key, values in groups.items()
                }
                for dimension, groups in self.strata.items()
            },
            "cluster_bootstrap_95ci": derive_cluster_bootstraps(main_state, self.strata, self.config),
            "approximation_metadata": {
                "histograms": (
                    f"{self.config.score_bins} equal-width bins on [0,1]; joint "
                    "histograms are additive across mapper shards"
                ),
                "ks_wasserstein_js": (
                    "computed from binned marginal distributions; Wasserstein uses "
                    "the discrete CDF integral and Jensen-Shannon uses base-2 logs"
                ),
                "spearman": (
                    "Pearson correlation of marginal midranks assigned to joint "
                    "histogram bins; this is binned approximate Spearman, not an "
                    "exact row-rank calculation"
                ),
                "bootstrap": (
                    "deterministic percentile cluster bootstrap from additive MAX "
                    "moments; it does not retain or resample individual rows"
                ),
            },
        }


def _optional_difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def derive_dp_metrics(dp: Mapping) -> dict:
    return {
        event: {
            threshold: {
                **{key: value for key, value in values.items() if key != "within"},
                "within": dict(values["within"]),
                "within_rates": {
                    tolerance: count / values["eligible"] if values["eligible"] else None
                    for tolerance, count in values["within"].items()
                },
            }
            for threshold, values in thresholds.items()
        }
        for event, thresholds in dp.items()
    }


def derive_dominant_metrics(matrix: Mapping) -> dict:
    labels = list(DOMINANT_LABELS)
    total = sum(int(matrix[left][right]) for left in labels for right in labels)
    diagonal = sum(int(matrix[label][label]) for label in labels)
    signal_total = sum(
        int(matrix[left][right]) for left in labels for right in labels if left != "NONE" or right != "NONE"
    )
    signal_diagonal = sum(int(matrix[label][label]) for label in labels if label != "NONE")
    row_normalized = {}
    for left in labels:
        row_total = sum(int(matrix[left][right]) for right in labels)
        row_normalized[left] = {right: int(matrix[left][right]) / row_total if row_total else None for right in labels}
    return {
        "labels": labels,
        "n": total,
        "overall_exact_rate": diagonal / total if total else None,
        "union_signal_n": signal_total,
        "union_signal_exact_rate": signal_diagonal / signal_total if signal_total else None,
        "row_normalized": row_normalized,
    }


def derive_histogram_metrics(left_hist: Sequence[int], right_hist: Sequence[int], joint_hist: Sequence[int]) -> dict:
    """Derive bounded-memory distribution and approximate rank statistics."""

    bins = len(left_hist)
    if bins < 2 or len(right_hist) != bins or len(joint_hist) != bins * bins:
        raise ValueError("invalid marginal/joint histogram dimensions")
    left_total = sum(int(value) for value in left_hist)
    right_total = sum(int(value) for value in right_hist)
    joint_total = sum(int(value) for value in joint_hist)
    if not left_total or not right_total or not joint_total:
        return {
            "n": joint_total,
            "ks_distance_binned": None,
            "wasserstein_1_binned": None,
            "jensen_shannon_divergence_base2": None,
            "spearman_r_binned": None,
            "score_bins": bins,
        }

    left_prob = [int(value) / left_total for value in left_hist]
    right_prob = [int(value) / right_total for value in right_hist]
    left_cdf = 0.0
    right_cdf = 0.0
    ks = 0.0
    wasserstein = 0.0
    for index, (left_value, right_value) in enumerate(zip(left_prob, right_prob)):
        left_cdf += left_value
        right_cdf += right_value
        difference = abs(left_cdf - right_cdf)
        ks = max(ks, difference)
        if index < bins - 1:
            wasserstein += difference / bins

    js = 0.0
    for left_value, right_value in zip(left_prob, right_prob):
        midpoint = (left_value + right_value) / 2.0
        if left_value:
            js += 0.5 * left_value * math.log2(left_value / midpoint)
        if right_value:
            js += 0.5 * right_value * math.log2(right_value / midpoint)

    # Assign each bin the average marginal rank of all observations tied in it.
    # Correlating those midranks through the joint histogram gives a deterministic
    # binned approximation to Spearman's rho.
    left_ranks = []
    cumulative = 0
    for count in left_hist:
        left_ranks.append(cumulative + (int(count) + 1) / 2.0)
        cumulative += int(count)
    right_ranks = []
    cumulative = 0
    for count in right_hist:
        right_ranks.append(cumulative + (int(count) + 1) / 2.0)
        cumulative += int(count)
    sum_left = sum_right = sum_left2 = sum_right2 = sum_cross = 0.0
    for left_index, left_rank in enumerate(left_ranks):
        offset = left_index * bins
        for right_index, right_rank in enumerate(right_ranks):
            count = int(joint_hist[offset + right_index])
            if not count:
                continue
            sum_left += count * left_rank
            sum_right += count * right_rank
            sum_left2 += count * left_rank * left_rank
            sum_right2 += count * right_rank * right_rank
            sum_cross += count * left_rank * right_rank
    n = float(joint_total)
    mean_left = sum_left / n
    mean_right = sum_right / n
    variance_left = max(0.0, sum_left2 / n - mean_left * mean_left)
    variance_right = max(0.0, sum_right2 / n - mean_right * mean_right)
    covariance = sum_cross / n - mean_left * mean_right
    denominator = math.sqrt(variance_left * variance_right)
    return {
        "n": joint_total,
        "ks_distance_binned": ks,
        "wasserstein_1_binned": wasserstein,
        "jensen_shannon_divergence_base2": js,
        "spearman_r_binned": covariance / denominator if denominator else None,
        "score_bins": bins,
    }


def _histogram_quantile(counts: Sequence[int], values: Sequence[float], q: float) -> Optional[float]:
    total = sum(int(value) for value in counts)
    if not total:
        return None
    target = q * (total - 1)
    cumulative = 0
    for count, value in zip(counts, values):
        cumulative += int(count)
        if cumulative > target:
            return value
    return values[-1]


def derive_difference_histogram_metrics(diff_hist: Sequence[int], bins: int) -> dict:
    if len(diff_hist) != 2 * bins + 1:
        raise ValueError("invalid difference histogram dimensions")
    values = [(index - bins) / bins for index in range(2 * bins + 1)]
    absolute_counts: Counter[float] = Counter()
    for count, value in zip(diff_hist, values):
        absolute_counts[abs(value)] += int(count)
    absolute_values = sorted(absolute_counts)
    absolute_hist = [absolute_counts[value] for value in absolute_values]
    return {
        "n": sum(int(value) for value in diff_hist),
        "p05_right_minus_left_binned": _histogram_quantile(diff_hist, values, 0.05),
        "median_right_minus_left_binned": _histogram_quantile(diff_hist, values, 0.5),
        "p95_right_minus_left_binned": _histogram_quantile(diff_hist, values, 0.95),
        "p90_abs_difference_binned": _histogram_quantile(absolute_hist, absolute_values, 0.9),
        "p99_abs_difference_binned": _histogram_quantile(absolute_hist, absolute_values, 0.99),
        "bin_width": 1.0 / bins,
    }


def derive_score_state(state: Mapping, config: AnalysisConfig) -> dict:
    return {
        "scores": {
            label: {
                **state["moments"][label].derived(),
                "distribution": derive_histogram_metrics(
                    state["score_hist"][label]["left"],
                    state["score_hist"][label]["right"],
                    state["joint_hist"][label],
                ),
                "difference_distribution": derive_difference_histogram_metrics(
                    state["diff_hist"][label], config.score_bins
                ),
            }
            for label in SCORE_LABELS
        },
        "thresholds": {
            label: {threshold: derive_threshold_metrics(table) for threshold, table in values.items()}
            for label, values in state["thresholds"].items()
        },
        "dominant": state["dominant"],
        "dominant_normalized": derive_dominant_metrics(state["dominant"]),
        "dp": derive_dp_metrics(state["dp"]),
    }


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def derive_cluster_bootstraps(
    state: Mapping,
    strata: Mapping[str, Mapping[str, Mapping]],
    config: AnalysisConfig,
) -> dict:
    """Deterministic gene/block percentile CIs from additive MAX moments."""

    result = {
        "method": "cluster bootstrap with replacement; percentile 95% interval",
        "score": "MAX",
        "replicates": config.bootstrap_replicates,
        "seed": config.bootstrap_seed,
        "dimensions": {},
    }
    metric_names = ("bias_right_minus_left", "pearson_r", "lin_ccc", "mae", "rmse")
    point = state["moments"]["MAX"].derived()
    for dimension in ("block_1mb", "gene"):
        clusters = [
            values["moments"] for _key, values in sorted(strata.get(dimension, {}).items()) if values["moments"].n
        ]
        dimension_result = {
            "cluster_count": len(clusters),
            "observation_count": sum(item.n for item in clusters),
            "metrics": {},
        }
        if not clusters or config.bootstrap_replicates == 0:
            for metric_name in metric_names:
                dimension_result["metrics"][metric_name] = {
                    "estimate": point.get(metric_name),
                    "lower": None,
                    "upper": None,
                }
            result["dimensions"][dimension] = dimension_result
            continue
        seed_material = f"{config.bootstrap_seed}|{dimension}".encode("utf-8")
        dimension_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        generator = random.Random(dimension_seed)
        replicate_values = {name: [] for name in metric_names}
        for _replicate in range(config.bootstrap_replicates):
            replicate = Moments()
            for _draw in range(len(clusters)):
                replicate.merge(clusters[generator.randrange(len(clusters))])
            metrics = replicate.derived()
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if value is not None and math.isfinite(value):
                    replicate_values[metric_name].append(value)
        for metric_name in metric_names:
            values = replicate_values[metric_name]
            dimension_result["metrics"][metric_name] = {
                "estimate": point.get(metric_name),
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "valid_replicates": len(values),
            }
        result["dimensions"][dimension] = dimension_result
    return result


def derive_threshold_metrics(table: Mapping[str, int]) -> dict:
    pp = int(table["both_positive"])
    left_only = int(table["left_only"])
    right_only = int(table["right_only"])
    nn = int(table["both_negative"])
    total = pp + left_only + right_only + nn
    union = pp + left_only + right_only
    left_positive = pp + left_only
    right_positive = pp + right_only
    observed = (pp + nn) / total if total else None
    expected = None
    kappa = None
    if total:
        expected = (left_positive * right_positive + (total - left_positive) * (total - right_positive)) / (
            total * total
        )
        if expected != 1.0:
            kappa = (observed - expected) / (1.0 - expected)
    mcc_denominator = math.sqrt(left_positive * (total - left_positive) * right_positive * (total - right_positive))
    return {
        **{key: int(value) for key, value in table.items()},
        "n": total,
        "overall_agreement": observed,
        "positive_agreement": 2 * pp / (2 * pp + left_only + right_only) if 2 * pp + left_only + right_only else None,
        "negative_agreement": 2 * nn / (2 * nn + left_only + right_only) if 2 * nn + left_only + right_only else None,
        "jaccard": pp / union if union else None,
        "dice": 2 * pp / (2 * pp + left_only + right_only) if 2 * pp + left_only + right_only else None,
        "kappa": kappa,
        "mcc": (pp * nn - left_only * right_only) / mcc_denominator if mcc_denominator else None,
        "left_positive_rate": left_positive / total if total else None,
        "right_positive_rate": right_positive / total if total else None,
        "call_rate_ratio_right_over_left": right_positive / left_positive if left_positive else None,
    }
