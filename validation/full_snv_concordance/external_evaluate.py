"""Functional-accuracy evaluation for harmonized experimental variants.

This module is intentionally independent from the full-genome concordance
reducers.  Predictor-to-predictor agreement and agreement with an experimental
binary outcome answer different questions and must not share denominators.

The implementation has no required numerical dependencies.  It performs an
exact ``(CHROM, POS, REF, ALT, gene)`` join, excludes conflicting predictions,
and reports every missingness state before calculating discrimination and
threshold metrics.  Matplotlib is used only for optional PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import itertools
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

from .vcf import Annotation, VariantKey, inspect_header, iter_vcf_records, parse_annotations
from .workflows import utc_now


DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.1, 0.2, 0.5, 0.8)
SUPPORTED_INFO_KEYS: Tuple[str, ...] = ("SpliceAI", "OpenSpliceAI")
HARMONIZED_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "source_row",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "label",
    "cohort",
)


@dataclass(frozen=True)
class FunctionalObservation:
    observation_id: str
    input_row: int
    dataset: str
    source_row: str
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    transcript: str
    raw_label: str
    label: Optional[int]
    label_status: str
    outcome: str
    cohort: str

    @property
    def variant_key(self) -> VariantKey:
        return VariantKey(self.chrom, self.pos, self.ref, self.alt)

    @property
    def prediction_key(self) -> tuple[str, int, str, str, str]:
        return self.chrom, self.pos, self.ref, self.alt, self.gene


@dataclass
class PredictorIndex:
    name: str
    path: Path
    info_key: str
    values: MutableMapping[tuple[str, int, str, str, str], set[Annotation]]
    variants: set[VariantKey]
    records: int = 0
    identical_duplicates: int = 0
    invalid_annotations: int = 0
    allele_mismatches: int = 0
    records_without_selected_info: int = 0
    parsed_file_sha256: Optional[str] = None
    parsed_canonical_digest: Optional[str] = None
    parsed_stat: Optional[dict] = None

    @property
    def conflicts(self) -> int:
        return sum(len(values) > 1 for values in self.values.values())

    @property
    def unique_gene_predictions(self) -> int:
        return sum(len(values) == 1 for values in self.values.values())

    def lookup(self, observation: FunctionalObservation) -> tuple[str, Optional[float]]:
        values = self.values.get(observation.prediction_key)
        if values:
            if len(values) > 1:
                return "prediction_conflict", None
            annotation = next(iter(values))
            # Functional benchmarks conventionally evaluate the maximum of the
            # four delta scores, not a selected event type.
            return "matched", max(annotation.scores)
        if observation.variant_key in self.variants:
            return "missing_exact_gene", None
        return "missing_variant", None

    def provenance(self) -> dict:
        if self.parsed_file_sha256 is None:
            raise ValueError(f"{self.name}: predictor byte identity was not captured while parsing")
        return {
            "name": self.name,
            "path": str(self.path.resolve()),
            "sha256": self.parsed_file_sha256,
            "canonical_digest": self.parsed_canonical_digest,
            "parsed_stat": self.parsed_stat,
            "info_key": self.info_key,
            "records": self.records,
            "unique_variants": len(self.variants),
            "unique_gene_predictions": self.unique_gene_predictions,
            "conflicting_gene_predictions": self.conflicts,
            "identical_duplicate_annotations": self.identical_duplicates,
            "invalid_annotations": self.invalid_annotations,
            "allele_mismatches": self.allele_mismatches,
            "records_without_selected_info": self.records_without_selected_info,
        }


@dataclass(frozen=True)
class JoinedPrediction:
    observation: FunctionalObservation
    predictor: str
    status: str
    score: Optional[float]


def normalize_functional_label(dataset: str, value: object) -> tuple[Optional[int], str]:
    """Normalize only the publication-specific labels declared by this study.

    Broad truthiness rules are deliberately avoided: for example, treating any
    nonempty string as positive would turn ``"False"`` into a positive label.
    """

    identifier = dataset.strip().lower()
    label = str(value).strip()
    folded = label.casefold()
    if identifier.startswith("smith"):
        mapping = {"sdv": 1, "neutral": 0}
        policy = "smith_sdv_vs_neutral"
    elif identifier.startswith("riepe"):
        mapping = {"splice_altering": 1, "neutral": 0}
        policy = "riepe_gt20_percent_mutant_rna"
    elif identifier.startswith("spip"):
        mapping = {"1": 1, "0": 0}
        policy = "spip_experimental_binary_1_vs_0"
    else:
        return None, "unsupported_dataset"
    if folded not in mapping:
        return None, f"invalid_label_for_{policy}"
    return mapping[folded], policy


def read_harmonized(
    path: str | Path, expected_sha256: Optional[str] = None
) -> tuple[list[FunctionalObservation], dict]:
    path = Path(path)
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: harmonized bytes differ from frozen provenance: "
            f"{observed_sha256} != {expected_sha256}"
        )
    observations: list[FunctionalObservation] = []
    identifiers: set[str] = set()
    label_counts: Dict[str, int] = {}
    with io.StringIO(raw.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(HARMONIZED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        for input_row, row in enumerate(reader, 2):
            if row.get("reason", "accepted") not in ("", "accepted"):
                raise ValueError(
                    f"{path}:{input_row}: harmonized input contains non-accepted row"
                )
            try:
                pos = int(str(row["pos"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{input_row}: invalid POS {row.get('pos')!r}") from exc
            if pos < 1:
                raise ValueError(f"{path}:{input_row}: POS must be one-based")
            dataset = str(row["dataset"]).strip()
            source_row = str(row["source_row"]).strip()
            observation_id = f"{dataset}|{source_row}"
            if observation_id in identifiers:
                raise ValueError(f"{path}:{input_row}: duplicate observation ID {observation_id}")
            identifiers.add(observation_id)
            ref, alt = str(row["ref"]).strip(), str(row["alt"]).strip()
            gene = str(row["gene"]).strip()
            if not dataset or not source_row or not gene:
                raise ValueError(f"{path}:{input_row}: dataset/source_row/gene must be nonempty")
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                raise ValueError(f"{path}:{input_row}: expected uppercase biallelic SNV")
            label, label_status = normalize_functional_label(dataset, row["label"])
            label_counts[label_status] = label_counts.get(label_status, 0) + 1
            observations.append(
                FunctionalObservation(
                    observation_id=observation_id,
                    input_row=input_row,
                    dataset=dataset,
                    source_row=source_row,
                    chrom=str(row["chrom"]).strip(),
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    gene=gene,
                    transcript=str(row.get("transcript", "")),
                    raw_label=str(row["label"]),
                    label=label,
                    label_status=label_status,
                    outcome=str(row.get("outcome", "")),
                    cohort=str(row["cohort"]).strip() or dataset,
                )
            )
    return observations, {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "rows": len(observations),
        "valid_binary_labels": sum(item.label is not None for item in observations),
        "invalid_or_unsupported_labels": sum(item.label is None for item in observations),
        "label_policy_counts": dict(sorted(label_counts.items())),
    }


def _choose_info_key(name: str, path: Path, explicit: Optional[str]) -> str:
    header = inspect_header(path)
    declared = set(header["info_ids"]) & set(SUPPORTED_INFO_KEYS)
    if explicit is not None:
        if explicit not in SUPPORTED_INFO_KEYS:
            raise ValueError(
                f"{name}: unsupported INFO key {explicit!r}; choose from {SUPPORTED_INFO_KEYS}"
            )
        if explicit not in declared:
            raise ValueError(f"{name}: {path} does not declare INFO/{explicit}")
        return explicit
    if len(declared) == 1:
        return next(iter(declared))
    if not declared:
        raise ValueError(f"{name}: {path} declares neither SpliceAI nor OpenSpliceAI INFO")
    folded = name.casefold()
    if "opensplice" in folded or "osai" in folded or folded.startswith("rs"):
        return "OpenSpliceAI"
    if folded == "spliceai" or folded.startswith("spliceai_") or folded.startswith("spliceai-"):
        return "SpliceAI"
    raise ValueError(
        f"{name}: {path} declares both supported INFO fields; provide an explicit info key"
    )


def load_predictor(
    name: str,
    path: str | Path,
    info_key: Optional[str] = None,
    *,
    expected_sha256: Optional[str] = None,
    expected_records: Optional[int] = None,
) -> PredictorIndex:
    if not name or any(character in name for character in "\t\r\n"):
        raise ValueError("predictor name must be nonempty and single-line")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = _choose_info_key(name, path, info_key)
    index = PredictorIndex(name, path, selected, {}, set())
    parsed_identity: dict[str, object] = {}
    before = path.stat()
    for record in iter_vcf_records(path, verification_result=parsed_identity):
        index.records += 1
        index.variants.add(record.key)
        if selected not in record.info or record.info.get(selected) in (None, "", "."):
            index.records_without_selected_info += 1
        annotations, invalid = parse_annotations(record.info.get(selected))
        index.invalid_annotations += invalid
        for annotation in annotations:
            if annotation.allele != record.key.alt:
                index.allele_mismatches += 1
                continue
            key = (
                record.key.chrom,
                record.key.pos,
                record.key.ref,
                record.key.alt,
                annotation.gene,
            )
            values = index.values.setdefault(key, set())
            if annotation in values:
                index.identical_duplicates += 1
            values.add(annotation)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"{name}: score snapshot changed while it was being parsed")
    observed_sha256 = str(parsed_identity["file_sha256"])
    observed_records = int(parsed_identity["records"])
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"{name}: parsed score bytes differ from receipt: "
            f"{observed_sha256} != {expected_sha256}"
        )
    if expected_records is not None and observed_records != expected_records:
        raise ValueError(
            f"{name}: parsed score record count differs from receipt: "
            f"{observed_records} != {expected_records}"
        )
    index.parsed_file_sha256 = observed_sha256
    index.parsed_canonical_digest = str(parsed_identity["canonical_digest"])
    index.parsed_stat = {
        "device": after.st_dev,
        "inode": after.st_ino,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }
    return index


def join_predictions(
    observations: Sequence[FunctionalObservation], predictors: Sequence[PredictorIndex]
) -> tuple[dict[str, list[JoinedPrediction]], dict]:
    joined: dict[str, list[JoinedPrediction]] = {}
    coverage: dict[str, dict] = {}
    for predictor in predictors:
        rows: list[JoinedPrediction] = []
        counts: Dict[str, int] = {}
        by_dataset: Dict[str, Dict[str, int]] = {}
        by_cohort: Dict[str, Dict[str, int]] = {}
        for observation in observations:
            status, score = predictor.lookup(observation)
            rows.append(JoinedPrediction(observation, predictor.name, status, score))
            counts[status] = counts.get(status, 0) + 1
            for target, key in (
                (by_dataset, observation.dataset),
                (by_cohort, f"{observation.dataset}|{observation.cohort}"),
            ):
                bucket = target.setdefault(key, {})
                bucket[status] = bucket.get(status, 0) + 1
        joined[predictor.name] = rows
        coverage[predictor.name] = {
            "all_harmonized_rows": len(observations),
            "labelled_rows": sum(item.label is not None for item in observations),
            "join_status": dict(sorted(counts.items())),
            "by_dataset": {key: dict(sorted(value.items())) for key, value in sorted(by_dataset.items())},
            "by_dataset_cohort": {
                key: dict(sorted(value.items())) for key, value in sorted(by_cohort.items())
            },
        }
    return joined, coverage


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    favorable = 0.0
    negatives_below = 0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        group = ordered[cursor:end]
        group_positive = sum(label for _, label in group)
        group_negative = len(group) - group_positive
        favorable += group_positive * (negatives_below + 0.5 * group_negative)
        negatives_below += group_negative
        cursor = end
    return favorable / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = false_positive = 0
    result = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        group_positive = sum(label for _, label in ordered[cursor:end])
        group_negative = end - cursor - group_positive
        true_positive += group_positive
        false_positive += group_negative
        result += (group_positive / positives) * (
            true_positive / (true_positive + false_positive)
        )
        cursor = end
    return result


def _confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if label and predicted:
            tp += 1
        elif not label and predicted:
            fp += 1
        elif not label and not predicted:
            tn += 1
        else:
            fn += 1
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_ratio(tp * tn - fp * fn, denominator)
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced,
    }


def _curve_points(labels: Sequence[int], scores: Sequence[float]) -> tuple[list[dict], list[dict]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    roc = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]
    pr = [{"threshold": None, "recall": 0.0, "precision": 1.0}]
    tp = fp = 0
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == threshold:
            end += 1
        group = ordered[cursor:end]
        tp += sum(label for _, label in group)
        fp += len(group) - sum(label for _, label in group)
        roc.append(
            {
                "threshold": threshold,
                "fpr": _safe_ratio(fp, negatives),
                "tpr": _safe_ratio(tp, positives),
            }
        )
        pr.append(
            {
                "threshold": threshold,
                "recall": _safe_ratio(tp, positives),
                "precision": _safe_ratio(tp, tp + fp),
            }
        )
        cursor = end
    return roc, pr


def _point_metrics(
    labels: Sequence[int], scores: Sequence[float], thresholds: Sequence[float]
) -> dict:
    if len(labels) != len(scores):
        raise ValueError("labels and scores differ in length")
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in scores):
        raise ValueError("scores must be finite and within [0, 1]")
    brier = (
        sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)
        if labels
        else None
    )
    return {
        "auroc": _auroc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "brier": brier,
        "thresholds": {
            _threshold_token(threshold): _confusion(labels, scores, threshold)
            for threshold in thresholds
        },
    }


def _threshold_token(value: float) -> str:
    return f"{value:.12g}"


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _seed_for(seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _stratified_resamples(
    labels: Sequence[int], strata: Sequence[str], replicates: int, seed: int
) -> Iterator[list[int]]:
    if len(labels) != len(strata):
        raise ValueError("labels and bootstrap strata differ in length")
    buckets: Dict[tuple[str, int], list[int]] = {}
    for index, (label, stratum) in enumerate(zip(labels, strata)):
        buckets.setdefault((stratum, label), []).append(index)
    randomizer = random.Random(seed)
    ordered_buckets = [buckets[key] for key in sorted(buckets)]
    for _ in range(replicates):
        sample: list[int] = []
        for bucket in ordered_buckets:
            sample.extend(bucket[randomizer.randrange(len(bucket))] for _ in bucket)
        yield sample


_CI_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "balanced_accuracy",
)


def _with_bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    strata: Sequence[str],
    thresholds: Sequence[float],
    replicates: int,
    seed: int,
) -> dict:
    point = _point_metrics(labels, scores, thresholds)
    samples: dict[str, list[float]] = {
        "auroc": [],
        "average_precision": [],
        "brier": [],
    }
    threshold_samples: dict[str, dict[str, list[float]]] = {
        token: {metric: [] for metric in _CI_METRICS}
        for token in point["thresholds"]
    }
    for indices in _stratified_resamples(labels, strata, replicates, seed):
        sampled_labels = [labels[index] for index in indices]
        sampled_scores = [scores[index] for index in indices]
        metrics = _point_metrics(sampled_labels, sampled_scores, thresholds)
        for metric in samples:
            value = metrics[metric]
            if value is not None:
                samples[metric].append(value)
        for token, values in metrics["thresholds"].items():
            for metric in _CI_METRICS:
                value = values[metric]
                if value is not None:
                    threshold_samples[token][metric].append(value)

    result: dict = {}
    for metric in ("auroc", "average_precision", "brier"):
        values = samples[metric]
        result[metric] = {
            "estimate": point[metric],
            "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
            "bootstrap_valid": len(values),
            "bootstrap_replicates": replicates,
        }
    result["thresholds"] = {}
    for token, raw in point["thresholds"].items():
        decorated = {metric: raw[metric] for metric in ("tp", "fp", "tn", "fn")}
        for metric in _CI_METRICS:
            values = threshold_samples[token][metric]
            decorated[metric] = {
                "estimate": raw[metric],
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "bootstrap_valid": len(values),
                "bootstrap_replicates": replicates,
            }
        result["thresholds"][token] = decorated
    roc, pr = _curve_points(labels, scores)
    result["roc_curve"] = roc
    result["precision_recall_curve"] = pr
    return result


def _group_definitions(observations: Sequence[FunctionalObservation]) -> list[dict]:
    definitions: list[dict] = []
    cohorts = sorted({(item.dataset, item.cohort) for item in observations})
    for dataset, cohort in cohorts:
        definitions.append(
            {
                "scope": "dataset_cohort",
                "dataset": dataset,
                "cohort": cohort,
                "members": {
                    item.observation_id
                    for item in observations
                    if item.dataset == dataset and item.cohort == cohort
                },
            }
        )
    for dataset in sorted({item.dataset for item in observations}):
        definitions.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "cohort": None,
                "members": {
                    item.observation_id for item in observations if item.dataset == dataset
                },
            }
        )
    definitions.append(
        {
            "scope": "pooled",
            "dataset": None,
            "cohort": None,
            "members": {item.observation_id for item in observations},
        }
    )
    # Pooled observation rows can repeat the same biological variant+gene in
    # more than one source dataset.  Keep the observation-level pool for
    # transparency, and add a one-key/one-label sensitivity estimand.  Keys
    # with contradictory normalized labels are excluded rather than resolved
    # by an arbitrary dataset priority.
    by_key: Dict[tuple, list[FunctionalObservation]] = {}
    for item in observations:
        if item.label is not None:
            by_key.setdefault(_functional_key(item), []).append(item)
    consensus_members = set()
    for items in by_key.values():
        if len({int(item.label) for item in items}) == 1:
            consensus_members.add(min(item.observation_id for item in items))
    definitions.append(
        {
            "scope": "pooled_unique_consensus",
            "dataset": None,
            "cohort": None,
            "members": consensus_members,
        }
    )
    return definitions


def _functional_key(observation: FunctionalObservation) -> tuple:
    return (
        observation.chrom,
        observation.pos,
        observation.ref,
        observation.alt,
        observation.gene,
    )


def overlap_accounting(observations: Sequence[FunctionalObservation]) -> dict:
    by_key: Dict[tuple, list[FunctionalObservation]] = {}
    for item in observations:
        by_key.setdefault(_functional_key(item), []).append(item)
    repeated = {key: items for key, items in by_key.items() if len(items) > 1}
    discordant = {
        key: items
        for key, items in by_key.items()
        if len({item.label for item in items if item.label is not None}) > 1
    }
    consensus_unique = sum(
        bool(items)
        and len({item.label for item in items if item.label is not None}) == 1
        and next((item.label for item in items if item.label is not None), None) is not None
        for items in by_key.values()
    )
    return {
        "observation_rows": len(observations),
        "unique_variant_gene_keys": len(by_key),
        "repeated_variant_gene_keys": len(repeated),
        "duplicate_observation_rows_beyond_first": sum(len(items) - 1 for items in repeated.values()),
        "discordant_binary_label_keys": len(discordant),
        "pooled_unique_consensus_keys": consensus_unique,
        "policy": (
            "Observation-level pooled metrics retain all source rows. The "
            "pooled_unique_consensus sensitivity view retains one deterministic "
            "observation per variant+gene key only when all valid binary labels agree; "
            "discordant-label keys are excluded."
        ),
    }


def _bootstrap_stratum(scope: str, observation: FunctionalObservation) -> str:
    if scope == "pooled_unique_consensus":
        return "unique_variant_gene"
    if scope == "pooled":
        return f"{observation.dataset}|{observation.cohort}"
    if scope == "dataset":
        return observation.cohort
    return "within_cohort"


def calculate_group_metrics(
    observations: Sequence[FunctionalObservation],
    joined: Mapping[str, Sequence[JoinedPrediction]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20240801,
) -> list[dict]:
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("thresholds must be nonempty and in [0, 1]")
    groups: list[dict] = []
    for definition in _group_definitions(observations):
        members = definition["members"]
        for predictor, prediction_rows in joined.items():
            relevant = [row for row in prediction_rows if row.observation.observation_id in members]
            eligible = [
                row
                for row in relevant
                if row.status == "matched" and row.observation.label is not None
            ]
            labels = [int(row.observation.label) for row in eligible]
            scores = [float(row.score) for row in eligible]
            strata = [
                _bootstrap_stratum(definition["scope"], row.observation) for row in eligible
            ]
            status_counts: Dict[str, int] = {}
            for row in relevant:
                status_counts[row.status] = status_counts.get(row.status, 0) + 1
            metrics = _with_bootstrap_ci(
                labels,
                scores,
                strata,
                thresholds,
                bootstrap_replicates,
                _seed_for(
                    bootstrap_seed,
                    f"single|{definition['scope']}|{definition['dataset']}|"
                    f"{definition['cohort']}|{predictor}",
                ),
            )
            groups.append(
                {
                    "scope": definition["scope"],
                    "dataset": definition["dataset"],
                    "cohort": definition["cohort"],
                    "predictor": predictor,
                    "n_harmonized": len(relevant),
                    "n_valid_label": sum(
                        row.observation.label is not None for row in relevant
                    ),
                    "n_scored": len(eligible),
                    "n_positive": sum(labels),
                    "n_negative": len(labels) - sum(labels),
                    "join_status": dict(sorted(status_counts.items())),
                    "heterogeneous_pool_warning": (
                        "Pooled performance combines assays, genes, and ascertainment schemes; "
                        "interpret cohort estimates first."
                        if definition["scope"] in {"pooled", "pooled_unique_consensus"}
                        else None
                    ),
                    "metrics": metrics,
                }
            )
    return groups


def calculate_paired_delta_aurocs(
    observations: Sequence[FunctionalObservation],
    joined: Mapping[str, Sequence[JoinedPrediction]],
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20240801,
) -> list[dict]:
    predictor_names = list(joined)
    score_maps = {
        predictor: {
            row.observation.observation_id: row
            for row in rows
            if row.status == "matched" and row.observation.label is not None
        }
        for predictor, rows in joined.items()
    }
    observation_index = {item.observation_id: item for item in observations}
    results: list[dict] = []
    for definition in _group_definitions(observations):
        for left, right in itertools.combinations(predictor_names, 2):
            overlap_ids = sorted(
                definition["members"] & score_maps[left].keys() & score_maps[right].keys()
            )
            overlap = [observation_index[identifier] for identifier in overlap_ids]
            labels = [int(item.label) for item in overlap]
            left_scores = [float(score_maps[left][item.observation_id].score) for item in overlap]
            right_scores = [float(score_maps[right][item.observation_id].score) for item in overlap]
            left_auroc = _auroc(labels, left_scores)
            right_auroc = _auroc(labels, right_scores)
            estimate = (
                left_auroc - right_auroc
                if left_auroc is not None and right_auroc is not None
                else None
            )
            bootstrap_values: list[float] = []
            if estimate is not None:
                strata = [_bootstrap_stratum(definition["scope"], item) for item in overlap]
                for indices in _stratified_resamples(
                    labels,
                    strata,
                    bootstrap_replicates,
                    _seed_for(
                        bootstrap_seed,
                        f"paired|{definition['scope']}|{definition['dataset']}|"
                        f"{definition['cohort']}|{left}|{right}",
                    ),
                ):
                    sampled_labels = [labels[index] for index in indices]
                    sampled_left = [left_scores[index] for index in indices]
                    sampled_right = [right_scores[index] for index in indices]
                    left_value = _auroc(sampled_labels, sampled_left)
                    right_value = _auroc(sampled_labels, sampled_right)
                    if left_value is not None and right_value is not None:
                        bootstrap_values.append(left_value - right_value)
            results.append(
                {
                    "scope": definition["scope"],
                    "dataset": definition["dataset"],
                    "cohort": definition["cohort"],
                    "left_predictor": left,
                    "right_predictor": right,
                    "contrast": f"AUROC({left}) - AUROC({right})",
                    "n_overlap": len(overlap),
                    "n_positive": sum(labels),
                    "n_negative": len(labels) - sum(labels),
                    "left_auroc": left_auroc,
                    "right_auroc": right_auroc,
                    "delta_auroc": estimate,
                    "ci95": [
                        _percentile(bootstrap_values, 0.025),
                        _percentile(bootstrap_values, 0.975),
                    ],
                    "bootstrap_valid": len(bootstrap_values),
                    "bootstrap_replicates": bootstrap_replicates,
                    "reason": None if estimate is not None else "overlap_lacks_both_classes",
                }
            )
    return results


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _validation_context(
    path: Optional[str | Path], expected_sha256: Optional[str] = None
) -> dict:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("validation plan SHA-256 was supplied without a validation plan")
        return {
            "state": "not_supplied",
            "plan": None,
            "plan_sha256": None,
            "dataset_completeness": None,
            "sources": [],
        }
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and plan_sha256 != expected_sha256:
        raise ValueError(
            f"{plan_path}: validation plan bytes differ from frozen provenance: "
            f"{plan_sha256} != {expected_sha256}"
        )
    plan = json.loads(raw)
    if plan.get("kind") != "external-validation-preparation":
        raise ValueError(f"not an external validation plan: {plan_path}")
    completeness = plan.get("dataset_completeness")
    sources = plan.get("sources")
    if not isinstance(completeness, Mapping) or not isinstance(sources, list):
        raise ValueError(f"{plan_path}: missing dataset completeness/source provenance")
    expected = completeness.get("configured_dataset_ids")
    observed = [source.get("id") for source in sources if isinstance(source, Mapping)]
    if not isinstance(expected, list) or observed != expected:
        raise ValueError(f"{plan_path}: dataset source statuses do not match configured IDs")
    return {
        "state": "complete" if completeness.get("complete") else completeness.get("mode"),
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "dataset_completeness": dict(completeness),
        "sources": sources,
    }


def _run_provenance_context(
    path: Optional[str | Path], expected_sha256: Optional[str] = None
) -> dict:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("run-provenance SHA-256 was supplied without run provenance")
        return {"state": "not_supplied", "path": None, "sha256": None}
    provenance_path = Path(path)
    raw = provenance_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"{provenance_path}: run-provenance bytes differ from frozen identity: "
            f"{observed_sha256} != {expected_sha256}"
        )
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("kind") != "external-validation-evaluation-inputs"
        or not isinstance(document.get("receipts"), list)
        or len(document["receipts"]) != 7
    ):
        raise ValueError(f"invalid frozen run provenance: {provenance_path}")
    receipt_manifest = json.dumps(
        document["receipts"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_sha256 = document.get("receipt_manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
        or hashlib.sha256(receipt_manifest).hexdigest() != manifest_sha256
    ):
        raise ValueError("invalid frozen receipt-manifest identity")
    for key in ("bundle_sha256", "tasks_sha256"):
        digest = document.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid {key} in frozen run provenance")
    return {
        "state": "frozen_run_bound",
        "path": str(provenance_path.resolve()),
        "sha256": observed_sha256,
        **document,
    }


def _enforce_prediction_coverage(groups: Sequence[Mapping], minimum: float) -> dict:
    if not math.isfinite(minimum) or minimum < 0.0 or minimum > 1.0:
        raise ValueError("minimum prediction coverage must be finite and in [0, 1]")
    checked = []
    violations = []
    for group in groups:
        if group["scope"] not in {"pooled", "dataset", "dataset_cohort"}:
            continue
        denominator = int(group["n_valid_label"])
        numerator = int(group["n_scored"])
        fraction = numerator / denominator if denominator else None
        row = {
            "scope": group["scope"],
            "dataset": group["dataset"],
            "cohort": group["cohort"],
            "predictor": group["predictor"],
            "n_scored": numerator,
            "n_valid_label": denominator,
            "fraction": fraction,
            "reason": "no_valid_binary_labels" if denominator == 0 else None,
        }
        checked.append(row)
        if denominator == 0 or fraction < minimum:
            violations.append(row)
    policy = {
        "minimum_fraction": minimum,
        "scope": (
            "each predictor over pooled valid labels and separately within every dataset "
            "and cohort; a zero-valid-label stratum fails"
        ),
        "checked": checked,
        "violations": violations,
    }
    if violations:
        details = []
        for row in violations:
            identity = (
                f"{row['predictor']}:{row['scope']}:"
                f"{row['dataset'] or 'all'}:{row['cohort'] or 'all'}"
            )
            if row["fraction"] is None:
                details.append(f"{identity}=no valid binary labels")
            else:
                details.append(
                    f"{identity}={row['n_scored']}/{row['n_valid_label']} "
                    f"({row['fraction']:.3f})"
                )
        detail = ", ".join(details)
        raise ValueError(
            f"prediction coverage is below the required {minimum:.3f}: {detail}"
        )
    return policy


def _prediction_tsv_rows(joined: Mapping[str, Sequence[JoinedPrediction]]) -> Iterator[dict]:
    for predictor, rows in joined.items():
        for item in rows:
            observation = item.observation
            yield {
                "observation_id": observation.observation_id,
                "dataset": observation.dataset,
                "cohort": observation.cohort,
                "source_row": observation.source_row,
                "chrom": observation.chrom,
                "pos": observation.pos,
                "ref": observation.ref,
                "alt": observation.alt,
                "gene": observation.gene,
                "transcript": observation.transcript,
                "raw_label": observation.raw_label,
                "binary_label": "" if observation.label is None else observation.label,
                "label_status": observation.label_status,
                "outcome": observation.outcome,
                "predictor": predictor,
                "join_status": item.status,
                "max_ds": "" if item.score is None else f"{item.score:.17g}",
            }


def _metric_tsv_rows(groups: Sequence[Mapping]) -> Iterator[dict]:
    for group in groups:
        base = {
            key: group.get(key)
            for key in (
                "scope",
                "dataset",
                "cohort",
                "predictor",
                "n_harmonized",
                "n_valid_label",
                "n_scored",
                "n_positive",
                "n_negative",
            )
        }
        for name in ("auroc", "average_precision", "brier"):
            metric = group["metrics"][name]
            yield {
                **base,
                "metric": name,
                "threshold": "",
                "estimate": metric["estimate"],
                "ci_low": metric["ci95"][0],
                "ci_high": metric["ci95"][1],
                "bootstrap_valid": metric["bootstrap_valid"],
            }
        for threshold, metrics in group["metrics"]["thresholds"].items():
            for name in ("tp", "fp", "tn", "fn"):
                yield {
                    **base,
                    "metric": name,
                    "threshold": threshold,
                    "estimate": metrics[name],
                    "ci_low": "",
                    "ci_high": "",
                    "bootstrap_valid": "",
                }
            for name in _CI_METRICS:
                metric = metrics[name]
                yield {
                    **base,
                    "metric": name,
                    "threshold": threshold,
                    "estimate": metric["estimate"],
                    "ci_low": metric["ci95"][0],
                    "ci_high": metric["ci95"][1],
                    "bootstrap_valid": metric["bootstrap_valid"],
                }


def _paired_tsv_rows(results: Sequence[Mapping]) -> Iterator[dict]:
    for result in results:
        yield {
            **result,
            "ci_low": result["ci95"][0],
            "ci_high": result["ci95"][1],
        }


def _format_ci(metric: Mapping) -> str:
    estimate = metric.get("estimate")
    low, high = metric.get("ci95", (None, None))
    if estimate is None:
        return "NA"
    if low is None or high is None:
        return f"{estimate:.3f}"
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def _render_markdown(payload: Mapping) -> str:
    overlap = payload["cross_dataset_overlap"]
    dataset_provenance = payload["dataset_provenance"]
    completeness = dataset_provenance.get("dataset_completeness") or {}
    coverage_policy = payload["coverage_policy"]
    run_provenance = payload["run_provenance"]
    lines = [
        "# External functional-accuracy evaluation",
        "",
        "This analysis uses exact variant-and-gene matches and the maximum of "
        "DS_AG, DS_AL, DS_DG, and DS_DL. Conflicting annotations are excluded.",
        f"Score masking provenance: **{payload['score_mask_interpretation']}**.",
        "",
        "> **Do not treat pooled performance as a single biological benchmark.** "
        "The pooled rows combine different assays, genes, and ascertainment schemes; "
        "cohort-specific results are primary.",
        "The `pooled_unique_consensus` rows are a secondary deduplicated sensitivity "
        "analysis: one observation per exact variant+gene key, excluding keys whose "
        "normalized labels disagree across sources.",
        "",
        "## Dataset finality",
        "",
        f"Dataset state: **{dataset_provenance['state']}**. "
        f"Configured datasets: {len(completeness.get('configured_dataset_ids', []))}; "
        f"incomplete datasets: {len(completeness.get('incomplete_datasets', []))}.",
        f"Scoring-run provenance: **{run_provenance['state']}**; "
        f"bundle SHA-256: `{run_provenance.get('bundle_sha256') or 'not supplied'}`; "
        f"receipt-manifest SHA-256: "
        f"`{run_provenance.get('sha256') or 'not supplied'}`.",
        "",
        "| Dataset | State | Source rows | Accepted | Rejected | Unprocessed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for source in dataset_provenance.get("sources", []):
        lines.append(
            f"| {source.get('id')} | {source.get('state')} | {source.get('rows', 0)} | "
            f"{source.get('accepted', 0)} | {source.get('rejected', 0)} | "
            f"{source.get('unprocessed', 0)} |"
        )
    lines.extend(
        [
        "",
        "## Cross-dataset overlap accounting",
        "",
        f"There are {overlap['observation_rows']:,} observation rows and "
        f"{overlap['unique_variant_gene_keys']:,} unique variant+gene keys. "
        f"{overlap['repeated_variant_gene_keys']:,} keys repeat across rows and "
        f"{overlap['discordant_binary_label_keys']:,} keys have discordant binary labels; "
        f"the consensus sensitivity view retains {overlap['pooled_unique_consensus_keys']:,} keys.",
        "",
        "## Coverage",
        "",
        f"Publication gate: matched prediction coverage must be at least "
        f"{coverage_policy['minimum_fraction']:.3f} globally and within every dataset "
        "and cohort; a stratum with no valid binary labels fails.",
        "",
        "| Predictor | Labelled rows | Matched | Missing variant | Missing exact gene | Conflict |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for predictor, coverage in payload["coverage"].items():
        states = coverage["join_status"]
        lines.append(
            f"| {predictor} | {coverage['labelled_rows']} | {states.get('matched', 0)} | "
            f"{states.get('missing_variant', 0)} | {states.get('missing_exact_gene', 0)} | "
            f"{states.get('prediction_conflict', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Discrimination and calibration",
            "",
            "Values are estimates with deterministic stratified-bootstrap 95% intervals. "
            "Brier score is descriptive because calibration depends on cohort prevalence.",
            "",
            "| Scope | Dataset | Cohort | Predictor | N | Pos | AUROC | Average precision | Brier |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in payload["groups"]:
        lines.append(
            "| {scope} | {dataset} | {cohort} | {predictor} | {n_scored} | "
            "{n_positive} | {auroc} | {ap} | {brier} |".format(
                scope=group["scope"],
                dataset=group["dataset"] or "all",
                cohort=group["cohort"] or "all",
                predictor=group["predictor"],
                n_scored=group["n_scored"],
                n_positive=group["n_positive"],
                auroc=_format_ci(group["metrics"]["auroc"]),
                ap=_format_ci(group["metrics"]["average_precision"]),
                brier=_format_ci(group["metrics"]["brier"]),
            )
        )
    if payload["paired_delta_auroc"]:
        lines.extend(
            [
                "",
                "## Paired predictor contrasts",
                "",
                "Delta AUROC is calculated on the exact shared set of scored observations.",
                "",
                "| Scope | Dataset | Cohort | Contrast | N overlap | Delta AUROC (95% CI) |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for result in payload["paired_delta_auroc"]:
            low, high = result["ci95"]
            if result["delta_auroc"] is None:
                value = "NA"
            elif low is None or high is None:
                value = f"{result['delta_auroc']:.3f}"
            else:
                value = f"{result['delta_auroc']:.3f} [{low:.3f}, {high:.3f}]"
            lines.append(
                f"| {result['scope']} | {result['dataset'] or 'all'} | "
                f"{result['cohort'] or 'all'} | {result['contrast']} | "
                f"{result['n_overlap']} | {value} |"
            )
    plots = payload.get("plots", [])
    if plots:
        lines.extend(["", "## Figures", ""])
        for plot in plots:
            lines.append(f"![{plot['description']}]({Path(plot['path']).name})")
            lines.append("")
    elif payload.get("plot_error"):
        lines.extend(["", f"Figures were not rendered: {payload['plot_error']}", ""])
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Missing predictions and prediction conflicts are not silently converted to negatives.",
            "- Threshold metrics use `max(DS) >= threshold` as the positive rule.",
            "- Average precision is prevalence-sensitive; compare it together with each cohort's class balance.",
            "- Confidence intervals describe sampling variation under the specified stratified bootstrap, not assay-systematic uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _render_plots(groups: Sequence[Mapping], output_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots: list[dict] = []
    aggregate_rows = [row for row in groups if row["scope"] in {"dataset", "pooled"}]
    if aggregate_rows:
        labels = [
            f"{row['dataset'] or 'pooled'}\n{row['predictor']}" for row in aggregate_rows
        ]
        x = list(range(len(aggregate_rows)))
        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(x) * 0.7), 7), sharex=True)
        for axis, metric_name, title in (
            (axes[0], "auroc", "AUROC"),
            (axes[1], "average_precision", "Average precision / AUPRC"),
        ):
            estimates = [row["metrics"][metric_name]["estimate"] for row in aggregate_rows]
            estimates_numeric = [math.nan if value is None else value for value in estimates]
            lower = []
            upper = []
            for value, row in zip(estimates, aggregate_rows):
                low, high = row["metrics"][metric_name]["ci95"]
                lower.append(0 if value is None or low is None else value - low)
                upper.append(0 if value is None or high is None else high - value)
            axis.errorbar(x, estimates_numeric, yerr=[lower, upper], fmt="o", capsize=3)
            axis.set_ylim(0, 1.02)
            axis.set_ylabel(title)
            axis.grid(axis="y", alpha=0.25)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        fig.suptitle("External functional discrimination by dataset")
        path = output_dir / "discrimination.png"
        _atomic_figure(fig, path)
        plt.close(fig)
        plots.append({"path": str(path.resolve()), "description": "AUROC and average precision"})

    pooled = [row for row in groups if row["scope"] == "pooled"]
    if pooled:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for row in pooled:
            predictor = row["predictor"]
            roc = row["metrics"]["roc_curve"]
            pr = row["metrics"]["precision_recall_curve"]
            axes[0].plot(
                [point["fpr"] for point in roc if point["fpr"] is not None],
                [point["tpr"] for point in roc if point["tpr"] is not None],
                label=predictor,
            )
            axes[1].plot(
                [point["recall"] for point in pr if point["recall"] is not None],
                [point["precision"] for point in pr if point["recall"] is not None],
                label=predictor,
            )
        axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Pooled ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Pooled precision-recall")
        for axis in axes:
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.2)
            axis.legend()
        path = output_dir / "pooled_curves.png"
        _atomic_figure(fig, path)
        plt.close(fig)
        plots.append({"path": str(path.resolve()), "description": "Pooled ROC and precision-recall curves"})
    return plots


def evaluate_external(
    harmonized_tsv: str | Path,
    scored_vcfs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    info_keys: Optional[Mapping[str, str]] = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20240801,
    render_plots: bool = True,
    score_mask: Optional[int] = None,
    expected_score_sha256: Optional[Mapping[str, str]] = None,
    expected_score_records: Optional[Mapping[str, int]] = None,
    validation_plan: Optional[str | Path] = None,
    expected_validation_plan_sha256: Optional[str] = None,
    minimum_prediction_coverage: float = 0.9,
    expected_harmonized_sha256: Optional[str] = None,
    run_provenance: Optional[str | Path] = None,
    expected_run_provenance_sha256: Optional[str] = None,
) -> dict:
    """Evaluate one or more scored VCFs against harmonized functional labels."""

    if not scored_vcfs:
        raise ValueError("at least one scored VCF is required")
    if len(set(scored_vcfs)) != len(scored_vcfs):
        raise ValueError("predictor names must be unique")
    if score_mask not in (None, 0, 1):
        raise ValueError("score_mask must be 0, 1, or unspecified")
    observations, harmonized_provenance = read_harmonized(
        harmonized_tsv, expected_sha256=expected_harmonized_sha256
    )
    info_keys = info_keys or {}
    expected_score_sha256 = expected_score_sha256 or {}
    expected_score_records = expected_score_records or {}
    unknown_overrides = set(info_keys) - set(scored_vcfs)
    if unknown_overrides:
        raise ValueError(f"INFO-key override has no scored VCF: {sorted(unknown_overrides)}")
    for label, values in (
        ("score SHA-256", expected_score_sha256),
        ("score record count", expected_score_records),
    ):
        unknown = set(values) - set(scored_vcfs)
        if unknown:
            raise ValueError(f"{label} has no scored VCF: {sorted(unknown)}")
    predictors = [
        load_predictor(
            name,
            path,
            info_keys.get(name),
            expected_sha256=expected_score_sha256.get(name),
            expected_records=expected_score_records.get(name),
        )
        for name, path in scored_vcfs.items()
    ]
    joined, coverage = join_predictions(observations, predictors)
    groups = calculate_group_metrics(
        observations,
        joined,
        thresholds=tuple(thresholds),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    paired = calculate_paired_delta_aurocs(
        observations,
        joined,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    coverage_policy = _enforce_prediction_coverage(groups, minimum_prediction_coverage)
    dataset_context = _validation_context(
        validation_plan, expected_sha256=expected_validation_plan_sha256
    )
    run_context = _run_provenance_context(
        run_provenance, expected_sha256=expected_run_provenance_sha256
    )

    final_output_dir = Path(output_dir)
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_output_dir.exists():
        raise FileExistsError(f"external evaluation output already exists: {final_output_dir}")
    # Publish the entire result directory in one rename. A killed/failed
    # evaluator can leave only a uniquely named staging directory, so retrying
    # never mistakes a partial report for a completed evaluation.
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging.", dir=final_output_dir.parent
        )
    )
    predictions_path = output_dir / "joined_predictions.tsv"
    metrics_path = output_dir / "functional_metrics.tsv"
    paired_path = output_dir / "paired_delta_auroc.tsv"
    json_path = output_dir / "external_evaluation.json"
    report_path = output_dir / "external_evaluation.md"
    _atomic_tsv(
        predictions_path,
        (
            "observation_id",
            "dataset",
            "cohort",
            "source_row",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "transcript",
            "raw_label",
            "binary_label",
            "label_status",
            "outcome",
            "predictor",
            "join_status",
            "max_ds",
        ),
        _prediction_tsv_rows(joined),
    )
    metric_fields = (
        "scope",
        "dataset",
        "cohort",
        "predictor",
        "n_harmonized",
        "n_valid_label",
        "n_scored",
        "n_positive",
        "n_negative",
        "metric",
        "threshold",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_valid",
    )
    _atomic_tsv(metrics_path, metric_fields, _metric_tsv_rows(groups))
    paired_fields = (
        "scope",
        "dataset",
        "cohort",
        "left_predictor",
        "right_predictor",
        "contrast",
        "n_overlap",
        "n_positive",
        "n_negative",
        "left_auroc",
        "right_auroc",
        "delta_auroc",
        "ci_low",
        "ci_high",
        "bootstrap_valid",
        "bootstrap_replicates",
        "reason",
    )
    _atomic_tsv(paired_path, paired_fields, _paired_tsv_rows(paired))

    payload = {
        "schema_version": 3,
        "kind": "external-functional-accuracy",
        "created_at": utc_now(),
        "harmonized": harmonized_provenance,
        "dataset_provenance": dataset_context,
        "run_provenance": run_context,
        "predictors": [predictor.provenance() for predictor in predictors],
        "label_policies": {
            "smith": "SDV=1, Neutral=0",
            "riepe": "splice_altering=1 (>20% mutant RNA materialized upstream), neutral=0",
            "spip": "experimental RNA/minigene class 1=1, 0=0",
        },
        "score_definition": "maximum of DS_AG, DS_AL, DS_DG, and DS_DL",
        "score_mask": score_mask,
        "score_mask_interpretation": (
            "unmasked primary external functional-discrimination estimand"
            if score_mask == 0
            else (
                "masked sensitivity estimand; do not interpret as conventional unmasked max-DS"
                if score_mask == 1
                else "not supplied; masking provenance is unknown"
            )
        ),
        "join_definition": "exact CHROM, POS, REF, ALT, and gene",
        "thresholds": list(thresholds),
        "bootstrap": {
            "method": "nonparametric resampling with replacement, stratified by label and cohort/dataset where applicable",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval": "percentile 95%",
        },
        "coverage": coverage,
        "coverage_policy": coverage_policy,
        "cross_dataset_overlap": overlap_accounting(observations),
        "groups": groups,
        "paired_delta_auroc": paired,
        "warnings": [
            *(
                [
                    "Dataset corpus is explicitly incomplete; results are a sensitivity analysis and are not the primary external benchmark."
                ]
                if dataset_context["state"] == "incomplete_sensitivity"
                else []
            ),
            "Pooled estimates combine heterogeneous assays, genes, prevalence, and ascertainment; cohort-specific results are primary.",
            "Brier score is descriptive and depends on cohort prevalence/calibration.",
            "Missing predictions and conflicting duplicate predictions are excluded, never relabeled as negative.",
            "Observation-level pooled metrics can double-weight cross-dataset overlaps; use pooled_unique_consensus as the deduplicated sensitivity view.",
        ],
        "outputs": {
            "joined_predictions_tsv": str(
                (final_output_dir / predictions_path.name).resolve()
            ),
            "functional_metrics_tsv": str((final_output_dir / metrics_path.name).resolve()),
            "paired_delta_auroc_tsv": str((final_output_dir / paired_path.name).resolve()),
            "report_markdown": str((final_output_dir / report_path.name).resolve()),
            "json": str((final_output_dir / json_path.name).resolve()),
        },
        "plots": [],
        "plot_error": None,
    }
    if render_plots:
        try:
            payload["plots"] = [
                {
                    **plot,
                    "path": str(
                        (final_output_dir / Path(str(plot["path"])).name).resolve()
                    ),
                }
                for plot in _render_plots(groups, output_dir)
            ]
        except (ImportError, ModuleNotFoundError) as exc:
            payload["plot_error"] = f"matplotlib unavailable: {exc}"
        except Exception as exc:  # Plotting is optional; preserve the numerical report.
            payload["plot_error"] = f"plot rendering failed: {type(exc).__name__}: {exc}"
    _atomic_text(report_path, _render_markdown(payload))
    _atomic_json(json_path, payload)
    os.rename(output_dir, final_output_dir)
    return payload


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FunctionalObservation",
    "JoinedPrediction",
    "PredictorIndex",
    "calculate_group_metrics",
    "calculate_paired_delta_aurocs",
    "evaluate_external",
    "join_predictions",
    "load_predictor",
    "normalize_functional_label",
    "read_harmonized",
]
