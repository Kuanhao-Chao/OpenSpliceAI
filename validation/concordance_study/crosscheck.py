"""Independent recomputation of headline statistics from the raw VCFs.

This deliberately shares **no code** with ``validation.full_snv_concordance``: it
re-parses the score VCFs with its own reader and its own arithmetic, so agreement
between the two is evidence that the map/reduce aggregation is faithful on real
data rather than only on the synthetic fixtures in the package's own tests.

Boundary convention, matched to the mapper/reducer: each chunk's first and last
variant group is deferred as an edge and reconciled against the neighbouring
chunk. Over a contiguous window whose neighbours are absent, the reducer
conservatively drops the two outermost groups, so this recomputation drops them
too -- otherwise the two sides would legitimately disagree by a handful of rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

EVENTS = 4


def _canonical(value: str) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _annotations(field: str) -> Tuple[List[Tuple[str, str, Tuple[float, ...]]], int]:
    """Parse one INFO value into (allele, gene, four scores), plus an invalid count.

    The validity rule is deliberately the same one the pipeline applies -- ten
    fields, non-empty allele and gene, four finite scores inside [0, 1] -- so a
    malformed annotation is discarded identically on both sides. Accepting
    something here that the pipeline rejects would make the two disagree for a
    reason that has nothing to do with the aggregation being checked.
    """
    parsed: List[Tuple[str, str, Tuple[float, ...]]] = []
    invalid = 0
    for entry in field.split(","):
        parts = entry.split("|")
        try:
            if len(parts) != 10:
                raise ValueError("field count")
            allele, gene = parts[0].strip(), parts[1].strip()
            if not allele or not gene:
                raise ValueError("empty allele or gene")
            scores = tuple(_canonical(part) for part in parts[2:6])
            if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in scores):
                raise ValueError("score outside [0, 1]")
            for part in parts[6:10]:
                int(part)
        except (TypeError, ValueError):
            invalid += 1
            continue
        parsed.append((allele, gene, scores))
    return parsed, invalid


def _groups(path: Path) -> Iterator[Tuple[Tuple[str, str, str, str], List[str]]]:
    """Yield (variant key, raw INFO strings) for each consecutive run of equal keys."""
    key = None
    infos: List[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            current = (fields[0], fields[1], fields[3], fields[4])
            if current != key:
                if key is not None:
                    yield key, infos
                key, infos = current, []
            infos.append(fields[7])
    if key is not None:
        yield key, infos


class Tally:
    """Additive statistics for the maximum delta score."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.source_rows = 0
        self.variant_groups = 0
        self.rows_with_prediction = 0
        self.left_annotations = 0
        self.right_annotations = 0
        self.paired = 0
        self.conflicts = 0
        self.invalid_annotations = 0
        self.gene_mismatch_left_only = 0
        self.sum_left = 0.0
        self.sum_right = 0.0
        self.sum_abs_diff = 0.0
        self.sum_sq_diff = 0.0
        self.exact_match = 0
        self.both_positive = 0
        self.left_only = 0
        self.right_only = 0
        self.both_negative = 0

    def add_pair(self, left: Sequence[float], right: Sequence[float]) -> None:
        self.paired += 1
        left_max, right_max = max(left), max(right)
        self.sum_left += left_max
        self.sum_right += right_max
        difference = right_max - left_max
        self.sum_abs_diff += abs(difference)
        self.sum_sq_diff += difference * difference
        self.exact_match += int(left_max == right_max)
        left_positive = left_max >= self.threshold
        right_positive = right_max >= self.threshold
        if left_positive and right_positive:
            self.both_positive += 1
        elif left_positive:
            self.left_only += 1
        elif right_positive:
            self.right_only += 1
        else:
            self.both_negative += 1

    def result(self) -> Dict[str, float]:
        n = self.paired or 1
        agreement_n = self.both_positive + self.left_only + self.right_only + self.both_negative
        union = self.both_positive + self.left_only + self.right_only
        return {
            "source_rows": self.source_rows,
            "variant_groups": self.variant_groups,
            "rows_with_prediction": self.rows_with_prediction,
            "left_valid_annotations": self.left_annotations,
            "right_valid_annotations": self.right_annotations,
            "paired_annotations": self.paired,
            "conflicts": self.conflicts,
            "invalid_annotations": self.invalid_annotations,
            "left_only_annotations": self.gene_mismatch_left_only,
            "mean_left": self.sum_left / n,
            "mean_right": self.sum_right / n,
            "bias_right_minus_left": (self.sum_right - self.sum_left) / n,
            "mae": self.sum_abs_diff / n,
            "rmse": math.sqrt(self.sum_sq_diff / n),
            "exact_match_rate": self.exact_match / n,
            "both_positive": self.both_positive,
            "left_only": self.left_only,
            "right_only": self.right_only,
            "both_negative": self.both_negative,
            "overall_agreement": (self.both_positive + self.both_negative) / (agreement_n or 1),
            "jaccard": (self.both_positive / union) if union else None,
        }


def recompute(prediction_vcfs: Sequence[Path], threshold: float = 0.5) -> Dict[str, float]:
    """Recompute over the concatenated chunk stream.

    Chunk boundaries split variant groups -- chunk 46,920 ends and 46,921 begins
    at the same position -- so a per-file pass would count one variant twice. The
    pipeline handles this by deferring each chunk's first and last group and
    rejoining them in the reducer; grouping the concatenated stream by key is
    equivalent and simpler. The first and last group of the whole window are
    dropped because their neighbours lie outside it.
    """
    tally = Tally(threshold)
    pending_key = None
    pending_infos: List[str] = []
    first_group = True
    for path in prediction_vcfs:
        for key, infos in _groups(path):
            if key == pending_key:
                pending_infos.extend(infos)
                continue
            if pending_key is not None:
                if first_group:
                    tally.variant_groups += 1
                    tally.source_rows += len(pending_infos)
                    first_group = False
                else:
                    _consume(tally, pending_key, pending_infos)
            pending_key, pending_infos = key, list(infos)
    if pending_key is not None:
        # trailing group of the window: its neighbour is outside, so drop it
        tally.variant_groups += 1
        tally.source_rows += len(pending_infos)
    return tally.result()


def _consume(tally: Tally, key, infos: Sequence[str]) -> None:
    tally.variant_groups += 1
    tally.source_rows += len(infos)
    alt = key[3]
    sides: Dict[str, Dict[str, set]] = {"left": defaultdict(set), "right": defaultdict(set)}
    for info in infos:
        has_prediction = False
        for item in info.split(";"):
            if item.startswith("SpliceAI="):
                side, payload = "left", item[len("SpliceAI="):]
            elif item.startswith("OpenSpliceAI="):
                side, payload = "right", item[len("OpenSpliceAI="):]
                has_prediction = True
            else:
                continue
            annotations, invalid = _annotations(payload)
            tally.invalid_annotations += invalid
            for allele, gene, scores in annotations:
                if allele != alt:
                    continue
                sides[side][gene].add(scores)
        tally.rows_with_prediction += int(has_prediction)

    left_genes, right_genes = {}, {}
    for side, store in (("left", left_genes), ("right", right_genes)):
        for gene, values in sides[side].items():
            if len(values) > 1:
                tally.conflicts += 1
                continue
            store[gene] = next(iter(values))
    tally.left_annotations += len(left_genes)
    tally.right_annotations += len(right_genes)
    for gene, left in left_genes.items():
        right = right_genes.get(gene)
        if right is None:
            tally.gene_mismatch_left_only += 1
            continue
        tally.add_pair(left, right)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m validation.concordance_study.crosscheck",
        description="Independently recompute headline statistics from prediction VCFs",
    )
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--compare-summary",
                        help="Reduced summary.json to check the recomputation against")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)

    paths: List[Path] = []
    chunk_ids: List[int] = []
    with open(args.pairs_file) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        column = header.index("prediction_vcf")
        chunk_column = header.index("chunk_id")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            paths.append(Path(fields[column]))
            chunk_ids.append(int(fields[chunk_column]))

    result = recompute(paths, args.threshold)
    result["chunks"] = len(paths)
    result["chunk_id_min"] = min(chunk_ids)
    result["chunk_id_max"] = max(chunk_ids)
    result["contiguous"] = (max(chunk_ids) - min(chunk_ids) + 1) == len(chunk_ids)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))

    if args.compare_summary:
        return _compare(result, json.loads(Path(args.compare_summary).read_text()),
                        args.threshold, args.tolerance)
    print(json.dumps(result, indent=2))
    return 0


COMPARISONS = (
    ("paired_annotations", "coverage", "paired_annotations", 0),
    ("left_valid_annotations", "coverage", "left_valid_annotations", 0),
    ("right_valid_annotations", "coverage", "right_valid_annotations", 0),
    ("variant_groups", "coverage", "variant_groups", 0),  # + excluded edge fragments
    ("invalid_annotations", "coverage", "invalid_annotations", 0),
    ("left_only_annotations", "coverage", "left_only_annotations", 0),
    ("source_rows", "coverage", "source_rows", 0),
    ("mean_left", "scores", "mean_left", None),
    ("mean_right", "scores", "mean_right", None),
    ("bias_right_minus_left", "scores", "bias_right_minus_left", None),
    ("mae", "scores", "mae", None),
    ("rmse", "scores", "rmse", None),
    ("exact_match_rate", "scores", "exact_match_rate", None),
    ("both_positive", "threshold", "both_positive", 0),
    ("left_only", "threshold", "left_only", 0),
    ("right_only", "threshold", "right_only", 0),
    ("both_negative", "threshold", "both_negative", 0),
    ("overall_agreement", "threshold", "overall_agreement", None),
    ("jaccard", "threshold", "jaccard", None),
)


def _compare(independent: Dict[str, float], summary: Dict, threshold: float,
             tolerance: float) -> int:
    metrics = summary["metrics"]
    edge_fragments = int(metrics["coverage"].get("excluded_incomplete_edge_fragments", 0))
    sources = {
        "coverage": metrics["coverage"],
        "scores": metrics["scores"]["MAX"],
        "threshold": metrics["thresholds"]["MAX"][f"{threshold:.10g}"],
    }
    width = max(len(name) for name, _, _, _ in COMPARISONS)
    print(f"{'statistic'.ljust(width)}  {'independent':>20}  {'pipeline':>20}  {'delta':>14}  verdict")
    print("-" * (width + 66))
    failures = 0
    for name, group, key, exact in COMPARISONS:
        mine = independent.get(name)
        theirs = sources[group].get(key)
        if name in ("variant_groups", "source_rows") and theirs is not None:
            # The pipeline's coverage counts only what entered the aggregate; the two
            # window-edge groups it drops (their neighbours lie outside the window) are
            # recorded separately. This recomputation counts them, so add them back
            # rather than reporting a difference that is purely an accounting boundary.
            theirs = theirs + edge_fragments
        if mine is None or theirs is None:
            print(f"{name.ljust(width)}  {'-':>20}  {'-':>20}  {'-':>14}  SKIP (undefined)")
            continue
        delta = mine - theirs
        if exact == 0:
            ok = int(mine) == int(theirs)
            fmt = f"{mine:>20,.0f}  {theirs:>20,.0f}  {delta:>14,.0f}"
        else:
            ok = abs(delta) <= tolerance
            fmt = f"{mine:>20.10f}  {theirs:>20.10f}  {delta:>14.2e}"
        failures += (not ok)
        print(f"{name.ljust(width)}  {fmt}  {'MATCH' if ok else 'DIFFERS'}")
    print()
    print(f"({edge_fragments} window-edge group(s) excluded by the pipeline were added back "
          f"to its variant_groups/source_rows before comparison.)")
    if failures:
        print(f"{failures} statistic(s) disagree beyond tolerance {tolerance:g}")
    else:
        print("Independent recomputation reproduces every pipeline statistic.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
