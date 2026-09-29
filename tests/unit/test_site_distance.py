"""Tests for the annotated-splice-site derivation and the site_distance stratum.

The coordinate convention here is not a free choice: it has to match the one the
scorer used when it applied `-M 1`, or the distance strata describe the gap
between two annotations rather than anything about the predictors. These tests
pin the convention (0-based EXON_START, 1-based-inclusive EXON_END, strand-swapped
roles, transcript ends excluded) against hand-computed coordinates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.full_snv_concordance.aggregate import (
    STANDARD_STRATA,
    Aggregate,
    AnalysisConfig,
    _bin_index,
)
from validation.full_snv_concordance.sites import (
    ACCEPTOR,
    DONOR,
    NO_SITE,
    distance_bin,
    load_site_index,
)
from validation.full_snv_concordance.vcf import Annotation, VariantGroup, VariantKey

HEADER = "#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n"


def _annotation(tmp_path: Path, rows) -> Path:
    path = tmp_path / "anno.txt"
    path.write_text(HEADER + "".join(rows))
    return path


def _row(name, chrom, strand, starts, ends):
    lo, hi = min(starts), max(ends)
    return (f"{name}\t{chrom}\t{strand}\t{lo}\t{hi}\t"
            f"{','.join(str(v) for v in starts)},\t{','.join(str(v) for v in ends)},\n")


# --------------------------------------------------------------------------
def test_plus_strand_roles_and_one_based_conversion(tmp_path: Path) -> None:
    # three exons, 0-based starts / 1-based-inclusive ends
    path = _annotation(tmp_path, [_row("G", "chr1", "+", [100, 200, 300], [150, 250, 350])])
    index = load_site_index(path)

    # internal boundaries only: exon-1 end and exon-2 end are donors; exon-2 and
    # exon-3 starts are acceptors. Starts get +1; ends do not.
    donors = {150, 250}
    acceptors = {201, 301}
    observed = {int(p): str(t) for p, t in zip(index.positions["chr1"], index.types["chr1"])}
    assert set(observed) == donors | acceptors
    assert {p for p, t in observed.items() if t == DONOR} == donors
    assert {p for p, t in observed.items() if t == ACCEPTOR} == acceptors
    # The transcript's own ends are not splice sites: the first exon's start
    # (0-based 100 -> 1-based 101) and the last exon's end (350) are both absent.
    assert 101 not in observed
    assert 350 not in observed
    assert min(observed) == 150 and max(observed) == 301


def test_minus_strand_swaps_donor_and_acceptor(tmp_path: Path) -> None:
    path = _annotation(tmp_path, [_row("G", "chr2", "-", [100, 200, 300], [150, 250, 350])])
    index = load_site_index(path)
    observed = {int(p): str(t) for p, t in zip(index.positions["chr2"], index.types["chr2"])}
    # same coordinates as the plus-strand gene, opposite labels
    assert {p for p, t in observed.items() if t == DONOR} == {201, 301}
    assert {p for p, t in observed.items() if t == ACCEPTOR} == {150, 250}


def test_single_exon_gene_contributes_no_sites(tmp_path: Path) -> None:
    path = _annotation(tmp_path, [_row("SOLO", "chr3", "+", [100], [200])])
    index = load_site_index(path)
    assert index.gene_count == 1
    assert index.site_count == 0
    assert index.nearest("chr3", 150) == (None, NO_SITE)


def test_nearest_returns_signed_distance(tmp_path: Path) -> None:
    path = _annotation(tmp_path, [_row("G", "chr1", "+", [100, 200], [150, 250])])
    index = load_site_index(path)          # donor 150, acceptor 201
    assert index.nearest("chr1", 150) == (0, DONOR)
    assert index.nearest("chr1", 148) == (2, DONOR)      # site lies after the variant
    assert index.nearest("chr1", 152) == (-2, DONOR)     # site lies before it
    assert index.nearest("chr1", 199) == (2, ACCEPTOR)
    assert index.nearest("chr1", 100_000)[1] == ACCEPTOR


def test_distance_bins() -> None:
    assert distance_bin(0) == "at_site"
    assert distance_bin(-1) == "1-2" and distance_bin(2) == "1-2"
    assert distance_bin(3) == "3-10" and distance_bin(-10) == "3-10"
    assert distance_bin(11) == "11-50"
    assert distance_bin(51) == "51-500"
    assert distance_bin(501) == ">500"
    assert distance_bin(None) == NO_SITE


def test_duplicate_coordinates_across_genes_are_collapsed(tmp_path: Path) -> None:
    rows = [_row("A", "chr1", "+", [100, 200], [150, 250]),
            _row("B", "chr1", "+", [100, 200], [150, 250])]
    index = load_site_index(_annotation(tmp_path, rows))
    assert index.gene_count == 2
    assert index.site_count == 2          # not 4
    assert list(index.positions["chr1"]) == [150, 201]


def test_opposite_roles_and_equal_distance_ties_are_explicit(tmp_path: Path) -> None:
    rows = [_row("A", "chr1", "+", [100, 200], [150, 250]),
            _row("B", "chr1", "-", [100, 200], [150, 250])]
    index = load_site_index(_annotation(tmp_path, rows))
    assert index.nearest("chr1", 150) == (0, "ambiguous")
    reversed_index = load_site_index(_annotation(tmp_path, rows[::-1]))
    assert reversed_index.nearest("chr1", 150) == (0, "ambiguous")
    # Sites 150 and 202 are equally near 176 but have different roles.
    index = load_site_index(_annotation(tmp_path, [_row("A", "chr1", "+", [100, 201], [150, 250])]))
    assert index.nearest("chr1", 176) == (-26, "ambiguous")


# --------------------------------------------------------------------------
def _pair(aggregate, chrom, pos, left=(0.4, 0, 0, 0), right=(0.35, 0, 0, 0)):
    key = VariantKey(chrom, pos, "G", "A")
    group = VariantGroup(key, rows=1)
    group.add("left", Annotation("A", "G", tuple(left), (1, 2, 3, 4)))
    group.add("right", Annotation("A", "G", tuple(right), (1, 2, 3, 4)))
    aggregate.add_group(group)


def test_stratum_keys_and_merge_guard(tmp_path: Path) -> None:
    index = load_site_index(_annotation(tmp_path, [_row("G", "chr1", "+", [100, 200], [150, 250])]))
    config = AnalysisConfig(thresholds=(0.5,), score_bins=100, sample_size=0, top_k=0,
                            strata=STANDARD_STRATA + ("site_distance",),
                            sites_digest=index.digest)
    aggregate = Aggregate(config, sites=index)
    _pair(aggregate, "chr1", 150)     # exactly on the donor
    _pair(aggregate, "chr1", 149)     # 1 bp away
    _pair(aggregate, "chr1", 9_000)   # far
    strata = aggregate.derived()["strata"]["site_distance"]
    assert {k: v["metrics"]["n"] for k, v in strata.items()} == {
        "at_site:donor": 1, "1-2:donor": 1, ">500:acceptor": 1,
    }

    # a differing sites digest must not merge
    other = Aggregate(AnalysisConfig(thresholds=(0.5,), score_bins=100, sample_size=0,
                                     top_k=0, strata=STANDARD_STRATA + ("site_distance",),
                                     sites_digest="other"), sites=index)
    with pytest.raises(ValueError, match="different configurations"):
        aggregate.merge(other)


def test_stratum_requires_an_index_and_a_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a sites file"):
        AnalysisConfig(strata=("site_distance",))
    config = AnalysisConfig(thresholds=(0.5,), score_bins=100, sample_size=0, top_k=0,
                            strata=("site_distance",), sites_digest="abc")
    aggregate = Aggregate(config)          # constructing is fine (the reducer does it)
    with pytest.raises(ValueError, match="requires a SiteIndex"):
        _pair(aggregate, "chr1", 150)      # adding a row is not


def test_reducer_can_rebuild_without_the_index(tmp_path: Path) -> None:
    """from_dict must work in the reducer, which has no annotation on hand."""
    index = load_site_index(_annotation(tmp_path, [_row("G", "chr1", "+", [100, 200], [150, 250])]))
    config = AnalysisConfig(thresholds=(0.5,), score_bins=100, sample_size=0, top_k=0,
                            strata=STANDARD_STRATA + ("site_distance",),
                            sites_digest=index.digest)
    source = Aggregate(config, sites=index)
    _pair(source, "chr1", 150)
    rebuilt = Aggregate.from_dict(source.to_dict())
    assert rebuilt.sites is None
    assert rebuilt.to_dict()["strata"]["site_distance"] == source.to_dict()["strata"]["site_distance"]
    merged = Aggregate.from_dict(source.to_dict())
    merged.merge(rebuilt)                  # merging populated aggregates needs no index
    assert merged.derived()["strata"]["site_distance"]["at_site:donor"]["metrics"]["n"] == 2


# --------------------------------------------------------------------------
def test_binning_puts_two_decimal_values_on_their_own_bin() -> None:
    """int(v * bins) drops 0.29/0.57/0.58 a bin; the comparator publishes two decimals."""
    for bins in (100, 200):
        for step in range(bins + 1):
            value = round(step / bins, 10)
            assert _bin_index(value, bins) == min(bins - 1, step), (value, bins)
    # the classic failures
    assert _bin_index(0.29, 100) == 29
    assert _bin_index(0.57, 100) == 57
    assert _bin_index(0.58, 100) == 58
    # and values genuinely below a boundary still bin down
    assert _bin_index(0.2899, 100) == 28
    assert _bin_index(1.0, 100) == 99
    assert _bin_index(0.0, 100) == 0


def test_site_distance_is_opt_in_not_default() -> None:
    """It needs an external annotation, so a default config must not demand one."""
    from validation.full_snv_concordance.aggregate import STANDARD_STRATA, SUPPORTED_STRATA

    assert "site_distance" not in STANDARD_STRATA
    assert "site_distance" in SUPPORTED_STRATA
    AnalysisConfig()                       # the default config needs no sites file
    with pytest.raises(ValueError, match="unsupported strata"):
        AnalysisConfig(strata=("no_such_stratum",))


# --------------------------------------------------------------------------
# Regression: the reducer is not a pure merger. It rejoins the chunk-boundary
# groups the mappers deferred, and those are real rows, so it needs the same
# annotation. A pilot run caught this by failing after the map step.
# --------------------------------------------------------------------------
def test_reducer_requires_a_matching_sites_file(tmp_path: Path) -> None:
    from validation.full_snv_concordance.aggregate import STANDARD_STRATA
    from validation.full_snv_concordance.workflows import run_map_concordance, run_reduce

    from tests.unit.test_full_snv_concordance import ann, row, write_pairs, write_vcf

    first, second = tmp_path / "one.vcf", tmp_path / "two.vcf"
    write_vcf(first, [row(100, ann("G", (0.4, 0, 0, 0)), ann("G", (0.3, 0, 0, 0))),
                      row(200, ann("G", (0.6, 0, 0, 0)), ann("G", (0.5, 0, 0, 0)))])
    write_vcf(second, [row(300, ann("G", (0.7, 0, 0, 0)), ann("G", (0.6, 0, 0, 0))),
                       row(400, ann("G", (0.2, 0, 0, 0)), ann("G", (0.1, 0, 0, 0)))])
    pairs = tmp_path / "pairs.tsv"
    write_pairs(pairs, [{"chunk_id": "1", "source_vcf": first, "prediction_vcf": first},
                        {"chunk_id": "2", "source_vcf": second, "prediction_vcf": second}])

    index = load_site_index(_annotation(tmp_path, [_row("G", "chr1", "+", [100, 250], [150, 400])]))
    config = AnalysisConfig(thresholds=(0.2,), score_bins=100, sample_size=0, top_k=0,
                            strata=STANDARD_STRATA + ("site_distance",),
                            sites_digest=index.digest)
    maps = tmp_path / "maps"
    maps.mkdir()
    for task in range(2):
        run_map_concordance(pairs, maps / f"map-{task}.json", config,
                            task_index=task, chunks_per_task=1, sites=index)

    # without the index the reducer must refuse rather than silently drop the stratum
    with pytest.raises(ValueError, match="--sites-file is required"):
        run_reduce("concordance", tmp_path / "s1.json", input_dir=maps, pairs_file=pairs,
                   expected_task_count=2, expected_total_chunks=2, finality="provisional")

    # a different annotation must not be substituted
    other = load_site_index(_annotation(tmp_path, [_row("H", "chr1", "+", [500, 700], [600, 800])]))
    with pytest.raises(ValueError, match="does not match the mappers"):
        run_reduce("concordance", tmp_path / "s2.json", input_dir=maps, pairs_file=pairs,
                   expected_task_count=2, expected_total_chunks=2, finality="provisional",
                   sites=other)

    # the matching index reduces, and the rejoined boundary group is stratified
    reduced = run_reduce("concordance", tmp_path / "s3.json", input_dir=maps, pairs_file=pairs,
                         expected_task_count=2, expected_total_chunks=2, finality="final",
                         sites=index)
    strata = reduced["metrics"]["strata"]["site_distance"]
    assert sum(v["metrics"]["n"] for v in strata.values()) == \
        reduced["metrics"]["coverage"]["paired_annotations"]
