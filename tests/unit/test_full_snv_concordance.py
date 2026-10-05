import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from validation.full_snv_concordance.aggregate import Aggregate, AnalysisConfig
from validation.full_snv_concordance.cli import build_parser, main
from validation.full_snv_concordance.external import (
    ChainLiftOver,
    _dataset_rows,
    harmonize_external,
    load_source_manifest,
)
from validation.full_snv_concordance.vcf import (
    Annotation,
    VariantGroup,
    VariantKey,
    iter_vcf_records,
    iter_variant_groups,
    parse_annotation,
)
from validation.full_snv_concordance.workflows import (
    PairRow,
    audit_pair,
    build_pairs_files,
    run_map_concordance,
    run_map_seeds,
    run_reduce,
)


HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI">
##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""
SOURCE_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


def ann(gene, scores=(0.0, 0.0, 0.0, 0.0), dps=(0, 0, 0, 0)):
    return "A|{}|{}|{}".format(
        gene,
        "|".join(str(value) for value in scores),
        "|".join(str(value) for value in dps),
    )


def row(pos, spliceai, openspliceai=None):
    info = f"SpliceAI={spliceai}"
    if openspliceai is not None:
        info += f";OpenSpliceAI={openspliceai}"
    return f"chr1\t{pos}\t.\tG\tA\t.\t.\t{info}\n"


def write_vcf(path: Path, rows, header=HEADER):
    path.write_text(header + "".join(rows), encoding="utf-8")


def write_pairs(path: Path, rows):
    frozen_rows = []
    for value in rows:
        source_path = Path(value["source_vcf"])
        prediction_path = Path(value["prediction_vcf"])
        if source_path.resolve() == prediction_path.resolve():
            source_path = path.parent / f"source-{value['chunk_id']}.vcf"
            source_path.write_text(
                "\n".join(
                    line.split(";OpenSpliceAI=", 1)[0]
                    for line in prediction_path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
        audited = audit_pair(
            PairRow(
                str(value["chunk_id"]),
                {
                    "source_vcf": str(source_path),
                    "prediction_vcf": str(prediction_path),
                },
            )
        )
        assert audited["state"] == "valid"
        frozen_rows.append(
            {
                "chunk_id": str(value["chunk_id"]),
                "source_vcf": str(source_path),
                "prediction_vcf": str(prediction_path),
                "seed": str(value.get("seed", "")),
                "source_bytes": audited["source_bytes"],
                "source_mtime_ns": audited["source_mtime_ns"],
                "source_rows": audited["source_rows"],
                "prediction_bytes": audited["prediction_bytes"],
                "prediction_mtime_ns": audited["prediction_mtime_ns"],
                "prediction_rows": audited["prediction_rows"],
                "source_digest": audited["source_digest"],
                "prediction_source_digest": audited["prediction_source_digest"],
                "input_file_sha256": audited["input_file_sha256"],
                "output_file_sha256": audited["output_file_sha256"],
                "manifest_sha256": "a" * 64,
                "output_provenance_class": "legacy_unprovenanced",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(frozen_rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(frozen_rows)


def test_annotation_negative_zero_and_exact_field_validation():
    parsed = parse_annotation("A|GENE|-0.00000|0.2|0|1|0|-1|2|3")
    assert parsed.scores[0] == 0.0
    assert str(parsed.scores[0]) == "0.0"
    with pytest.raises(ValueError, match="10 annotation fields"):
        parse_annotation("A|GENE|0|0")


def test_campaign_manifest_aliases_and_snapshot_digest_are_supported(tmp_path):
    source = tmp_path / "source.vcf"
    prediction = tmp_path / "prediction.vcf"
    write_vcf(source, [row(1, ann("G", (0.4, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(
        prediction,
        [row(1, ann("G", (0.4, 0, 0, 0)), ann("G", (0.5, 0, 0, 0)))],
    )
    source_identity = {}
    prediction_identity = {}
    list(iter_vcf_records(source, verification_result=source_identity))
    list(
        iter_vcf_records(
            prediction,
            verification_result=prediction_identity,
            canonical_drop=("OpenSpliceAI",),
        )
    )
    manifest = tmp_path / "campaign.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "chunk_id",
                "state",
                "input_path",
                "output_path",
                "input_size",
                "input_mtime_ns",
                "input_records",
                "input_source_digest",
                "input_file_sha256",
                "output_size",
                "output_mtime_ns",
                "output_records",
                "output_source_digest",
                "output_file_sha256",
                "provenance_state",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "chunk_id": "1",
                "state": "valid",
                "input_path": str(source),
                "output_path": str(prediction),
                "input_size": source.stat().st_size,
                "input_mtime_ns": source.stat().st_mtime_ns,
                "input_records": source_identity["records"],
                "input_source_digest": source_identity["length_prefixed_digest"],
                "input_file_sha256": source_identity["file_sha256"],
                "output_size": prediction.stat().st_size,
                "output_mtime_ns": prediction.stat().st_mtime_ns,
                "output_records": prediction_identity["records"],
                "output_source_digest": prediction_identity["length_prefixed_digest"],
                "output_file_sha256": prediction_identity["file_sha256"],
                "provenance_state": "legacy_unprovenanced",
            }
        )
    pairs = tmp_path / "pairs.tsv"
    result = build_pairs_files(manifest, pairs, require_left_count=1)
    assert result["concordance_count"] == 1
    with pairs.open(encoding="utf-8", newline="") as handle:
        row_data = next(csv.DictReader(handle, delimiter="\t"))
    assert row_data["source_digest"] == source_identity["length_prefixed_digest"]
    mapped = tmp_path / "map.json"
    run_map_concordance(
        pairs,
        mapped,
        AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
        task_index=0,
        chunks_per_task=1,
    )
    assert mapped.exists()


def test_multigene_dedup_conflict_gene_mismatch_and_dp_zero():
    config = AnalysisConfig(thresholds=(0.2, 0.5), score_bins=10, sample_size=5, top_k=5)
    aggregate = Aggregate(config)

    group = VariantGroup(VariantKey("chr1", 10, "G", "A"), rows=2)
    left_a = Annotation("A", "A", (0.8, 0.0, 0.0, 0.0), (0, 0, 0, 0))
    right_a = Annotation("A", "A", (0.7, 0.0, 0.0, 0.0), (0, 0, 0, 0))
    left_b = Annotation("A", "B", (0.1, 0.0, 0.0, 0.0), (7, 0, 0, 0))
    right_b = Annotation("A", "B", (0.3, 0.0, 0.0, 0.0), (9, 0, 0, 0))
    for annotation in (left_a, left_b):
        group.add("left", annotation)
    for _ in range(2):
        for annotation in (right_a, right_b):
            group.add("right", annotation)
    aggregate.add_group(group)

    mismatch = VariantGroup(VariantKey("chr1", 11, "G", "A"), rows=1)
    mismatch.add("left", Annotation("A", "LEFT", (0.4, 0, 0, 0), (1, 0, 0, 0)))
    mismatch.add("right", Annotation("A", "RIGHT", (0.4, 0, 0, 0), (1, 0, 0, 0)))
    aggregate.add_group(mismatch)

    conflict = VariantGroup(VariantKey("chr1", 12, "G", "A"), rows=1)
    conflict.add("left", left_a)
    conflict.add("right", right_a)
    conflict.add("right", Annotation("A", "A", (0.6, 0, 0, 0), (0, 0, 0, 0)))
    aggregate.add_group(conflict)

    assert aggregate.coverage["paired_annotations"] == 2
    assert aggregate.coverage["duplicate_annotations"] == 2
    assert aggregate.coverage["gene_mismatch_variants"] == 1
    assert aggregate.coverage["right_conflicts"] == 1
    # DP=0 is a valid at-variant prediction after score gating.
    dp = aggregate.dp["AG"]["0.5"]
    assert dp["both_above"] == dp["eligible"] == 1
    assert dp["within"]["0"] == 1
    table = aggregate.thresholds["AG"]["0.2"]
    assert table == {
        "both_positive": 1,
        "left_only": 0,
        "right_only": 1,
        "both_negative": 0,
    }


def test_map_reduce_matches_direct_and_reconciles_chunk_boundary(tmp_path):
    first = tmp_path / "chunk1.vcf"
    second = tmp_path / "chunk2.vcf"
    repeated_osai = ",".join(
        [ann("X", (0.9, 0, 0, 0), (0, 0, 0, 0)), ann("Y", (0, 0.8, 0, 0), (0, 2, 0, 0))]
    )
    write_vcf(
        first,
        [
            row(100, ann("ONE", (0.1, 0, 0, 0)), ann("ONE", (0.2, 0, 0, 0))),
            row(200, ann("MID", (0.3, 0, 0, 0)), ann("MID", (0.4, 0, 0, 0))),
            row(300, ann("X", (0.9, 0, 0, 0)), repeated_osai),
        ],
    )
    write_vcf(
        second,
        [
            row(300, ann("Y", (0, 0.8, 0, 0)), repeated_osai),
            row(400, ann("LEFT", (0.6, 0, 0, 0)), ann("RIGHT", (0.6, 0, 0, 0))),
        ],
    )
    pairs = tmp_path / "pairs.tsv"
    write_pairs(
        pairs,
        [
            {"chunk_id": "1", "source_vcf": first, "prediction_vcf": first},
            {"chunk_id": "2", "source_vcf": second, "prediction_vcf": second},
        ],
    )
    config = AnalysisConfig(thresholds=(0.2,), score_bins=10, sample_size=10, top_k=10)
    maps = tmp_path / "maps"
    maps.mkdir()
    for task in range(2):
        run_map_concordance(
            pairs,
            maps / f"map-{task}.json",
            config,
            task_index=task,
            chunks_per_task=1,
        )
    reduced = run_reduce(
        "concordance",
        tmp_path / "summary.json",
        input_dir=maps,
        pairs_file=pairs,
        expected_task_count=2,
        expected_total_chunks=2,
        finality="final",
    )

    direct = Aggregate(config)
    for group in iter_variant_groups(
        (first, second), {"left": "SpliceAI", "right": "OpenSpliceAI"}
    ):
        direct.add_group(group)
    assert reduced["raw"]["moments"] == direct.to_dict()["moments"]
    assert reduced["metrics"]["coverage"] == direct.derived()["coverage"]
    assert reduced["metrics"]["coverage"]["paired_annotations"] == 4
    assert reduced["finality"]["status"] == "final"
    assert reduced["verified_provenance"]["status"] == "verified"
    # X/Y OpenSpliceAI lists repeat on both source rows but each gene pairs once.
    assert reduced["metrics"]["coverage"]["duplicate_annotations"] == 2


def test_reducer_fails_closed_on_missing_task_or_chunk(tmp_path):
    vcf = tmp_path / "one.vcf"
    write_vcf(vcf, [row(1, ann("G"), ann("G"))])
    pairs = tmp_path / "pairs.tsv"
    write_pairs(pairs, [{"chunk_id": "1", "source_vcf": vcf, "prediction_vcf": vcf}])
    maps = tmp_path / "maps"
    maps.mkdir()
    run_map_concordance(
        pairs,
        maps / "map-0.json",
        AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
        task_index=0,
        chunks_per_task=1,
    )
    with pytest.raises(ValueError, match="incomplete mapper task set"):
        run_reduce(
            "concordance",
            tmp_path / "summary.json",
            input_dir=maps,
            pairs_file=pairs,
            expected_task_count=2,
        )


def test_mapper_rejects_mutated_output_even_when_stat_and_source_identity_match(tmp_path):
    source = tmp_path / "source.vcf"
    prediction = tmp_path / "prediction.vcf"
    write_vcf(source, [row(1, ann("G", (0.4, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(
        prediction,
        [row(1, ann("G", (0.4, 0, 0, 0)), ann("G", (0.5, 0, 0, 0)))],
    )
    pairs = tmp_path / "pairs.tsv"
    write_pairs(
        pairs,
        [{"chunk_id": "1", "source_vcf": source, "prediction_vcf": prediction}],
    )

    frozen_stat = prediction.stat()
    write_vcf(
        prediction,
        [row(1, ann("G", (0.4, 0, 0, 0)), ann("G", (0.6, 0, 0, 0)))],
    )
    assert prediction.stat().st_size == frozen_stat.st_size
    os.utime(
        prediction,
        ns=(frozen_stat.st_atime_ns, frozen_stat.st_mtime_ns),
    )

    with pytest.raises(ValueError, match="VCF snapshot identity mismatch"):
        run_map_concordance(
            pairs,
            tmp_path / "map.json",
            AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
            task_index=0,
            chunks_per_task=1,
        )


def test_provisional_gap_excludes_only_groups_touching_missing_chunk(tmp_path):
    first = tmp_path / "chunk1.vcf"
    third = tmp_path / "chunk3.vcf"
    write_vcf(
        first,
        [
            row(10, ann("SAFE1", (0.2, 0, 0, 0)), ann("SAFE1", (0.2, 0, 0, 0))),
            row(20, ann("GAPLEFT", (0.3, 0, 0, 0)), ann("GAPLEFT", (0.3, 0, 0, 0))),
        ],
    )
    write_vcf(
        third,
        [
            row(30, ann("GAPRIGHT", (0.4, 0, 0, 0)), ann("GAPRIGHT", (0.4, 0, 0, 0))),
            row(40, ann("SAFE3", (0.5, 0, 0, 0)), ann("SAFE3", (0.5, 0, 0, 0))),
        ],
    )
    pairs = tmp_path / "pairs.tsv"
    write_pairs(
        pairs,
        [
            {"chunk_id": "1", "source_vcf": first, "prediction_vcf": first},
            {"chunk_id": "3", "source_vcf": third, "prediction_vcf": third},
        ],
    )
    maps = tmp_path / "maps"
    maps.mkdir()
    run_map_concordance(
        pairs,
        maps / "map-0.json",
        AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
        task_index=0,
        chunks_per_task=2,
    )
    reduced = run_reduce(
        "concordance",
        tmp_path / "summary.json",
        input_dir=maps,
        pairs_file=pairs,
        expected_task_count=1,
        expected_total_chunks=3,
    )
    assert reduced["metrics"]["coverage"]["paired_annotations"] == 2
    assert reduced["metrics"]["coverage"]["excluded_incomplete_edge_fragments"] == 2
    assert reduced["finality"]["status"] == "provisional"
    assert reduced["finality"]["excluded_incomplete_edge_fragments"] == 2
    with pytest.raises(ValueError, match="final concordance requires pair IDs exactly"):
        run_reduce(
            "concordance",
            tmp_path / "invalid-final.json",
            input_dir=maps,
            pairs_file=pairs,
            expected_task_count=1,
            expected_total_chunks=3,
            finality="final",
        )


@pytest.mark.parametrize("missing_left", [True, False])
def test_gap_excludes_entire_variant_spanning_a_whole_chunk(tmp_path, missing_left):
    # A fragment that fills a chunk can carry incompleteness into the next
    # fragment. Keeping the locally safe piece would hide its conflicting value.
    rows = ([row(10, ann("LONG", (0.3, 0, 0, 0)), ann("LONG"))],
            [row(10, ann("LONG", (0.4, 0, 0, 0)), ann("LONG")),
             row(20, ann("SAFE"), ann("SAFE"))]) if missing_left else (
                [row(10, ann("SAFE"), ann("SAFE")),
                 row(20, ann("LONG", (0.3, 0, 0, 0)), ann("LONG"))],
                [row(20, ann("LONG", (0.4, 0, 0, 0)), ann("LONG"))])
    ids = (2, 3) if missing_left else (1, 2)
    pair_rows = []
    for chunk, records in zip(ids, rows):
        path = tmp_path / f"chunk{chunk}.vcf"
        write_vcf(path, records)
        pair_rows.append({"chunk_id": str(chunk), "source_vcf": path, "prediction_vcf": path})
    pairs = tmp_path / "pairs.tsv"
    write_pairs(pairs, pair_rows)
    maps = tmp_path / "maps"
    maps.mkdir()
    run_map_concordance(pairs, maps / "map-0.json",
                        AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
                        task_index=0, chunks_per_task=2)
    result = run_reduce("concordance", tmp_path / "summary.json", input_dir=maps,
                        pairs_file=pairs, expected_task_count=1, expected_total_chunks=3)
    assert result["metrics"]["coverage"]["paired_annotations"] == 1
    assert set(result["metrics"]["strata"]["gene"]) == {"SAFE"}
    assert result["finality"]["excluded_incomplete_edge_fragments"] == 2


def test_audit_header_dp_bounds_and_source_preservation(tmp_path):
    source = tmp_path / "source.vcf"
    prediction = tmp_path / "prediction.vcf"
    write_vcf(source, [row(1, ann("G", (0.8, 0, 0, 0), (0, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(
        prediction,
        [row(1, ann("G", (0.8, 0, 0, 0), (0, 0, 0, 0)), ann("G", (0.7, 0, 0, 0), (50, 0, 0, 0)))],
    )
    pair = PairRow("1", {"source_vcf": str(source), "prediction_vcf": str(prediction)})
    result = audit_pair(pair, dp_distance=50)
    assert result["state"] == "valid"
    assert result["openspliceai_info_header"] is True
    assert result["preserved_rows"] == 1

    write_vcf(
        prediction,
        [row(1, ann("G", (0.8, 0, 0, 0)), ann("G", (0.7, 0, 0, 0), (51, 0, 0, 0)))],
    )
    result = audit_pair(pair, dp_distance=50)
    assert result["state"] == "annotation_error"
    assert result["out_of_range_openspliceai_dp"] == 1

    no_osai_header = HEADER.replace(
        '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI">\n', ""
    )
    write_vcf(
        prediction,
        [row(1, ann("G", (0.8, 0, 0, 0)), ann("G", (0.7, 0, 0, 0)))],
        no_osai_header,
    )
    assert audit_pair(pair)["state"] == "missing_openspliceai_info_header"


def test_build_pairs_filters_valid_intersection_and_sorts_numerically(tmp_path):
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    source2 = tmp_path / "s2.vcf"
    source10 = tmp_path / "s10.vcf"
    left2 = tmp_path / "l2.vcf"
    left10 = tmp_path / "l10.vcf"
    right2 = tmp_path / "r2.vcf"
    write_vcf(source2, [row(2, ann("G", (0.2, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(source10, [row(10, ann("G", (0.3, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(left2, [row(2, ann("G", (0.2, 0, 0, 0)), ann("G"))])
    write_vcf(left10, [row(10, ann("G", (0.3, 0, 0, 0)), ann("G"))])
    write_vcf(right2, [row(2, ann("G", (0.2, 0, 0, 0)), ann("G"))])

    left_rows = [
        audit_pair(PairRow("10", {"source_vcf": str(source10), "prediction_vcf": str(left10)})),
        audit_pair(PairRow("2", {"source_vcf": str(source2), "prediction_vcf": str(left2)})),
        {"chunk_id": "3", "state": "truncated", "source_vcf": "s3", "prediction_vcf": "l3"},
    ]
    # The production scorer auditor calls this field provenance_state.  The
    # pair builder must preserve and normalize it for analysis/reporting.
    left_rows[1].pop("output_provenance_class", None)
    left_rows[1]["provenance_state"] = "receipt_bound"
    right_rows = [
        audit_pair(PairRow("2", {"source_vcf": str(source2), "prediction_vcf": str(right2)})),
        {"chunk_id": "10", "state": "missing", "source_vcf": str(source10), "prediction_vcf": "r10"},
    ]

    def write_manifest(path, records):
        fields = sorted({field for record in records for field in record})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

    write_manifest(left, left_rows)
    write_manifest(right, right_rows)
    concordance = tmp_path / "concordance.tsv"
    seeds = tmp_path / "seeds.tsv"
    result = build_pairs_files(left, concordance, right, seeds)
    assert result["concordance_count"] == 2
    assert result["seed_overlap_count"] == 1
    assert concordance.read_text().splitlines()[1].startswith("2\t")
    with concordance.open(encoding="utf-8", newline="") as handle:
        concordance_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert concordance_rows[0]["input_file_sha256"]
    assert concordance_rows[0]["output_file_sha256"]
    assert concordance_rows[0]["manifest_sha256"] == result["left_manifest_sha256"]
    assert concordance_rows[0]["output_provenance_class"] == "receipt_bound"
    assert concordance_rows[0]["provenance_state"] == "receipt_bound"
    with seeds.open(encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert seed_rows[0]["chunk_id"] == "2"
    assert seed_rows[0]["left_vcf"] == str(left2)
    assert seed_rows[0]["right_vcf"] == str(right2)
    assert seed_rows[0]["left_source_digest"] == seed_rows[0]["right_source_digest"]
    assert seed_rows[0]["left_output_file_sha256"]
    assert seed_rows[0]["right_output_file_sha256"]
    assert seed_rows[0]["left_output_provenance_class"] == "receipt_bound"
    assert seed_rows[0]["right_output_provenance_class"] == "legacy_unprovenanced"

    seed_maps = tmp_path / "seed-maps"
    seed_maps.mkdir()
    run_map_seeds(
        seeds,
        seed_maps / "map-0.json",
        AnalysisConfig(score_bins=10, sample_size=0, top_k=0),
        task_index=0,
        chunks_per_task=1,
    )
    with pytest.raises(ValueError, match="explicit expected_overlap_count policy"):
        run_reduce(
            "seeds",
            tmp_path / "seed-final-missing-policy.json",
            input_dir=seed_maps,
            pairs_file=seeds,
            expected_task_count=1,
            expected_total_chunks=10,
            finality="final",
        )
    final_seed = run_reduce(
        "seeds",
        tmp_path / "seed-final.json",
        input_dir=seed_maps,
        pairs_file=seeds,
        expected_task_count=1,
        expected_total_chunks=10,
        finality="final",
        expected_overlap_count=1,
    )
    assert final_seed["finality"]["policy"] == "explicit_seed_overlap_count"


def test_cli_exposes_all_workflow_commands():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {
        "audit",
        "build-pairs",
        "map-concordance",
        "reduce-concordance",
        "map-seeds",
        "reduce-seeds",
        "render-report",
        "validate-external",
        "evaluate-external",
    } <= set(choices)


@pytest.mark.parametrize("failed_call", [1, 2, 3])
def test_launcher_cancels_only_registered_jobs_on_dependency_failure(tmp_path, failed_call):
    pairs = tmp_path / "launcher-pairs.tsv"
    pairs.write_text("chunk_id\n1\n", encoding="utf-8")
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    state = tmp_path / "mock-slurm"
    state.mkdir()
    scripts = {
        "sbatch": """#!/usr/bin/env bash
set -eu
count_file="$MOCK_SLURM_DIR/sbatch.count"
count=0
if [[ -f "$count_file" ]]; then read -r count <"$count_file"; fi
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
printf '%s\n' "$*" >>"$MOCK_SLURM_DIR/sbatch.calls"
if [[ "$count" -eq "$MOCK_FAILED_CALL" ]]; then exit 42; fi
printf '%s;testcluster\n' "$((100 + count))"
""",
        "scancel": """#!/usr/bin/env bash
printf '%s\n' "$*" >"$MOCK_SLURM_DIR/scancel.args"
""",
        "scontrol": """#!/usr/bin/env bash
printf '%s\n' "$*" >"$MOCK_SLURM_DIR/scontrol.args"
""",
    }
    for name, content in scripts.items():
        executable = mock_bin / name
        executable.write_text(content, encoding="utf-8")
        executable.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{mock_bin}:{environment['PATH']}"
    environment["MOCK_SLURM_DIR"] = str(state)
    environment["MOCK_FAILED_CALL"] = str(failed_call)
    output_dir = tmp_path / "submitted-run"
    completed = subprocess.run(
        [
            "validation/full_snv_concordance/slurm/run_map_reduce.sh",
            "--kind",
            "concordance",
            "--pairs-file",
            str(pairs),
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
            "--submit",
        ],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 42
    registered = [str(100 + index) for index in range(1, failed_call)]
    if registered:
        assert (state / "scancel.args").read_text(encoding="utf-8").strip() == " ".join(registered)
    else:
        assert not (state / "scancel.args").exists()
    assert not (state / "scontrol.args").exists()
    calls = (state / "sbatch.calls").read_text(encoding="utf-8")
    if failed_call > 1:
        assert "--dependency=afterok:101" in calls
    if failed_call > 2:
        assert "--dependency=afterok:102" in calls
    status = (output_dir / "submission_status.tsv").read_text(encoding="utf-8")
    assert "status\tfailed" in status
    assert "submitted_jobs\t" + ",".join(registered) + "\n" in status
    assert "python\t" + os.path.abspath(sys.executable) + "\n" in (
        output_dir / "launch.tsv"
    ).read_text(encoding="utf-8")


def test_chain_liftover_maps_forward_and_reverse_complements(tmp_path):
    chain = tmp_path / "tiny.chain"
    chain.write_text(
        "chain 100 chr1 1000 + 100 200 chr1 1000 + 150 250 1\n"
        "100\n\n"
        "chain 90 chr2 1000 + 10 20 chr9 1000 - 30 40 2\n"
        "10\n",
        encoding="utf-8",
    )
    lifter = ChainLiftOver(chain)
    assert lifter.map_snv("chr1", 101, "A", "G") == ("chr1", 151, "A", "G")
    assert lifter.map_snv("2", 11, "A", "C") == ("chr9", 970, "T", "G")
    with pytest.raises(ValueError, match="liftover_unmapped"):
        lifter.map_snv("chr3", 1, "A", "G")


def test_harmonization_enforces_manifest_liftover_path_and_digest(tmp_path):
    table = tmp_path / "source.tsv"
    table.write_text(
        "chrom\tpos\tref\talt\tgene\tlabel\nchr1\t1\tA\tG\tGENE\t1\n",
        encoding="utf-8",
    )
    expected_chain = tmp_path / "expected.chain"
    expected_chain.write_text(
        "chain 1 chr1 10 + 0 10 chr1 10 + 0 10 1\n10\n", encoding="utf-8"
    )
    wrong_chain = tmp_path / "wrong.chain"
    wrong_chain.write_text(expected_chain.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "acquisition_provenance": {
                    "tiny": {
                        "liftover_chain": str(expected_chain),
                        "liftover_chain_sha256": __import__("hashlib").sha256(
                            expected_chain.read_bytes()
                        ).hexdigest(),
                    }
                },
                "datasets": [
                    {
                        "id": "tiny",
                        "title": "tiny",
                        "article_url": "https://example.invalid",
                        "coordinate_build": "GRCh37",
                        "local_path": str(table),
                        "provenance_key": "tiny",
                        "columns": {
                            "chrom": "chrom",
                            "pos": "pos",
                            "ref": "ref",
                            "alt": "alt",
                            "gene": "gene",
                            "label": "label",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "external"
    with pytest.raises(ValueError, match="path differs from manifest pin"):
        harmonize_external(manifest, output, liftover_chain=wrong_chain)
    assert not output.exists()

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["acquisition_provenance"]["tiny"]["liftover_chain_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="liftover chain checksum mismatch"):
        harmonize_external(manifest, output, liftover_chain=expected_chain)
    assert not output.exists()

    document["acquisition_provenance"]["tiny"]["liftover_chain_sha256"] = (
        __import__("hashlib").sha256(expected_chain.read_bytes()).hexdigest()
    )
    manifest.write_text(json.dumps(document), encoding="utf-8")
    sensitivity = harmonize_external(
        manifest,
        tmp_path / "incomplete-sensitivity",
        include_spliceai=False,
        allow_incomplete_sensitivity=True,
    )
    assert sensitivity["dataset_completeness"]["mode"] == "incomplete_sensitivity"
    assert sensitivity["sources"][0]["unprocessed"] == 1


def test_riepe_adapter_uses_published_cohort_labels_when_assets_exist():
    manifest_path = Path("validation/full_snv_concordance/external_sources.json")
    manifest = load_source_manifest(manifest_path)
    datasets = {dataset["id"]: dataset for dataset in manifest["datasets"]}
    expected = {
        "riepe_abca4_noncanonical": (71, 64),
        "riepe_abca4_deep_intronic": (81, 21),
        "riepe_mybpc3": (61, 34),
    }
    for identifier, (row_count, positive_count) in expected.items():
        dataset = datasets[identifier]
        path = Path(dataset["local_path"])
        if not path.exists():
            pytest.skip("persisted Riepe benchmark assets are unavailable")
        rows = [row for _source, row, _columns in _dataset_rows(dataset, path)]
        assert len(rows) == row_count
        positives = sum(
            row["label"] in {"splice_altering", "significant_affects"} for row in rows
        )
        assert positives == positive_count


def test_all_cli_subcommands_execute_on_tiny_inputs(tmp_path):
    source = tmp_path / "source.vcf"
    prediction = tmp_path / "prediction.vcf"
    write_vcf(source, [row(1, ann("G", (0.4, 0, 0, 0)))], SOURCE_HEADER)
    write_vcf(
        prediction,
        [row(1, ann("G", (0.4, 0, 0, 0)), ann("G", (0.5, 0, 0, 0)))],
    )
    pairs = tmp_path / "pairs.tsv"
    write_pairs(pairs, [{"chunk_id": "1", "source_vcf": source, "prediction_vcf": prediction}])

    audit_json = tmp_path / "audit.json"
    assert main(["audit", "--pairs-file", str(pairs), "--output", str(audit_json)]) == 0
    built = tmp_path / "built.tsv"
    seed_pairs = tmp_path / "seed_pairs.tsv"
    assert main(
        [
            "build-pairs",
            "--left-manifest",
            str(audit_json),
            "--concordance-output",
            str(built),
            "--right-manifest",
            str(audit_json),
            "--seeds-output",
            str(seed_pairs),
            "--require-left-count",
            "1",
            "--require-overlap-count",
            "1",
        ]
    ) == 0

    maps = tmp_path / "maps"
    maps.mkdir()
    assert main(
        [
            "map-concordance",
            "--pairs-file",
            str(built),
            "--output",
            str(maps / "map-0.json"),
            "--task-index",
            "0",
            "--chunks-per-task",
            "1",
            "--score-bins",
            "10",
        ]
    ) == 0
    summary = tmp_path / "summary.json"
    assert main(
        [
            "reduce-concordance",
            "--input-dir",
            str(maps),
            "--output",
            str(summary),
            "--pairs-file",
            str(built),
            "--expected-task-count",
            "1",
            "--expected-total-chunks",
            "1",
        ]
    ) == 0
    assert main(
        ["render-report", "--summary", str(summary), "--output-dir", str(tmp_path / "report")]
    ) == 0

    seed_maps = tmp_path / "seed_maps"
    seed_maps.mkdir()
    assert main(
        [
            "map-seeds",
            "--pairs-file",
            str(seed_pairs),
            "--output",
            str(seed_maps / "map-0.json"),
            "--task-index",
            "0",
            "--chunks-per-task",
            "1",
            "--score-bins",
            "10",
        ]
    ) == 0
    assert main(
        [
            "reduce-seeds",
            "--input-dir",
            str(seed_maps),
            "--output",
            str(tmp_path / "seed_summary.json"),
            "--pairs-file",
            str(seed_pairs),
            "--expected-task-count",
            "1",
            "--expected-total-chunks",
            "1",
        ]
    ) == 0

    table = tmp_path / "functional.tsv"
    table.write_text(
        "chrom\tpos\tref\talt\tgene\tlabel\nchr1\t1\tG\tA\tG\t1\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "external.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "tiny",
                        "title": "tiny",
                        "article_url": "https://example.invalid",
                        "coordinate_build": "GRCh38",
                        "local_path": str(table),
                        "columns": {
                            "chrom": "chrom",
                            "pos": "pos",
                            "ref": "ref",
                            "alt": "alt",
                            "gene": "gene",
                            "label": "label",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "validate-external",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "external"),
            "--no-spliceai",
        ]
    ) == 0
    validation_vcf = tmp_path / "external" / "validation_variants.hg38.vcf"
    assert "##contig=<ID=chr1>" in validation_vcf.read_text(encoding="utf-8")
    external_plan = json.loads(
        (tmp_path / "external" / "validation_plan.json").read_text(encoding="utf-8")
    )
    assert external_plan["dataset_completeness"]["mode"] == "complete"
    assert external_plan["dataset_completeness"]["configured_dataset_ids"] == ["tiny"]
    assert external_plan["sources"][0]["rows"] == 1
    assert external_plan["sources"][0]["accepted"] == 1
