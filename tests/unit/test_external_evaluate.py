import csv
import json
from pathlib import Path

import pytest
import validation.full_snv_concordance.external_evaluate as external_module

from validation.full_snv_concordance.cli import main
from validation.full_snv_concordance.external_evaluate import (
    _auroc,
    _average_precision,
    _confusion,
    evaluate_external,
    join_predictions,
    load_predictor,
    normalize_functional_label,
    read_harmonized,
)
from validation.full_snv_concordance.workflows import sha256_file


HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI">
##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


def annotation(gene, scores, dps=(0, 0, 0, 0)):
    return "A|{}|{}|{}".format(
        gene,
        "|".join(str(value) for value in scores),
        "|".join(str(value) for value in dps),
    )


def write_vcf(path: Path, rows):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")


def vcf_row(pos, spliceai=".", openspliceai="."):
    return (
        f"chr1\t{pos}\t.\tG\tA\t.\t.\t"
        f"SpliceAI={spliceai};OpenSpliceAI={openspliceai}\n"
    )


def write_harmonized(path: Path, rows):
    fields = (
        "dataset",
        "source_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "transcript",
        "label",
        "outcome",
        "cohort",
        "reason",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def harmonized_row(dataset, source_row, pos, gene, label, cohort="assay"):
    return {
        "dataset": dataset,
        "source_row": source_row,
        "chrom": "chr1",
        "pos": pos,
        "ref": "G",
        "alt": "A",
        "gene": gene,
        "transcript": "TX",
        "label": label,
        "outcome": "measured",
        "cohort": cohort,
        "reason": "accepted",
    }


def test_dataset_specific_label_normalization_is_explicit():
    assert normalize_functional_label("smith_kitzman_multiplexed", "SDV") == (
        1,
        "smith_sdv_vs_neutral",
    )
    assert normalize_functional_label("smith_kitzman_multiplexed", "Neutral")[0] == 0
    assert normalize_functional_label("riepe_abca4_noncanonical", "splice_altering")[0] == 1
    assert normalize_functional_label("riepe_abca4_noncanonical", "neutral")[0] == 0
    assert normalize_functional_label("spip_rna_minigene", "1")[0] == 1
    assert normalize_functional_label("spip_rna_minigene", "0")[0] == 0
    # No generic truthiness or cross-dataset label leakage.
    assert normalize_functional_label("smith_kitzman_multiplexed", "True")[0] is None
    assert normalize_functional_label("spip_rna_minigene", "SDV")[0] is None
    assert normalize_functional_label("unknown", "1") == (None, "unsupported_dataset")


def test_exact_join_reports_duplicate_conflict_gene_and_variant_missingness(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    write_harmonized(
        harmonized,
        [
            harmonized_row("smith_test", "1", 10, "GOOD", "SDV"),
            harmonized_row("smith_test", "2", 20, "CONFLICT", "Neutral"),
            harmonized_row("smith_test", "3", 30, "EXPECTED", "SDV"),
            harmonized_row("smith_test", "4", 40, "ABSENT", "Neutral"),
        ],
    )
    scored = tmp_path / "scored.vcf"
    good = annotation("GOOD", (0.1, 0.8, 0.2, 0.0))
    first_conflict = annotation("CONFLICT", (0.3, 0, 0, 0))
    second_conflict = annotation("CONFLICT", (0.4, 0, 0, 0))
    write_vcf(
        scored,
        [
            # Exact repetition is a benign duplicate and max(DS) is 0.8.
            vcf_row(10, openspliceai=f"{good},{good}"),
            vcf_row(20, openspliceai=f"{first_conflict},{second_conflict}"),
            vcf_row(30, openspliceai=annotation("OTHER", (0.9, 0, 0, 0))),
        ],
    )
    observations, _ = read_harmonized(harmonized)
    predictor = load_predictor("rs10", scored)
    joined, coverage = join_predictions(observations, [predictor])
    assert predictor.info_key == "OpenSpliceAI"
    assert predictor.identical_duplicates == 1
    assert predictor.conflicts == 1
    assert [(row.status, row.score) for row in joined["rs10"]] == [
        ("matched", 0.8),
        ("prediction_conflict", None),
        ("missing_exact_gene", None),
        ("missing_variant", None),
    ]
    assert coverage["rs10"]["join_status"] == {
        "matched": 1,
        "missing_exact_gene": 1,
        "missing_variant": 1,
        "prediction_conflict": 1,
    }


def test_ambiguous_dual_info_requires_explicit_selection(tmp_path):
    scored = tmp_path / "both.vcf"
    write_vcf(
        scored,
        [
            vcf_row(
                1,
                spliceai=annotation("G", (0.2, 0, 0, 0)),
                openspliceai=annotation("G", (0.7, 0, 0, 0)),
            )
        ],
    )
    with pytest.raises(ValueError, match="provide an explicit info key"):
        load_predictor("mystery", scored)
    values = next(iter(load_predictor("mystery", scored, "SpliceAI").values.values()))
    assert max(next(iter(values)).scores) == 0.2


def test_predictor_parse_is_bound_to_expected_receipt_bytes(tmp_path):
    scored = tmp_path / "score.vcf"
    write_vcf(scored, [vcf_row(1, spliceai=annotation("G", (0.2, 0, 0, 0)))])
    expected = sha256_file(scored)
    scored.write_text(
        scored.read_text(encoding="utf-8").replace("0.2|0|0|0", "0.9|0|0|0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parsed score bytes differ from receipt"):
        load_predictor(
            "SpliceAI", scored, "SpliceAI", expected_sha256=expected, expected_records=1
        )


def test_hand_calculated_metrics_ties_and_threshold_rule():
    # One tied positive/negative pair contributes one half to AUROC.
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.5, 0.5, 0.9]
    assert _auroc(labels, scores) == pytest.approx(0.875)
    assert _average_precision(labels, scores) == pytest.approx(5 / 6)
    confusion = _confusion(labels, scores, 0.5)
    assert confusion == {
        "tp": 2,
        "fp": 1,
        "tn": 1,
        "fn": 0,
        "sensitivity": 1.0,
        "specificity": 0.5,
        "ppv": pytest.approx(2 / 3),
        "npv": 1.0,
        "f1": 0.8,
        "mcc": pytest.approx(1 / (3 ** 0.5)),
        "balanced_accuracy": 0.75,
    }


def test_end_to_end_outputs_bootstrap_and_paired_delta_are_deterministic(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    datasets = [
        ("smith_test", "SDV", "S1"),
        ("smith_test", "Neutral", "S1"),
        ("riepe_test", "splice_altering", "R1"),
        ("riepe_test", "neutral", "R1"),
        ("spip_test", "1", "P1"),
        ("spip_test", "0", "P1"),
    ]
    write_harmonized(
        harmonized,
        [
            harmonized_row(dataset, str(index), index, "G", label, cohort)
            for index, (dataset, label, cohort) in enumerate(datasets, 1)
        ],
    )
    left = tmp_path / "left.vcf"
    right = tmp_path / "right.vcf"
    left_scores = (0.9, 0.1, 0.8, 0.2, 0.7, 0.3)
    right_scores = (0.6, 0.4, 0.4, 0.6, 0.55, 0.45)
    write_vcf(
        left,
        [vcf_row(index, spliceai=annotation("G", (score, 0, 0, 0))) for index, score in enumerate(left_scores, 1)],
    )
    write_vcf(
        right,
        [vcf_row(index, openspliceai=annotation("G", (score, 0, 0, 0))) for index, score in enumerate(right_scores, 1)],
    )
    validation_plan = tmp_path / "validation_plan.json"
    source_statuses = [
        {
            "id": dataset,
            "state": "ready",
            "rows": 2,
            "accepted": 2,
            "rejected": 0,
        }
        for dataset in ("smith_test", "riepe_test", "spip_test")
    ]
    validation_plan.write_text(
        json.dumps(
            {
                "kind": "external-validation-preparation",
                "sources": source_statuses,
                "dataset_completeness": {
                    "mode": "complete",
                    "complete": True,
                    "configured_dataset_ids": [row["id"] for row in source_statuses],
                    "incomplete_datasets": [],
                },
            }
        ),
        encoding="utf-8",
    )

    first = evaluate_external(
        harmonized,
        {"SpliceAI": left, "rs10": right},
        tmp_path / "first",
        thresholds=(0.5,),
        bootstrap_replicates=40,
        bootstrap_seed=17,
        render_plots=True,
        validation_plan=validation_plan,
    )
    second = evaluate_external(
        harmonized,
        {"SpliceAI": left, "rs10": right},
        tmp_path / "second",
        thresholds=(0.5,),
        bootstrap_replicates=40,
        bootstrap_seed=17,
        render_plots=False,
        validation_plan=validation_plan,
    )
    first_groups = [{k: v for k, v in row.items() if k != "heterogeneous_pool_warning"} for row in first["groups"]]
    second_groups = [{k: v for k, v in row.items() if k != "heterogeneous_pool_warning"} for row in second["groups"]]
    assert first_groups == second_groups
    assert first["paired_delta_auroc"] == second["paired_delta_auroc"]
    assert first["plot_error"] is None
    assert {Path(plot["path"]).name for plot in first["plots"]} == {
        "discrimination.png",
        "pooled_curves.png",
    }
    assert all(Path(plot["path"]).is_file() for plot in first["plots"])
    pooled_left = next(
        row
        for row in first["groups"]
        if row["scope"] == "pooled" and row["predictor"] == "SpliceAI"
    )
    assert pooled_left["metrics"]["auroc"]["estimate"] == 1.0
    assert pooled_left["metrics"]["thresholds"]["0.5"]["tp"] == 3
    paired = next(row for row in first["paired_delta_auroc"] if row["scope"] == "pooled")
    assert paired["n_overlap"] == 6
    assert paired["delta_auroc"] > 0
    assert paired["bootstrap_valid"] == 40
    for filename in (
        "external_evaluation.json",
        "external_evaluation.md",
        "joined_predictions.tsv",
        "functional_metrics.tsv",
        "paired_delta_auroc.tsv",
    ):
        assert (tmp_path / "first" / filename).is_file()
    persisted = json.loads((tmp_path / "first" / "external_evaluation.json").read_text())
    assert "heterogeneous" in persisted["warnings"][0].lower()
    assert persisted["join_definition"] == "exact CHROM, POS, REF, ALT, and gene"
    assert persisted["dataset_provenance"]["state"] == "complete"
    assert persisted["coverage_policy"]["violations"] == []
    assert all(item["sha256"] for item in persisted["predictors"])

    with pytest.raises(FileExistsError):
        evaluate_external(
            harmonized,
            {"SpliceAI": left, "rs10": right},
            tmp_path / "first",
            bootstrap_replicates=0,
            render_plots=False,
            validation_plan=validation_plan,
        )

    cli_output = tmp_path / "cli"
    assert main(
        [
            "evaluate-external",
            "--harmonized",
            str(harmonized),
            "--score",
            f"SpliceAI={left}",
            "--score",
            f"rs10={right}",
            "--output-dir",
            str(cli_output),
            "--thresholds",
            "0.5",
            "--bootstrap-replicates",
            "2",
            "--no-plots",
        ]
    ) == 0
    assert (cli_output / "external_evaluation.json").is_file()


def test_pooled_unique_consensus_deduplicates_and_excludes_label_conflicts(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    write_harmonized(
        harmonized,
        [
            harmonized_row("smith_test", "s1", 1, "G", "SDV"),
            harmonized_row("spip_test", "p1", 1, "G", "1"),
            harmonized_row("smith_test", "s2", 2, "G", "SDV"),
            harmonized_row("spip_test", "p2", 2, "G", "0"),
            harmonized_row("smith_test", "s3", 3, "G", "Neutral"),
        ],
    )
    scored = tmp_path / "scored.vcf"
    write_vcf(
        scored,
        [
            vcf_row(pos, spliceai=annotation("G", (score, 0, 0, 0)))
            for pos, score in ((1, 0.9), (2, 0.8), (3, 0.1))
        ],
    )
    result = evaluate_external(
        harmonized,
        {"SpliceAI": scored},
        tmp_path / "result",
        bootstrap_replicates=0,
        render_plots=False,
        score_mask=0,
    )
    overlap = result["cross_dataset_overlap"]
    assert overlap["observation_rows"] == 5
    assert overlap["unique_variant_gene_keys"] == 3
    assert overlap["repeated_variant_gene_keys"] == 2
    assert overlap["duplicate_observation_rows_beyond_first"] == 2
    assert overlap["discordant_binary_label_keys"] == 1
    assert overlap["pooled_unique_consensus_keys"] == 2
    consensus = next(
        row
        for row in result["groups"]
        if row["scope"] == "pooled_unique_consensus"
        and row["predictor"] == "SpliceAI"
    )
    assert consensus["n_scored"] == 2
    assert consensus["n_positive"] == 1
    assert result["score_mask"] == 0


def test_duplicate_harmonized_observation_id_fails_closed(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    row = harmonized_row("smith_test", "same", 1, "G", "SDV")
    write_harmonized(harmonized, [row, row])
    with pytest.raises(ValueError, match="duplicate observation ID"):
        read_harmonized(harmonized)


def test_final_evaluation_rejects_sparse_prediction_coverage_before_writing(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    write_harmonized(
        harmonized,
        [
            harmonized_row(
                "smith_test", str(index), index, "G", "SDV" if index % 2 else "Neutral"
            )
            for index in range(1, 11)
        ],
    )
    scored = tmp_path / "sparse.vcf"
    write_vcf(scored, [vcf_row(1, spliceai=annotation("G", (0.9, 0, 0, 0)))])
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="prediction coverage is below"):
        evaluate_external(
            harmonized,
            {"SpliceAI": scored},
            output,
            bootstrap_replicates=0,
            render_plots=False,
            minimum_prediction_coverage=0.9,
        )
    assert not output.exists()


def test_coverage_gate_cannot_hide_a_missing_small_cohort(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    write_harmonized(
        harmonized,
        [
            harmonized_row(
                "smith_test",
                str(index),
                index,
                "G",
                "SDV" if index % 2 else "Neutral",
                cohort="large" if index < 10 else "small",
            )
            for index in range(1, 11)
        ],
    )
    scored = tmp_path / "score.vcf"
    write_vcf(
        scored,
        [
            vcf_row(index, spliceai=annotation("G", (0.9, 0, 0, 0)))
            for index in range(1, 10)
        ],
    )
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=r"dataset_cohort:smith_test:small=0/1"):
        evaluate_external(
            harmonized,
            {"SpliceAI": scored},
            output,
            bootstrap_replicates=0,
            render_plots=False,
            minimum_prediction_coverage=0.9,
        )
    assert not output.exists()


def test_zero_valid_label_stratum_and_partial_output_fail_closed(tmp_path, monkeypatch):
    invalid = tmp_path / "invalid.tsv"
    write_harmonized(
        invalid,
        [harmonized_row("smith_test", "1", 1, "G", "not-a-smith-label")],
    )
    scored = tmp_path / "score.vcf"
    write_vcf(scored, [vcf_row(1, spliceai=annotation("G", (0.9, 0, 0, 0)))])
    with pytest.raises(ValueError, match="no valid binary labels"):
        evaluate_external(
            invalid,
            {"SpliceAI": scored},
            tmp_path / "invalid-result",
            bootstrap_replicates=0,
            render_plots=False,
        )

    valid = tmp_path / "valid.tsv"
    write_harmonized(valid, [harmonized_row("smith_test", "1", 1, "G", "SDV")])
    final = tmp_path / "retryable-result"
    original = external_module._atomic_tsv
    calls = 0

    def fail_during_publication(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(external_module, "_atomic_tsv", fail_during_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        evaluate_external(
            valid,
            {"SpliceAI": scored},
            final,
            bootstrap_replicates=0,
            render_plots=False,
        )
    assert not final.exists()
    monkeypatch.setattr(external_module, "_atomic_tsv", original)
    result = evaluate_external(
        valid,
        {"SpliceAI": scored},
        final,
        bootstrap_replicates=0,
        render_plots=False,
    )
    assert Path(result["outputs"]["json"]).is_file()
