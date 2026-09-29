import json

import pytest

from validation.full_snv_concordance.aggregate import (
    Aggregate,
    AnalysisConfig,
    derive_histogram_metrics,
)
from validation.full_snv_concordance.reporting import render_report
from validation.full_snv_concordance.vcf import Annotation, VariantGroup, VariantKey


def annotation(gene, scores, dps=(0, 0, 0, 0)):
    return Annotation("A", gene, tuple(scores), tuple(dps))


def group(pos, left_annotations, right_annotations, ref="G", alt="A"):
    result = VariantGroup(VariantKey("chr1", pos, ref, alt), rows=1)
    for value in left_annotations:
        result.add("left", value)
    for value in right_annotations:
        result.add("right", value)
    return result


def compact_config(**kwargs):
    values = {
        "score_bins": 20,
        "sample_size": 0,
        "top_k": 0,
        "bootstrap_replicates": 30,
        "bootstrap_seed": 17,
    }
    values.update(kwargs)
    return AnalysisConfig(**values)


def test_rounding_signal_subsets_variant_collapse_and_strata():
    aggregate = Aggregate(compact_config())
    aggregate.add_group(
        group(
            1_000_001,
            [
                annotation("LEFT_ONLY", (0.8, 0.0, 0.0, 0.0), (4, 0, 0, 0)),
                annotation("COMMON", (0.12, 0.0, 0.0, 0.0)),
            ],
            [
                annotation("RIGHT_ONLY", (0.0, 0.7, 0.0, 0.0), (0, -3, 0, 0)),
                annotation("COMMON", (0.124, 0.0, 0.0, 0.0)),
            ],
        )
    )
    aggregate.add_group(group(1_000_002, [annotation("ZERO", (0, 0, 0, 0))], [annotation("ZERO", (0, 0, 0, 0))]))
    metrics = aggregate.derived()

    # Primary view remains exact gene matching: one COMMON and one ZERO pair.
    assert metrics["scores"]["AG"]["n"] == 2
    assert metrics["scores"]["AG"]["mae"] == pytest.approx(0.002)
    rounded = metrics["right_rounded_2dp"]["scores"]["AG"]
    assert rounded["mae"] == 0.0
    assert rounded["exact_match_rate"] == 1.0
    assert metrics["right_rounded_2dp"]["impact_vs_raw"]["AG"]["exact_match_rate_gain"] == pytest.approx(0.5)

    # Zero/zero pairs cannot inflate signal-only continuous agreement.
    assert metrics["signal_subsets"]["AG"]["either_gt_0"]["n"] == 1
    assert metrics["signal_subsets"]["AG"]["either_ge_0.1"]["n"] == 1
    assert metrics["signal_subsets"]["AG"]["either_ge_0.2"]["n"] == 0

    # The annotation-agnostic view pairs once per allele despite different genes.
    collapsed = metrics["variant_collapsed_view"]
    assert collapsed["scores"]["MAX"]["n"] == 2
    assert collapsed["thresholds"]["MAX"]["0.5"]["both_positive"] == 1
    assert aggregate.coverage["variant_collapsed_pairs"] == 2

    assert "chr1:1000001-2000000" in aggregate.strata["block_1mb"]
    assert "G>A" in aggregate.strata["substitution"]
    assert "AG>AG" in aggregate.strata["dominant_pair"]
    assert set(aggregate.strata["site_event"]) == {"AG", "AL", "DG", "DL"}


def test_histogram_distribution_metrics_are_well_defined():
    identical = derive_histogram_metrics(
        [1, 2, 1],
        [1, 2, 1],
        [1, 0, 0, 0, 2, 0, 0, 0, 1],
    )
    assert identical["ks_distance_binned"] == 0.0
    assert identical["wasserstein_1_binned"] == 0.0
    assert identical["jensen_shannon_divergence_base2"] == 0.0
    assert identical["spearman_r_binned"] == pytest.approx(1.0)

    reversed_order = derive_histogram_metrics(
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
    )
    assert reversed_order["spearman_r_binned"] == pytest.approx(-1.0)


def test_all_new_state_is_mergeable_serializable_and_bootstrap_deterministic():
    config = compact_config(thresholds=(0.1, 0.5))
    groups = [
        group(
            position,
            [annotation(f"G{position % 3}", (left, 0.0, 0.0, 0.0))],
            [annotation(f"G{position % 3}", (right, 0.0, 0.0, 0.0))],
        )
        for position, left, right in (
            (1, 0.125, 0.125),
            (2, 0.25, 0.375),
            (1_000_001, 0.5, 0.625),
            (2_000_001, 0.75, 0.875),
        )
    ]
    direct = Aggregate(config)
    first = Aggregate(config)
    second = Aggregate(config)
    for value in groups:
        direct.add_group(value)
    for value in groups[:2]:
        first.add_group(value)
    for value in groups[2:]:
        second.add_group(value)
    first.merge(second)

    # Every additive field, including collapsed/rounded/subset/strata state, merges.
    assert first.to_dict() == direct.to_dict()
    restored = Aggregate.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored.to_dict() == first.to_dict()

    direct_intervals = direct.derived()["cluster_bootstrap_95ci"]
    merged_intervals = first.derived()["cluster_bootstrap_95ci"]
    restored_intervals = restored.derived()["cluster_bootstrap_95ci"]
    assert direct_intervals == merged_intervals == restored_intervals
    for dimension in ("block_1mb", "gene"):
        values = direct_intervals["dimensions"][dimension]
        assert values["cluster_count"] > 0
        assert values["metrics"]["mae"]["valid_replicates"] == 30

    stale = first.to_dict()
    stale["schema_version"] = 1
    with pytest.raises(ValueError, match="rerun mappers"):
        Aggregate.from_dict(stale)


def test_detailed_report_renders_all_bounded_density_figures(tmp_path):
    pytest.importorskip("matplotlib")
    aggregate = Aggregate(compact_config(score_bins=10, bootstrap_replicates=2))
    aggregate.add_group(
        group(
            1,
            [annotation("G", (0.5, 0.2, 0.0, 0.0), (1, -2, 0, 0))],
            [annotation("G", (0.55, 0.1, 0.0, 0.0), (2, -2, 0, 0))],
        )
    )
    summary = {
        "kind": "reduced-concordance",
        "run_labels": ["provisional"],
        "left_label": "SpliceAI",
        "right_label": "OpenSpliceAI",
        "raw": aggregate.to_dict(),
        "metrics": aggregate.derived(),
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report_path = render_report(summary_path, tmp_path / "report")
    text = report_path.read_text(encoding="utf-8")
    assert "Analysis status: PROVISIONAL" in text
    assert "Union-signal MAX subsets" in text
    assert "Annotation-agnostic per-variant" in text
    expected = {
        "score_distributions.png",
        "joint_score_heatmaps.png",
        "difference_distributions.png",
        "threshold_agreement.png",
        "dp_agreement.png",
        "dominant_event_agreement.png",
        "chrom_gene_block_summaries.png",
        "quantization_sensitivity.png",
    }
    assert expected <= {path.name for path in (tmp_path / "report").glob("*.png")}
    html_path = tmp_path / "report" / "report.html"
    html_text = html_path.read_text(encoding="utf-8")
    assert "Analysis finality: <strong>PROVISIONAL</strong>" in html_text
    assert "data:image/png;base64," in html_text
