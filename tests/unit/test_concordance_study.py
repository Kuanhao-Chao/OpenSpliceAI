"""Tests for the cross-run synthesis package.

The load-bearing test is `test_operating_point_transfer_reproduces_reducer_counts`:
the transfer analysis reads confusion counts off the additive joint histogram,
while the reducer counts the same events directly. They must agree exactly, or the
derived operating points are not trustworthy.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import re
from pathlib import Path

import pytest

from validation.concordance_study import derive, report
from validation.concordance_study.loading import ContractError, Run, load_run
from validation.full_snv_concordance.aggregate import AnalysisConfig, Aggregate
from validation.full_snv_concordance.vcf import Annotation, VariantGroup, VariantKey

THRESHOLDS = (0.05, 0.1, 0.2, 0.5, 0.8)


def _scores(rng: random.Random) -> tuple:
    """A mixture that mimics the real distribution: mostly zero, a heavy low tail."""
    draw = rng.random()
    if draw < 0.55:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(round(max(0.0, rng.betavariate(0.35, 4.0)), 5) for _ in range(4))


def _build_aggregate(n: int = 4000, seed: int = 7) -> Aggregate:
    rng = random.Random(seed)
    config = AnalysisConfig(thresholds=THRESHOLDS, score_bins=100, sample_size=50, top_k=25)
    aggregate = Aggregate(config)
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    for i in range(n):
        gene = genes[i % len(genes)]
        key = VariantKey("chr1", 1000 + i, "G", "ACT"[i % 3])
        group = VariantGroup(key, rows=1)
        left = _scores(rng)
        # right is a perturbed, two-decimal-quantised view of left, plus its own draws
        right = tuple(round(min(1.0, max(0.0, value + rng.gauss(0, 0.05))), 5) for value in left)
        group.add("left", Annotation(key.alt, gene, tuple(round(v, 2) for v in left),
                                     (1, 2, 3, 4)))
        group.add("right", Annotation(key.alt, gene, right, (1, 2, 5, 4)))
        aggregate.add_group(group)
    return aggregate


def _summary_from(aggregate: Aggregate, *, kind: str = "concordance", chunks: int = 3,
                  left: str = "SpliceAI", right: str = "OpenSpliceAI") -> dict:
    return {
        "schema_version": 2,
        "kind": kind,
        "created_at": "2026-09-07T00:00:00+00:00",
        "left_label": left,
        "right_label": right,
        "mapper_run_labels": ["unit-test"],
        "expected_task_count": 1,
        "expected_total_chunks": 100000,
        "finality": {
            "status": "provisional",
            "contract": "explicit-reducer-finality-v1",
            "policy": "partial_coverage_allowed",
            "observed_pair_count": chunks,
            "pair_id_domain_complete": False,
            "unselected_chunk_count": 100000 - chunks,
            "unexpected_chunk_ids": [],
            "excluded_incomplete_edge_fragments": 0,
        },
        "verified_provenance": {
            "status": "verified",
            "contract": "audited-vcf-snapshot-v1",
            "verified_chunk_count": chunks,
            "output_provenance_class_counts": {"legacy_unprovenanced": chunks},
        },
        "raw": aggregate.to_dict(),
        "metrics": aggregate.derived(),
    }


@pytest.fixture(scope="module")
def concordance_run(tmp_path_factory) -> Run:
    path = tmp_path_factory.mktemp("study") / "summary.json"
    path.write_text(json.dumps(_summary_from(_build_aggregate())))
    return load_run(path, arm="A", role="primary")


# --------------------------------------------------------------------------
def test_operating_point_transfer_reproduces_reducer_counts(concordance_run: Run) -> None:
    for label in ("AG", "AL", "DG", "DL", "MAX"):
        for row in derive.operating_point_transfer(concordance_run, label=label):
            mine = row["identical_cutoff"]
            theirs = concordance_run.thresholds(label, row["spliceai_threshold"])
            for field in ("both_positive", "left_only", "right_only", "both_negative"):
                assert mine[field] == pytest.approx(theirs[field], abs=1e-9), (label, field)
            for field in ("jaccard", "kappa", "mcc", "overall_agreement"):
                if theirs[field] == theirs[field]:  # skip NaN
                    assert mine[field] == pytest.approx(theirs[field], abs=1e-9), (label, field)


def _positive_rate_at(run: Run, cutoff: float, label: str = "MAX") -> float:
    """Independent recomputation of the OpenSpliceAI positive rate from the marginal."""
    hist = run.score_hist(label)["right"]
    bins = run.score_bins
    index = int(round(cutoff * bins))
    return sum(hist[index:]) / sum(hist)


def test_rate_matched_cutoff_is_the_closest_on_the_grid(concordance_run: Run) -> None:
    bins = concordance_run.score_bins
    for row in derive.operating_point_transfer(concordance_run):
        matched = row["rate_matched"]
        target = row["spliceai_positive_rate"]
        assert 0.0 <= matched["right_cutoff"] <= 1.0
        # cross-check the chosen rate against the independent marginal histogram
        assert matched["right_positive_rate"] == pytest.approx(
            _positive_rate_at(concordance_run, matched["right_cutoff"]), abs=1e-12)
        best = min(abs(_positive_rate_at(concordance_run, b / bins) - target) for b in range(bins))
        assert abs(matched["right_positive_rate"] - target) <= best + 1e-12


def test_agreement_optimal_cutoff_is_the_argmax(concordance_run: Run) -> None:
    for row in derive.operating_point_transfer(concordance_run):
        curve = derive.transfer_curve(concordance_run, row["spliceai_threshold"])
        defined = [value for value in curve["mcc"] if value is not None]
        assert defined, "no cutoff produced a defined MCC"
        assert row["agreement_optimal"]["mcc"] == pytest.approx(max(defined), abs=1e-12)


def test_undefined_metrics_are_null_not_nan(concordance_run: Run) -> None:
    """Undefined statistics must serialise as JSON null, exactly as the reducer does."""
    for row in derive.operating_point_transfer(concordance_run):
        for variant in ("identical_cutoff", "rate_matched", "agreement_optimal"):
            for key, value in row[variant].items():
                assert not (isinstance(value, float) and value != value), (variant, key)
        json.dumps(row)  # would raise on a NaN under strict mode below
    payload = json.dumps(derive.operating_point_transfer(concordance_run), allow_nan=False)
    assert "NaN" not in payload


def test_quantization_share_is_bounded(concordance_run: Run) -> None:
    for row in derive.quantization_profile(concordance_run):
        share = row["share_of_mae_beyond_quantization"]
        assert 0.0 <= share <= 1.0
        assert row["rounded_exact_match_rate"] >= row["raw_exact_match_rate"]


def test_stratum_dispersion_is_size_weighted() -> None:
    rows = [{"mae": 0.0, "n": 1}, {"mae": 1.0, "n": 99}]
    stats = derive.stratum_dispersion(rows, "mae")
    assert stats["weighted_mean"] == pytest.approx(0.99)
    assert stats["median"] == pytest.approx(1.0)
    assert stats["min"] == 0.0 and stats["max"] == 1.0


def test_seed_versus_model_ratio(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    model_path = tmp_path / "model.json"
    seed_summary = _summary_from(
        _build_aggregate(seed=1), kind="seeds",
        left="OpenSpliceAI rs10", right="OpenSpliceAI rs13",
    )
    model_summary = _summary_from(_build_aggregate(seed=2))
    for summary in (seed_summary, model_summary):
        summary["matched_domain"] = {"digest": "same-three-way-pairs", "n": 4000}
    seed_path.write_text(json.dumps(seed_summary))
    model_path.write_text(json.dumps(model_summary))
    seed = load_run(seed_path, arm="B", role="seed")
    model = load_run(model_path, arm="C", role="model")
    result = derive.seed_versus_model(seed, [model])
    ratio = result["error_ratios_model_over_seed"]["C"]["mae"]
    assert ratio == pytest.approx(model.scores("MAX")["mae"] / seed.scores("MAX")["mae"])


def test_seed_versus_model_requires_shared_matched_domain(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("seed.json", "model.json")]
    summaries = [
        _summary_from(_build_aggregate(seed=1), kind="seeds",
                      left="OpenSpliceAI rs10", right="OpenSpliceAI rs13"),
        _summary_from(_build_aggregate(seed=2)),
    ]
    for path, summary in zip(paths, summaries):
        path.write_text(json.dumps(summary))
    seed = load_run(paths[0], arm="B", role="seed")
    model = load_run(paths[1], arm="C", role="model")

    with pytest.raises(ValueError, match="matched-domain metadata"):
        derive.seed_versus_model(seed, [model])


# --------------------------------------------------------------------------
def test_contract_rejects_unverified_and_mislabelled(tmp_path: Path) -> None:
    payload = _summary_from(_build_aggregate(n=200))
    bad = tmp_path / "bad.json"

    payload["finality"]["status"] = "definitely-final"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="unknown finality status"):
        load_run(bad, arm="X", role="primary")

    payload["finality"]["status"] = "final"          # claims final without the domain
    bad.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="complete pair-id domain"):
        load_run(bad, arm="X", role="primary")

    payload["finality"]["status"] = "provisional"
    payload["verified_provenance"]["status"] = "unverified"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="not verified"):
        load_run(bad, arm="X", role="primary")

    payload["verified_provenance"]["status"] = "verified"
    payload["verified_provenance"]["verified_chunk_count"] = 2
    bad.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="verified chunks"):
        load_run(bad, arm="X", role="primary")


def test_contract_rejects_wrong_chunk_count(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_summary_from(_build_aggregate(n=200), chunks=3)))
    with pytest.raises(ContractError, match="expected 99"):
        load_run(path, arm="X", role="primary", expected_chunks=99)


# --------------------------------------------------------------------------
def test_placeholders_resolve_and_fail_closed() -> None:
    facts = {"primary": {"chunks": 99283, "agreement": {"MAX": {"mae": 0.0123456}}}}
    assert report.resolve("{{primary.chunks|,d}}", facts) == "99,283"
    assert report.resolve("{{primary.agreement.MAX.mae|.4f}}", facts) == "0.0123"
    assert report.resolve("{{primary.agreement.MAX.mae|pct2}}", facts) == "1.23%"
    with pytest.raises(KeyError, match="unresolved placeholders"):
        report.resolve("{{primary.nope}}", facts)
    with pytest.raises(KeyError, match="unresolved placeholders"):
        report.resolve("{{primary.agreement.MAX.typo|.2f}}", facts)


def test_undefined_placeholder_metric_renders_na() -> None:
    assert report.resolve("{{metric|pct2}}", {"metric": None}) == "n/a"


def test_report_template_has_no_literal_numbers_left(concordance_run: Run) -> None:
    facts = report.build_facts({"A": concordance_run}, primary="A", seed_arm=None,
                               model_arms=[], study_meta={"name": "unit"})
    text = report.resolve("pairs={{primary.agreement.MAX.n|,d}} mae={{primary.agreement.MAX.mae|.5f}}",
                          facts)
    assert "{{" not in text and "}}" not in text
    assert str(concordance_run.paired_n()) in text.replace(",", "")


def test_html_is_self_contained(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    png = figures_dir / "f01_coverage.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    markdown_text = '# T\n\n<figure><img src="figures/f01_coverage.png" alt="c"></figure>\n'
    html = report.render_html(markdown_text, figures_dir, "Title")
    assert "data:image/png;base64," in html
    assert 'src="figures/' not in html
    # Every image is inlined and no script or stylesheet is fetched from a third party.
    # The one permitted external reference is the Google Fonts stylesheet, which the
    # artifact CSP allows and which degrades to the declared fallback stack offline.
    assert "<script" not in html.lower()
    external = re.findall(r'https?://[^\s"\')]+', html)
    assert external, "expected the font import to be present"
    assert all(url.startswith("https://fonts.googleapis.com/") for url in external), external
    assert 'src="http' not in html and "//cdn" not in html


def test_tables_written_for_every_arm(concordance_run: Run, tmp_path: Path) -> None:
    written = report.write_tables({"A": concordance_run}, tmp_path)
    assert any(name.endswith("agreement.csv") for name in written)
    assert any(name.endswith("operating_point_transfer.csv") for name in written)
    content = (tmp_path / "A" / "agreement.csv").read_text().splitlines()
    assert content[0].startswith("label,left_label,right_label,n,")
    assert len(content) == 6  # header + AG/AL/DG/DL/MAX


def test_configured_point_zero_five_is_in_threshold_and_signal_tables(
        concordance_run: Run) -> None:
    facts = report.build_facts(
        {"A": concordance_run}, primary="A", seed_arm=None, model_arms=[], study_meta={}
    )
    thresholds = report.resolve("{{table:thresholds_max}}", facts)
    subsets = report.resolve("{{table:signal_subsets}}", facts)

    assert "| 0.05 |" in thresholds
    assert "either score ≥ 0.05" in subsets


def test_generic_seed_exports_use_run_labels_not_tool_aliases(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(_summary_from(
        _build_aggregate(n=200), kind="seeds",
        left="OpenSpliceAI rs10", right="OpenSpliceAI rs13",
    )))
    seed = load_run(path, arm="B", role="seed")

    agreement = derive.agreement_table(seed)[0]
    stratum = derive.stratum_rows(seed, "chrom")[0]
    for row in (agreement, stratum):
        assert row["left_label"] == "OpenSpliceAI rs10"
        assert row["right_label"] == "OpenSpliceAI rs13"
        assert "mean_left" in row and "mean_right" in row
        assert "mean_spliceai" not in row and "mean_openspliceai" not in row


def test_operating_point_table_preserves_point_zero_zero_five_grid() -> None:
    facts = {
        "primary": {
            "operating_point": {
                "0.05": {
                    "spliceai_positive_calls": 8,
                    "identical_cutoff": {"both_positive": 4, "right_only": 2, "mcc": None},
                    "rate_matched": {"right_cutoff": 0.005, "mcc": None},
                    "agreement_optimal": {"right_cutoff": 0.015, "mcc": None},
                }
            }
        }
    }
    rendered = report.resolve("{{table:operating_point}}", facts)
    assert "| 0.005 |" in rendered
    assert "| 0.015 |" in rendered
    assert "n/a" in rendered


def test_undefined_dp_rate_renders_na_not_zero(concordance_run: Run) -> None:
    facts = report.build_facts(
        {"A": concordance_run}, primary="A", seed_arm=None, model_arms=[], study_meta={}
    )
    facts["primary"]["dp"]["AG"]["0.5"]["within_0bp"] = None
    rendered = report.resolve("{{table:dp_agreement}}", facts)
    ag_row = next(line for line in rendered.splitlines() if "acceptor gain" in line)
    assert "n/a" in ag_row
    assert "0.00%" not in [cell.strip() for cell in ag_row.split("|")]


def test_facts_file_is_strict_json(concordance_run: Run, tmp_path: Path) -> None:
    """No NaN or Infinity may reach study_facts.json; undefined becomes null."""
    facts = report.build_facts({"A": concordance_run}, primary="A", seed_arm=None,
                               model_arms=[], study_meta={})
    path = tmp_path / "study_facts.json"
    report.write_facts(facts, path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict parse


def test_table_directives_render_and_fail_closed(concordance_run: Run) -> None:
    facts = report.build_facts({"A": concordance_run}, primary="A", seed_arm=None,
                               model_arms=[], study_meta={})
    for name in ("runs_overview", "coverage", "agreement", "equivalence", "thresholds_max",
                 "thresholds_by_event", "operating_point", "quantization", "signal_subsets",
                 "dp_agreement", "dominant", "chromosomes", "top_genes", "substitutions",
                 "stratum_dispersion", "bootstrap", "provenance", "seed_versus_model",
                 "generalization", "tail_asymmetry", "discordance"):
        rendered = report.resolve("{{table:" + name + "}}", facts)
        assert rendered.strip(), name
        if not rendered.startswith("_"):          # the two "not available" placeholders
            assert rendered.count("\n") >= 2, name
            assert rendered.splitlines()[1].startswith("|"), name
    with pytest.raises(KeyError, match="unresolved placeholders"):
        report.resolve("{{table:no_such_table}}", facts)


def test_primary_namespaced_site_distance_table_is_rendered() -> None:
    facts = {
        "primary": {
            "site_distance": {
                "order": ["at_site"],
                "pooled": {
                    "at_site": {
                        "n": 10,
                        "left_call_rate": 0.2,
                        "right_call_rate": 0.3,
                        "call_rate_ratio": 1.5,
                        "jaccard": 0.25,
                        "mean_left": 0.1,
                        "mean_right": 0.12,
                    }
                },
            }
        }
    }
    rendered = report.resolve("{{table:site_distance}}", facts)
    assert "at_site" in rendered
    assert "20.0000%" in rendered and "30.0000%" in rendered


def test_event_strata_are_written_with_json_thresholds(
        concordance_run: Run, tmp_path: Path) -> None:
    summary = copy.deepcopy(concordance_run.summary)
    threshold_payload = {"0.05": {"both_positive": 2, "left_only": 1,
                                  "right_only": 3, "both_negative": 4}}
    metric_payload = {**concordance_run.scores("AG"), "n": 10,
                      "mean_left": 0.1, "mean_right": 0.2,
                      "bias_right_minus_left": 0.1, "mae": 0.1}
    dimensions = (
        "chrom", "gene", "block_1mb", "substitution", "dominant_pair", "site_distance",
    )
    summary["metrics"]["event_strata"] = {}
    for dimension in dimensions:
        key = "at_site:donor" if dimension == "site_distance" else "bucket"
        summary["metrics"]["event_strata"][dimension] = {
            key: {
                event: {"metrics": metric_payload, "thresholds": threshold_payload}
                for event in ("AG", "AL", "DG", "DL")
            }
        }
    summary["metrics"]["strata"]["site_distance"] = {
        "at_site:donor": {"metrics": metric_payload, "thresholds": threshold_payload}
    }
    run = Run(arm="A", role="primary", path=concordance_run.path, summary=summary)

    written = report.write_tables({"A": run}, tmp_path)
    for dimension in dimensions:
        relative = f"A/event_stratum_{dimension}.csv"
        assert relative in written
        with (tmp_path / relative).open(newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row["left_label"] == "SpliceAI"
        assert row["right_label"] == "OpenSpliceAI"
        assert row["event"] == "AG"
        assert json.loads(row["thresholds"]) == threshold_payload


# --------------------------------------------------------------------------
# The independent recomputation must agree with the package's own parser on
# identical input, including where a variant group straddles a chunk boundary.
# --------------------------------------------------------------------------
_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=100000>\n"
    '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI">\n'
    '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def _ann(alt, gene, scores):
    return f"{alt}|{gene}|" + "|".join(f"{v:.5f}" for v in scores) + "|1|2|3|4"


def _row(pos, alt, spliceai, openspliceai=None):
    info = f"SpliceAI={spliceai}"
    if openspliceai is not None:
        info += f";OpenSpliceAI={openspliceai}"
    return f"chr1\t{pos}\t.\tG\t{alt}\t.\t.\t{info}\n"


def _synthetic_chunks(tmp_path: Path) -> tuple:
    """Two chunk files whose boundary splits one variant group across them."""
    rng = random.Random(11)
    rows_a, rows_b = [], []
    for pos in range(100, 160):
        for alt in "ACT":
            left = tuple(round(rng.random() * 0.4, 2) for _ in range(4))
            right = tuple(round(min(1.0, max(0.0, v + rng.gauss(0, 0.05))), 5) for v in left)
            rows_a.append(_row(pos, alt, _ann(alt, "GENE1", left), _ann(alt, "GENE1", right)))
    # the straddling group: same (POS, REF, ALT) rows on both sides of the split
    straddle = []
    for alt in "ACT":
        left = (0.7, 0.1, 0.0, 0.0)
        right = (0.66, 0.12, 0.0, 0.0)
        straddle.append(_row(160, alt, _ann(alt, "GENE1", left), _ann(alt, "GENE1", right)))
    rows_a.append(straddle[0])
    rows_b.extend(straddle[1:])
    for pos in range(161, 220):
        for alt in "ACT":
            left = tuple(round(rng.random() * 0.4, 2) for _ in range(4))
            right = tuple(round(min(1.0, max(0.0, v + rng.gauss(0, 0.05))), 5) for v in left)
            rows_b.append(_row(pos, alt, _ann(alt, "GENE1", left), _ann(alt, "GENE1", right)))

    first, second = tmp_path / "chunk1.vcf", tmp_path / "chunk2.vcf"
    first.write_text(_VCF_HEADER + "".join(rows_a))
    second.write_text(_VCF_HEADER + "".join(rows_b))
    return first, second


def test_independent_recomputation_matches_package_parser(tmp_path: Path) -> None:
    from validation.concordance_study import crosscheck
    from validation.full_snv_concordance.vcf import iter_variant_groups

    first, second = _synthetic_chunks(tmp_path)
    mine = crosscheck.recompute([first, second], threshold=0.5)

    # The recomputation drops the window's first and last group by design, so the
    # package side is computed over the same interior set.
    groups = list(iter_variant_groups((first, second),
                                      {"left": "SpliceAI", "right": "OpenSpliceAI"}))
    aggregate = Aggregate(AnalysisConfig(thresholds=(0.5,), score_bins=100,
                                         sample_size=0, top_k=0))
    for group in groups[1:-1]:
        aggregate.add_group(group)
    theirs = aggregate.derived()

    assert mine["paired_annotations"] == theirs["coverage"]["paired_annotations"]
    assert mine["invalid_annotations"] == theirs["coverage"]["invalid_annotations"]
    assert mine["left_valid_annotations"] == theirs["coverage"]["left_valid_annotations"]
    assert mine["right_valid_annotations"] == theirs["coverage"]["right_valid_annotations"]
    for field in ("mean_left", "mean_right", "bias_right_minus_left", "mae", "rmse",
                  "exact_match_rate"):
        assert mine[field] == pytest.approx(theirs["scores"]["MAX"][field], abs=1e-12), field
    table = theirs["thresholds"]["MAX"]["0.5"]
    for field in ("both_positive", "left_only", "right_only", "both_negative"):
        assert mine[field] == table[field], field


def test_straddling_group_is_counted_once(tmp_path: Path) -> None:
    """A variant split across the chunk boundary must not be paired twice."""
    from validation.concordance_study import crosscheck
    from validation.full_snv_concordance.vcf import iter_variant_groups

    first, second = _synthetic_chunks(tmp_path)
    mine = crosscheck.recompute([first, second], threshold=0.5)
    groups = list(iter_variant_groups((first, second),
                                      {"left": "SpliceAI", "right": "OpenSpliceAI"}))
    # 60 + 1 + 59 distinct positions, one group each; two are dropped as window edges
    assert mine["variant_groups"] == len(groups)
    assert mine["paired_annotations"] == len(groups) - 2


def test_malformed_annotations_are_rejected_identically(tmp_path: Path) -> None:
    """Both implementations must discard the same bad annotations."""
    from validation.concordance_study import crosscheck
    from validation.full_snv_concordance.aggregate import AnalysisConfig, Aggregate
    from validation.full_snv_concordance.vcf import iter_variant_groups

    good = _ann("A", "GENE1", (0.4, 0.0, 0.0, 0.0))
    rows = [
        _row(10, "A", good, good),                                   # window edge, dropped
        _row(11, "A", good, "A|GENE1|1.5|0|0|0|1|2|3|4"),            # score above 1
        _row(12, "A", good, "A|GENE1|-0.3|0|0|0|1|2|3|4"),           # score below 0
        _row(13, "A", good, "A||0.2|0|0|0|1|2|3|4"),                 # empty gene
        _row(14, "A", good, "A|GENE1|0.2|0|0|1|2|3|4"),              # nine fields
        _row(15, "A", good, _ann("A", "GENE1", (0.3, 0.0, 0.0, 0.0))),
        _row(16, "A", good, good),                                   # window edge, dropped
    ]
    path = tmp_path / "chunk.vcf"
    path.write_text(_VCF_HEADER + "".join(rows))

    mine = crosscheck.recompute([path], threshold=0.5)
    groups = list(iter_variant_groups((path,), {"left": "SpliceAI", "right": "OpenSpliceAI"}))
    aggregate = Aggregate(AnalysisConfig(thresholds=(0.5,), score_bins=100,
                                         sample_size=0, top_k=0))
    for group in groups[1:-1]:
        aggregate.add_group(group)
    coverage = aggregate.derived()["coverage"]

    assert mine["invalid_annotations"] == coverage["invalid_annotations"] == 4
    assert mine["paired_annotations"] == coverage["paired_annotations"] == 1
    assert mine["left_only_annotations"] == coverage["left_only_annotations"] == 4


def test_missing_figure_is_an_error_not_a_broken_image(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    markdown_text = '<figure><img src="figures/absent.png" alt="x"></figure>'
    with pytest.raises(FileNotFoundError, match="absent.png"):
        report.render_html(markdown_text, figures_dir, "Title")


def test_artifact_html_omits_the_document_skeleton(tmp_path: Path) -> None:
    """The Artifact publisher supplies doctype/html/head/body; we must not."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    html = report.render_artifact_html(
        '<header class="masthead" markdown="1">\n\n# Heading\n\n</header>\n\ntext\n',
        figures_dir, "T")
    lowered = html.lower()
    assert "<!doctype" not in lowered
    # match real tags only: "<header>" must not be mistaken for "<head>"
    for tag in ("html", "head", "body"):
        assert not re.search(r"<%s(?:\s|>|/)" % tag, lowered), tag
    assert re.search(r"<header(?:\s|>)", lowered), "a <header> element must still be allowed"
    assert lowered.startswith("<title>")
    assert "<style>" in lowered and "prefers-color-scheme" in lowered
    assert '[data-theme="dark"]' in html


def test_bracket_paths_reach_keys_containing_dots(concordance_run: Run) -> None:
    facts = report.build_facts({"A": concordance_run}, primary="A", seed_arm=None,
                               model_arms=[], study_meta={})
    kappa = concordance_run.thresholds("MAX", 0.5)["kappa"]
    assert report.resolve("{{primary.thresholds.MAX[0.5].kappa|.4f}}", facts) == f"{kappa:.4f}"
    equivalence = facts["primary"]["agreement"]["MAX"]["equivalence_0.01"]
    assert report.resolve("{{primary.agreement.MAX[equivalence_0.01]|pct2}}", facts) == \
        f"{equivalence * 100:.2f}%"
    with pytest.raises(KeyError, match="unresolved placeholders"):
        report.resolve("{{primary.thresholds.MAX[0.7].kappa}}", facts)


def test_reducer_kind_string_is_recognised(concordance_run: Run, tmp_path: Path) -> None:
    """The reducer writes 'reduced-concordance', not 'concordance'."""
    payload = json.loads(concordance_run.path.read_text())
    payload["kind"] = "reduced-concordance"
    path = tmp_path / "reduced.json"
    path.write_text(json.dumps(payload))
    run = load_run(path, arm="A", role="primary")
    assert run.is_concordance
    written = report.write_tables({"A": run}, tmp_path / "tables")
    assert any(name.endswith("operating_point_transfer.csv") for name in written)


def test_mdx_template_is_resolved_not_converted(concordance_run: Run, tmp_path: Path) -> None:
    """The MDX path substitutes placeholders only; it must not rewrite the markup."""
    facts = report.build_facts({"A": concordance_run}, primary="A", seed_arm=None,
                               model_arms=[], study_meta={"name": "unit"})
    template = (
        "---\ntitle: 'T'\n---\n"
        "import ZoomFigure from '../../components/ZoomFigure.astro';\n\n"
        "Paired: {{primary.coverage.paired_annotations|,d}}\n\n"
        "<ZoomFigure src={fig} alt=\"a\" variant=\"wide\">cap</ZoomFigure>\n"
    )
    resolved = report.resolve(template, facts)
    assert "{{" not in resolved
    assert "<ZoomFigure src={fig}" in resolved          # markup untouched
    assert resolved.startswith("---\ntitle: 'T'\n---")   # frontmatter untouched
    assert f"{concordance_run.coverage['paired_annotations']:,d}" in resolved


def test_seed_fact_base_omits_comparator_specific_transfer(tmp_path):
    path = tmp_path / 'seed-summary.json'
    path.write_text(json.dumps(_summary_from(_build_aggregate(n=20), kind='seeds',
        left='OpenSpliceAI rs10', right='OpenSpliceAI rs13')))
    seed = load_run(path, arm='B', role='seed')
    facts = report.build_facts({'B':seed}, primary='B', seed_arm=None,
                             model_arms=[], study_meta={})['primary']
    assert facts['operating_point'] == {}
    assert facts['operating_point_by_event'] == {}
    assert 'paired_share_of_spliceai_annotations' not in facts['coverage_rates']
    assert 'paired_share_of_left_annotations' in facts['coverage_rates']


def test_missing_markdown_dependency_is_explicit(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__

    def without_markdown(name, *args, **kwargs):
        if name == "markdown":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_markdown)
    with pytest.raises(RuntimeError, match="requires Markdown"):
        report.render_html("# Report", tmp_path, "Title")
