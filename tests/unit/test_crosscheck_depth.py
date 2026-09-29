"""Real-parser differential checks and fail-closed verifier contracts."""

import copy
import csv

import pytest

from validation.concordance_study import crosscheck_depth as check
from validation.concordance_study.depth.aggregate import DepthAggregate
from validation.full_snv_concordance.aggregate import AnalysisConfig, STANDARD_STRATA
from validation.full_snv_concordance.sites import load_site_index
from validation.full_snv_concordance.vcf import iter_variant_groups
from tests.unit.test_full_snv_concordance import ann, row, write_vcf


@pytest.fixture
def check_pair(tmp_path):
    sites = tmp_path / "sites.tsv"
    sites.write_text("#NAME\tCHROM\tSTRAND\tEXON_START\tEXON_END\nG\tchr1\t+\t100,200,\t150,250,\n")
    vcf = tmp_path / "chunk.vcf"
    write_vcf(
        vcf,
        [
            row(pos, ann("G", x), ann("G", y))
            for pos, x, y in (
                (100, (0, 0, 0, 0), (0, 0, 0, 0)),
                (149, (0.29, 0.5, 0.8, 0), (0.57, 0.6, 0.7, 0)),
                (151, (0, 0, 0, 0), (0, 0, 0, 0)),
                (201, (0.8, 0.1, 0.2, 0.7), (0.6, 0.3, 0.4, 0.7)),
                (300, (0, 0, 0, 0), (0, 0, 0, 0)),
            )
        ],
    )
    pairs = tmp_path / "pairs.tsv"
    with pairs.open("w") as h:
        w = csv.DictWriter(h, fieldnames=["chunk_id", "prediction_vcf"], delimiter="\t")
        w.writeheader()
        w.writerow({"chunk_id": "2", "prediction_vcf": str(vcf)})
    index = load_site_index(sites)
    agg = DepthAggregate(
        AnalysisConfig(
            thresholds=check.THRESHOLDS,
            score_bins=200,
            strata=STANDARD_STRATA + ("site_distance",),
            sites_digest=index.digest,
            bootstrap_replicates=0,
            sample_size=0,
            top_k=0,
        ),
        index,
    )
    for group in list(iter_variant_groups([vcf], {"left": "SpliceAI", "right": "OpenSpliceAI"}))[1:-1]:
        agg.add_group(group)
    independent = check.recompute(pairs, sites, "primary")
    summary = {
        "kind": "reduced-concordance",
        "pairs_sha256": independent["pairs_sha256"],
        "raw": agg.to_dict(),
        "metrics": agg.derived(),
        "chunk_ids": ["2"],
        "finality": {"observed_pair_count": 1, "expected_total_chunks": 100000},
    }
    return independent, summary, pairs, sites


def test_independent_checks_full_metrics_and_histograms(check_pair):
    independent, summary, _, _ = check_pair
    result = check.compare(independent, summary)
    assert result["status"] == "passed"
    assert result["checks"] > 1000
    assert result["relative_tolerance"] == 1e-9
    altered = copy.deepcopy(summary)
    altered["raw"]["depth"]["event_strata"]["gene"]["G"]["AG"]["moments"]["sum_cross"] += 1
    assert check.compare(independent, altered)["status"] == "failed"
    altered = copy.deepcopy(summary)
    altered["raw"]["depth"]["site_histograms"]["1-2:donor"]["AG"]["joint"]["11600"] = 100
    assert check.compare(independent, altered)["status"] == "failed"


@pytest.mark.parametrize("remove", ["arms", "event", "dp"])
def test_partial_verification_is_rejected(check_pair, remove):
    independent, summary, _, _ = check_pair
    if remove == "arms":
        independent["arms"] = {}
    elif remove == "event":
        del independent["arms"]["A_rs10_genomewide"]["all"]["AG"]
    else:
        del summary["raw"]["dp"]["AG"]
    with pytest.raises(ValueError):
        check.compare(independent, summary)


def test_cache_binds_current_inputs_and_kind(check_pair, tmp_path):
    independent, _, pairs, sites = check_pair
    check.validate_cache(independent, pairs, sites, "primary")
    with pytest.raises(ValueError):
        check.validate_cache(independent, pairs, sites, "matched")
    sites.write_text(sites.read_text() + "G2\tchr2\t+\t1,20,\t10,30,\n")
    with pytest.raises(ValueError):
        check.validate_cache(independent, pairs, sites, "primary")


@pytest.mark.parametrize(
    "left,right,expected", [(False, True, []), (True, False, []), (False, False, []), (True, True, [1])]
)
def test_single_group_requires_both_outer_boundaries(left, right, expected):
    assert list(check.interior(iter([1]), left, right)) == expected
