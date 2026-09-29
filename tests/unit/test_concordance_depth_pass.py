"""Hand-computed depth and three-way membership checks."""
import json

import pytest

from validation.concordance_study.depth.aggregate import DepthAggregate, MatchedAggregate
from validation.full_snv_concordance.aggregate import Aggregate, AnalysisConfig, STANDARD_STRATA
from validation.full_snv_concordance.sites import load_site_index
from validation.full_snv_concordance.vcf import Annotation, VariantGroup, VariantKey


@pytest.fixture
def setup_depth(tmp_path):
    path = tmp_path / "sites.tsv"
    path.write_text("#NAME\tCHROM\tSTRAND\tEXON_START\tEXON_END\n"
                    "G\tchr1\t+\t100,200,\t150,250,\n")
    sites = load_site_index(path)
    config = AnalysisConfig(thresholds=(0.05, 0.5), score_bins=200,
                            strata=STANDARD_STRATA+("site_distance",), sites_digest=sites.digest,
                            bootstrap_replicates=0, sample_size=10, top_k=10)
    return config, sites


def annotation(gene="G", scores=(0.6, 0.1, 0, 0), dps=(0, 0, 0, 0)):
    return Annotation("A", gene, scores, dps)


def group(pos=149):
    g = VariantGroup(VariantKey("chr1", pos, "G", "A"), rows=1)
    g.add("left", annotation())
    g.add("right", annotation(scores=(0.4, 0.7, 0, 0), dps=(1, 2, 0, 0)))
    return g


def test_event_moments_thresholds_histograms_and_dp(setup_depth):
    config, sites = setup_depth
    a = DepthAggregate(config, sites)
    a.add_group(group())
    s = a.derived()["event_strata"]["site_distance"]["1-2:donor"]
    assert s["AG"]["metrics"]["mae"] == pytest.approx(0.2)
    assert s["AL"]["metrics"]["mae"] == pytest.approx(0.6)
    assert s["AG"]["thresholds"]["0.5"]["left_only"] == 1
    assert s["AL"]["thresholds"]["0.5"]["right_only"] == 1
    hist = a.site_histograms["1-2:donor"]["AG"]
    assert hist["left"][120] == hist["right"][80] == 1
    assert hist["joint"][str(120*200+80)] == 1
    dp = a.site_dp["1-2:donor"]["AG"]["0.05"]
    assert dp["eligible"] == 1 and dp["within"]["0"] == 0 and dp["within"]["1"] == 1


def test_depth_merge_and_json_roundtrip_match_direct(setup_depth):
    config, sites = setup_depth
    left, right, direct = (DepthAggregate(config, sites) for _ in range(3))
    for a, pos in ((left, 149), (right, 201)):
        a.add_group(group(pos))
        direct.add_group(group(pos))
    restored = DepthAggregate.from_dict(json.loads(json.dumps(right.to_dict())))
    left.merge(restored)
    assert left.to_dict() == direct.to_dict()
    assert left.derived() == direct.derived()
    with pytest.raises(ValueError, match="legacy"):
        left.merge(Aggregate(config, sites))
    with pytest.raises(ValueError, match="version"):
        DepthAggregate.from_dict(Aggregate(config, sites).to_dict())


def test_zero_bulk_updates_match_direct_event_metrics(setup_depth):
    config, sites = setup_depth
    a = DepthAggregate(config, sites)
    for pos in (147, 148, 151):
        g = VariantGroup(VariantKey("chr1", pos, "G", "A"), rows=1)
        for side in ("left", "right"):
            g.add(side, annotation(scores=(0, 0, 0, 0)))
        a.add_group(g)
    a.add_group(group())
    derived = a.derived()
    assert derived["event_strata"]["gene"]["G"]["AG"]["metrics"] == a.moments["AG"].derived()
    assert derived["event_strata"]["gene"]["G"]["AG"]["thresholds"] == derived["thresholds"]["AG"]
    before = json.dumps(a.to_dict(), sort_keys=True)
    assert json.dumps(a.to_dict(), sort_keys=True) == before


def test_three_way_uses_only_common_nonconflicting_gene_observations(setup_depth):
    config, sites = setup_depth
    a = MatchedAggregate(config, sites)
    g = group()
    g.add("reference", annotation(scores=(0.8, 0, 0, 0)))
    # Valid in just two methods: must be excluded from all three comparisons.
    for side in ("left", "right"):
        g.add(side, annotation(gene="MISSING_REFERENCE"))
    # Conflicting on one side: must likewise be excluded everywhere.
    for side in ("left", "right", "reference"):
        g.add(side, annotation(gene="CONFLICT"))
    g.add("right", annotation(gene="CONFLICT", scores=(0, 0, 0, 0)))
    a.add_group(g)
    assert a.coverage["shared_annotations"] == 1
    assert a.coverage["excluded_nonshared_annotations"] == 2
    assert {v.moments["MAX"].n for v in a.comparisons.values()} == {1}
    assert a.comparisons["C_rs10_matched"].moments["AG"].sum_abs_diff == pytest.approx(0.2)
    assert a.comparisons["D_rs13_matched"].moments["AG"].sum_abs_diff == pytest.approx(0.4)
    restored = MatchedAggregate.from_dict(json.loads(json.dumps(a.to_dict())))
    assert restored.to_dict() == a.to_dict()
    restored.merge(a)
    assert restored.coverage["shared_annotations"] == 2


def test_boundary_conflict_is_resolved_before_three_way_filter(setup_depth):
    config, sites = setup_depth
    first, last = group(), group()
    first.add("reference", annotation())
    last.add("reference", annotation(scores=(0, 0, 0, 0)))
    first.merge(last)
    a = MatchedAggregate(config, sites)
    a.add_group(first)
    assert a.coverage["shared_annotations"] == 0
    assert all(v.moments["MAX"].n == 0 for v in a.comparisons.values())


def test_audited_matched_map_reduce_rejoins_real_boundaries(setup_depth, tmp_path):
    from tests.unit.test_full_snv_concordance import SOURCE_HEADER, ann, row, write_vcf
    from validation.full_snv_concordance.workflows import (
        PairRow, audit_pair, build_pairs_files, run_map_seeds, run_reduce,
    )
    config, sites = setup_depth
    manifests = []
    for seed, value in (("rs10", 0.4), ("rs13", 0.7)):
        audited = []
        for chunk, positions in ((1, (149, 150)), (2, (150, 201))):
            source = tmp_path/f"source-{seed}-{chunk}.vcf"
            predicted = tmp_path/f"predicted-{seed}-{chunk}.vcf"
            write_vcf(source, [row(pos, ann("G", (0.6, 0, 0, 0))) for pos in positions], SOURCE_HEADER)
            write_vcf(predicted, [row(pos, ann("G", (0.6, 0, 0, 0)),
                                     ann("G", (value, 0, 0, 0))) for pos in positions])
            r = audit_pair(PairRow(str(chunk), {"source_vcf": str(source),
                           "prediction_vcf": str(predicted), "seed": seed}))
            assert r["state"] == "valid"
            audited.append(r)
        path = tmp_path/f"{seed}.json"
        path.write_text(json.dumps({"chunks": audited}))
        manifests.append(path)
    pairs = tmp_path/"seeds.tsv"
    build_pairs_files(manifests[0], tmp_path/"primary.tsv", manifests[1], pairs)
    for task in range(2):
        run_map_seeds(pairs, tmp_path/f"maps/map-{task}.json", config,
                      task_index=task, chunks_per_task=1, sites=sites,
                      aggregate_type=MatchedAggregate, shared_info="SpliceAI")
    result = run_reduce("seeds", tmp_path/"summary.json", input_dir=tmp_path/"maps",
                        expected_task_count=2, pairs_file=pairs, expected_total_chunks=2,
                        sites=sites, aggregate_type=MatchedAggregate)
    assert result["raw"]["coverage"]["shared_annotations"] == 3
    for child in result["metrics"]["comparisons"].values():
        assert child["scores"]["AG"]["n"] == 3
        assert child["event_strata"]["site_distance"]["at_site:donor"]["AG"]["metrics"]["n"] == 1
    # A legacy reducer may not accidentally accept the nested matched state.
    with pytest.raises(ValueError, match="schema version"):
        run_reduce("seeds", tmp_path/"invalid.json", input_dir=tmp_path/"maps",
                   expected_task_count=2, pairs_file=pairs, expected_total_chunks=2, sites=sites)
