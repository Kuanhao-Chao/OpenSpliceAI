"""CLI for the frozen depth pass; rendering remains outside this package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.full_snv_concordance.aggregate import AnalysisConfig, STANDARD_STRATA
from validation.full_snv_concordance.sites import load_site_index
from validation.full_snv_concordance.workflows import (
    atomic_write_json, run_map_concordance, run_map_seeds, run_reduce, sha256_file,
)
from .aggregate import DepthAggregate, MatchedAggregate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("map-primary", "map-matched", "reduce-primary", "reduce-matched"))
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--sites-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input-dir")
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--chunks-per-task", type=int, default=250)
    p.add_argument("--expected-task-count", type=int)
    p.add_argument("--expected-total-chunks", type=int, default=100000)
    p.add_argument("--run-label", default="provisional-depth-20260908")
    p.add_argument("--bootstrap-replicates", type=int, default=200)
    a = p.parse_args(argv)
    sites = load_site_index(a.sites_file)
    config = AnalysisConfig(thresholds=(0.05, 0.1, 0.2, 0.5, 0.8), score_bins=200,
                            sample_size=5000, top_k=30000,
                            strata=STANDARD_STRATA+("site_distance",), sites_digest=sites.digest,
                            bootstrap_replicates=a.bootstrap_replicates)
    matched = a.command.endswith("matched")
    factory = MatchedAggregate if matched else DepthAggregate
    if a.command.startswith("map"):
        common = dict(pairs_file=a.pairs_file, output=a.output, config=config,
                      task_index=a.task_index, chunks_per_task=a.chunks_per_task,
                      run_label=a.run_label, sites=sites, aggregate_type=factory)
        if matched:
            result = run_map_seeds(**common, shared_info="SpliceAI")
        else:
            result = run_map_concordance(**common)
        print(json.dumps({"output": a.output, "chunks": len(result["chunk_ids"]),
                          "coverage": result["aggregate"]["coverage"]}))
        return 0
    result = run_reduce("seeds" if matched else "concordance", output=a.output,
                        input_dir=a.input_dir, expected_task_count=a.expected_task_count,
                        pairs_file=a.pairs_file, expected_total_chunks=a.expected_total_chunks,
                        finality="provisional", sites=sites, aggregate_type=factory)
    if matched:
        # Each arm carries the joint source's verified provenance and explicit
        # observation-domain declaration. No pairwise summary is relabelled as
        # a three-way comparison: these data were aggregated together above.
        labels = {"B_seeds_rs10_rs13": ("rs10", "rs13"),
                  "C_rs10_matched": ("SpliceAI", "rs10"),
                  "D_rs13_matched": ("SpliceAI", "rs13")}
        source_sha = sha256_file(a.output)
        for name, (left, right) in labels.items():
            payload = {k: v for k, v in result.items() if k not in ("raw", "metrics")}
            payload.update(raw=result["raw"]["comparisons"][name],
                           metrics=result["metrics"]["comparisons"][name],
                           left_label=left, right_label=right,
                           kind="reduced-seeds" if name.startswith("B_") else "reduced-concordance")
            payload["matched_domain"] = {"policy": result["raw"]["domain"],
                "joint_summary_sha256": source_sha, "pairs_sha256": result["pairs_sha256"],
                "shared_annotations": result["raw"]["coverage"]["shared_annotations"],
                "excluded_nonshared_annotations": result["raw"]["coverage"].get("excluded_nonshared_annotations", 0)}
            atomic_write_json(Path(a.output).parent / name / "summary.json", payload)
    print(f"Reduced: {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
