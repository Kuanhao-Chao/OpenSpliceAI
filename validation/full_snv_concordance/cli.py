"""Command-line entry point for the full-SNV concordance workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .aggregate import AnalysisConfig
from .external import harmonize_external
from .external_evaluate import evaluate_external
from .reporting import render_report
from .sites import load_site_index
from .workflows import (
    build_pairs_files,
    run_audit,
    run_map_concordance,
    run_map_seeds,
    run_reduce,
)


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return result


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_map_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pairs-file", required=True, help="Headered TSV describing VCF chunks")
    parser.add_argument("--output", required=True, help="Mapper JSON shard")
    parser.add_argument("--task-index", type=int, help="Zero-based slice (defaults to SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--chunks-per-task", type=int, default=250)
    parser.add_argument(
        "--run-label",
        default="provisional",
        help="Descriptive mapper label only; reducer finality is set separately",
    )
    parser.add_argument("--thresholds", type=_comma_floats, default=(0.1, 0.2, 0.5, 0.8))
    parser.add_argument("--score-bins", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=1_000)
    parser.add_argument(
        "--strata",
        type=_comma_strings,
        default=(
            "chrom",
            "gene",
            "block_1mb",
            "substitution",
            "dominant_pair",
            "site_event",
        ),
    )
    parser.add_argument(
        "--sites-file",
        help=(
            "SpliceAI-format annotation table (e.g. data/grch38_chr.txt) from which "
            "annotated splice sites are derived for the site_distance stratum. Required "
            "whenever that stratum is requested; its SHA-256 is recorded in the config so "
            "shards built against different annotations cannot be merged."
        ),
    )
    parser.add_argument("--parquet-dir", help="Optional compact sample/top-discrepancy Parquet directory")


def _sites(args: argparse.Namespace):
    """Load the splice-site index when the site_distance stratum is requested."""
    if "site_distance" not in tuple(args.strata):
        if getattr(args, "sites_file", None):
            raise SystemExit("--sites-file was given but the site_distance stratum is not enabled")
        return None
    if not getattr(args, "sites_file", None):
        raise SystemExit("--sites-file is required when the site_distance stratum is enabled")
    return load_site_index(args.sites_file)


def _config(args: argparse.Namespace, sites=None) -> AnalysisConfig:
    return AnalysisConfig(
        thresholds=tuple(args.thresholds),
        score_bins=args.score_bins,
        sample_size=args.sample_size,
        top_k=args.top_k,
        strata=tuple(args.strata),
        sites_digest=sites.digest if sites is not None else "",
    )


def _task_index(value: Optional[int]) -> Optional[int]:
    if value is not None:
        return value
    environment = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(environment) if environment is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validation.full_snv_concordance",
        description="Streaming audit and concordance analysis for full-genome SNV VCFs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Field-level source/output VCF integrity audit")
    audit.add_argument("--pairs-file", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--parquet")
    audit.add_argument("--task-index", type=int)
    audit.add_argument("--chunks-per-task", type=int, default=250)
    audit.add_argument("--distance", type=int, default=50, help="Allowed absolute DP bound")

    pairs = subparsers.add_parser(
        "build-pairs", help="Build numerically sorted mapper TSVs from audit manifests"
    )
    pairs.add_argument("--left-manifest", required=True)
    pairs.add_argument("--concordance-output", required=True)
    pairs.add_argument("--right-manifest")
    pairs.add_argument("--seeds-output")
    pairs.add_argument(
        "--accepted-state", action="append", default=[], help="Repeatable; default: valid"
    )
    pairs.add_argument("--require-left-count", type=int)
    pairs.add_argument("--require-overlap-count", type=int)

    map_concordance = subparsers.add_parser(
        "map-concordance", help="Map SpliceAI versus OpenSpliceAI statistics"
    )
    _add_map_options(map_concordance)
    map_seeds = subparsers.add_parser(
        "map-seeds", help="Map exact-gene OpenSpliceAI seed reproducibility statistics"
    )
    _add_map_options(map_seeds)

    for name in ("reduce-concordance", "reduce-seeds"):
        reduce_parser = subparsers.add_parser(name, help=f"Merge {name[7:]} mapper shards")
        reduce_parser.add_argument("--input", action="append", default=[], help="Mapper JSON; repeatable")
        reduce_parser.add_argument("--input-dir", help="Directory containing map-*.json")
        reduce_parser.add_argument("--output", required=True)
        reduce_parser.add_argument("--parquet-dir")
        reduce_parser.add_argument("--pairs-file", required=True)
        reduce_parser.add_argument(
            "--sites-file",
            help=(
                "Annotation used for the site_distance stratum. Required when the mapper "
                "shards carry one: the reducer rejoins the chunk-boundary groups the mappers "
                "deferred, and those are real rows that must be stratified against the same "
                "annotation as the interior."
            ),
        )
        reduce_parser.add_argument("--expected-task-count", type=int, required=True)
        reduce_parser.add_argument("--expected-total-chunks", type=int, default=100000)
        reduce_parser.add_argument(
            "--finality",
            choices=("provisional", "final"),
            default="provisional",
            help="Explicit reducer status contract; final enforces complete coverage",
        )
        if name == "reduce-seeds":
            reduce_parser.add_argument(
                "--expected-overlap-count",
                type=int,
                help="Required explicit audited overlap policy for final seed reduction",
            )

    report = subparsers.add_parser("render-report", help="Render Markdown and bounded-density PNGs")
    report.add_argument("--summary", required=True)
    report.add_argument("--output-dir", required=True)

    external = subparsers.add_parser(
        "validate-external",
        help="Validate a functional-data manifest, harmonize local tables, and plan scoring",
    )
    external.add_argument("--manifest", required=True)
    external.add_argument("--output-dir", required=True)
    external.add_argument("--reference-fasta")
    external.add_argument("--liftover-chain", help="UCSC hg19-to-hg38 chain for GRCh37 cohorts")
    external.add_argument("--annotation")
    external.add_argument(
        "--openspliceai-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Individual checkpoint or ensemble directory; repeatable",
    )
    external.add_argument("--no-spliceai", action="store_true")
    external.add_argument(
        "--mask",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Score masking mode: 0 (default) for conventional external functional "
            "max-DS evaluation; 1 only for an explicitly labelled masked sensitivity run"
        ),
    )
    external.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DATASET_ID=PATH",
        help="Override a manifest local_path; repeatable",
    )
    external.add_argument(
        "--allow-incomplete-sensitivity",
        action="store_true",
        help=(
            "Label a plan with unavailable/zero-accepted datasets as an incomplete "
            "sensitivity analysis; primary scoring otherwise fails closed"
        ),
    )

    evaluate = subparsers.add_parser(
        "evaluate-external",
        help="Evaluate scored VCFs against harmonized experimental binary outcomes",
    )
    evaluate.add_argument("--harmonized", required=True, help="harmonized.tsv from validate-external")
    evaluate.add_argument(
        "--harmonized-sha256",
        help="Expected SHA-256 bound to the harmonized bytes parsed for evaluation",
    )
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Scored VCF and its unique predictor name; repeatable",
    )
    evaluate.add_argument(
        "--score-info",
        action="append",
        default=[],
        metavar="NAME=INFO_ID",
        help="Resolve a VCF containing both SpliceAI and OpenSpliceAI INFO; repeatable",
    )
    evaluate.add_argument(
        "--score-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="Expected raw SHA-256 bound to the bytes parsed for a score; repeatable",
    )
    evaluate.add_argument(
        "--score-records",
        action="append",
        default=[],
        metavar="NAME=N",
        help="Expected VCF record count from the frozen-run receipt; repeatable",
    )
    evaluate.add_argument(
        "--validation-plan",
        help="Frozen validation_plan.json carrying dataset completeness provenance",
    )
    evaluate.add_argument(
        "--validation-plan-sha256",
        help="Expected SHA-256 bound to the frozen validation plan bytes",
    )
    evaluate.add_argument(
        "--run-provenance",
        help="Frozen run/bundle/receipt identity document produced by the scoring launcher",
    )
    evaluate.add_argument(
        "--run-provenance-sha256",
        help="Expected SHA-256 bound to the frozen run-provenance document",
    )
    evaluate.add_argument(
        "--thresholds", type=_comma_floats, default=(0.1, 0.2, 0.5, 0.8)
    )
    evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20240801)
    evaluate.add_argument(
        "--score-mask",
        type=int,
        choices=(0, 1),
        help="Masking provenance shared by all supplied score VCFs",
    )
    evaluate.add_argument(
        "--minimum-prediction-coverage",
        type=float,
        default=0.9,
        help=(
            "Minimum matched fraction for every predictor globally and within every "
            "dataset and cohort"
        ),
    )
    evaluate.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = run_audit(
            args.pairs_file,
            args.output,
            parquet=args.parquet,
            task_index=_task_index(args.task_index),
            chunks_per_task=args.chunks_per_task,
            dp_distance=args.distance,
        )
        print(json.dumps(result["state_counts"], sort_keys=True))
    elif args.command == "build-pairs":
        result = build_pairs_files(
            args.left_manifest,
            args.concordance_output,
            right_manifest=args.right_manifest,
            seeds_output=args.seeds_output,
            accepted_states=args.accepted_state or ("valid",),
            require_left_count=args.require_left_count,
            require_overlap_count=args.require_overlap_count,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "map-concordance":
        sites = _sites(args)
        result = run_map_concordance(
            args.pairs_file,
            args.output,
            _config(args, sites),
            task_index=_task_index(args.task_index),
            chunks_per_task=args.chunks_per_task,
            run_label=args.run_label,
            parquet_dir=args.parquet_dir,
            sites=sites,
        )
        print(f"mapped {len(result['chunk_ids'])} chunks to {args.output}")
    elif args.command == "map-seeds":
        sites = _sites(args)
        result = run_map_seeds(
            args.pairs_file,
            args.output,
            _config(args, sites),
            task_index=_task_index(args.task_index),
            chunks_per_task=args.chunks_per_task,
            run_label=args.run_label,
            parquet_dir=args.parquet_dir,
            sites=sites,
        )
        print(f"mapped {len(result['chunk_ids'])} chunks to {args.output}")
    elif args.command in {"reduce-concordance", "reduce-seeds"}:
        kind = "concordance" if args.command.endswith("concordance") else "seeds"
        reduce_sites = load_site_index(args.sites_file) if args.sites_file else None
        result = run_reduce(
            kind,
            args.output,
            inputs=args.input,
            input_dir=args.input_dir,
            parquet_dir=args.parquet_dir,
            pairs_file=args.pairs_file,
            expected_task_count=args.expected_task_count,
            expected_total_chunks=args.expected_total_chunks,
            finality=args.finality,
            expected_overlap_count=getattr(args, "expected_overlap_count", None),
            sites=reduce_sites,
        )
        print(
            f"reduced {len(result['input_shards'])} shards and "
            f"{len(result['chunk_ids'])} chunks to {args.output}"
        )
    elif args.command == "render-report":
        report = render_report(args.summary, args.output_dir)
        print(report)
    elif args.command == "validate-external":
        source_overrides = {}
        for specification in args.source:
            if "=" not in specification:
                raise ValueError("--source values must be DATASET_ID=PATH")
            identifier, path = specification.split("=", 1)
            source_overrides[identifier] = path
        result = harmonize_external(
            args.manifest,
            args.output_dir,
            reference_fasta=args.reference_fasta,
            annotation=args.annotation,
            openspliceai_models=args.openspliceai_model,
            include_spliceai=not args.no_spliceai,
            source_overrides=source_overrides,
            liftover_chain=args.liftover_chain,
            score_mask=args.mask,
            allow_incomplete_sensitivity=args.allow_incomplete_sensitivity,
        )
        print(
            json.dumps(
                {
                    "accepted_rows": result["accepted_rows"],
                    "rejected_rows": result["rejected_rows"],
                    "unique_variants": result["unique_variants"],
                    "dataset_completeness": result["dataset_completeness"],
                    "plan": str(Path(args.output_dir) / "validation_plan.json"),
                },
                sort_keys=True,
            )
        )
    elif args.command == "evaluate-external":
        scored_vcfs = {}
        for specification in args.score:
            if "=" not in specification:
                raise ValueError("--score values must be NAME=PATH")
            name, path = specification.split("=", 1)
            if not name or not path:
                raise ValueError("--score values must have nonempty NAME and PATH")
            if name in scored_vcfs:
                raise ValueError(f"duplicate --score predictor name: {name}")
            scored_vcfs[name] = path
        info_keys = {}
        for specification in args.score_info:
            if "=" not in specification:
                raise ValueError("--score-info values must be NAME=INFO_ID")
            name, info_key = specification.split("=", 1)
            if not name or not info_key:
                raise ValueError("--score-info values must have nonempty NAME and INFO_ID")
            if name in info_keys:
                raise ValueError(f"duplicate --score-info predictor name: {name}")
            info_keys[name] = info_key
        expected_sha256 = {}
        for specification in args.score_sha256:
            if "=" not in specification:
                raise ValueError("--score-sha256 values must be NAME=SHA256")
            name, digest = specification.split("=", 1)
            if name in expected_sha256 or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()
            ):
                raise ValueError("--score-sha256 requires unique NAME=64_HEX_SHA256 values")
            expected_sha256[name] = digest.lower()
        expected_records = {}
        for specification in args.score_records:
            if "=" not in specification:
                raise ValueError("--score-records values must be NAME=N")
            name, value = specification.split("=", 1)
            if name in expected_records:
                raise ValueError(f"duplicate --score-records predictor name: {name}")
            try:
                records = int(value)
            except ValueError as exc:
                raise ValueError("--score-records values must be NAME=N") from exc
            if records < 0:
                raise ValueError("--score-records counts must be nonnegative")
            expected_records[name] = records
        for label, values in (
            ("--score-sha256", expected_sha256),
            ("--score-records", expected_records),
        ):
            unknown = set(values) - set(scored_vcfs)
            if unknown:
                raise ValueError(f"{label} has no scored VCF: {sorted(unknown)}")
        result = evaluate_external(
            args.harmonized,
            scored_vcfs,
            args.output_dir,
            info_keys=info_keys,
            thresholds=args.thresholds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            render_plots=not args.no_plots,
            score_mask=args.score_mask,
            expected_score_sha256=expected_sha256,
            expected_score_records=expected_records,
            validation_plan=args.validation_plan,
            expected_validation_plan_sha256=args.validation_plan_sha256,
            minimum_prediction_coverage=args.minimum_prediction_coverage,
            expected_harmonized_sha256=args.harmonized_sha256,
            run_provenance=args.run_provenance,
            expected_run_provenance_sha256=args.run_provenance_sha256,
        )
        pooled = [row for row in result["groups"] if row["scope"] == "pooled"]
        print(
            json.dumps(
                {
                    "predictors": list(scored_vcfs),
                    "labelled_rows": result["harmonized"]["valid_binary_labels"],
                    "pooled_scored_rows": {
                        row["predictor"]: row["n_scored"] for row in pooled
                    },
                    "report": result["outputs"]["report_markdown"],
                    "json": result["outputs"]["json"],
                },
                sort_keys=True,
            )
        )
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
