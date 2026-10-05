"""Command line for the cross-run synthesis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from . import figures, report
from .loading import load_study


def _parse_arm(value: str) -> tuple:
    """``arm=directory:role[:expected_chunks]``"""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected arm=directory:role, got {value!r}")
    arm, rest = value.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"expected directory:role in {value!r}")
    directory, role = parts[0], parts[1]
    expected = int(parts[2]) if len(parts) > 2 else None
    return arm, {"directory": directory, "role": role, "expected_chunks": expected}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validation.concordance_study",
        description="Synthesise several reduced concordance runs into one report",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render-study", help="Build facts, tables, figures and the report")
    render.add_argument("--runs-root", required=True)
    render.add_argument("--arm", action="append", required=True, type=_parse_arm,
                        help="arm=directory:role[:expected_chunks]; repeatable")
    render.add_argument("--primary", required=True)
    render.add_argument("--seed-arm")
    render.add_argument("--model-arm", action="append", default=[])
    render.add_argument("--template", help="Markdown narrative containing {{fact}} placeholders")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--title", default="OpenSpliceAI vs SpliceAI")
    render.add_argument("--study-meta", help="JSON file of study-level metadata")
    render.add_argument("--sites-file", help="Digest-matched annotation for top-discrepancy context")
    render.add_argument(
        "--mdx-template",
        help=(
            "Optional MDX template resolved against the same fact base. It is authored as "
            "real MDX (frontmatter + ZoomFigure), so no HTML-to-MDX conversion is involved; "
            "only the {{fact}} placeholders are substituted."
        ),
    )
    render.add_argument("--mdx-output", help="Where to write the resolved MDX")
    render.add_argument(
        "--figure-dest",
        help="Copy the rendered figures here (e.g. the website's report assets directory)",
    )
    render.add_argument("--facts-only", action="store_true",
                        help="Write study_facts.json and tables, skip figures and document")

    facts = sub.add_parser("show-fact", help="Print one value from a study_facts.json")
    facts.add_argument("--facts", required=True)
    facts.add_argument("path")
    facts.add_argument("--format")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "show-fact":
        payload = json.loads(Path(args.facts).read_text())
        print(report.resolve("{{" + args.path + (f"|{args.format}" if args.format else "") + "}}", payload))
        return 0

    spec = dict(args.arm)
    runs = load_study(args.runs_root, spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    study_meta = json.loads(Path(args.study_meta).read_text()) if args.study_meta else {}
    sites = None
    if args.sites_file:
        from validation.full_snv_concordance.sites import load_site_index
        sites = load_site_index(args.sites_file)
    facts = report.build_facts(runs, primary=args.primary, seed_arm=args.seed_arm,
                               model_arms=args.model_arm, study_meta=study_meta, sites=sites)
    report.write_facts(facts, out_dir / "study_facts.json")
    written = report.write_tables(runs, out_dir / "tables", sites=sites)
    if facts.get("seed_versus_model", {}).get("strata"):
        report._write_csv(out_dir / "tables" / "seed_versus_method_strata.csv",
                          facts["seed_versus_model"]["strata"])
    print(f"facts: {out_dir / 'study_facts.json'}")
    print(f"tables: {len(written)} files under {out_dir / 'tables'}")

    if args.facts_only:
        return 0

    figure_dir = out_dir / "figures"
    index = figures.render_all(runs, primary=args.primary, seed_arm=args.seed_arm or "",
                               model_arms=args.model_arm, out_dir=figure_dir)
    (out_dir / "figures" / "index.json").write_text(json.dumps(index, indent=2))
    print(f"figures: {len(index)} written to {figure_dir}")

    if args.template:
        template = Path(args.template).read_text()
        markdown_text = report.resolve(template, facts)
        stem = Path(args.template).stem
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(markdown_text)
        html_path = out_dir / f"{stem}.html"
        html_path.write_text(report.render_html(markdown_text, figure_dir, args.title))
        artifact_path = out_dir / f"{stem}.artifact.html"
        artifact_path.write_text(report.render_artifact_html(markdown_text, figure_dir, args.title))
        print(f"report: {md_path}")
        print(f"report: {html_path}")
        print(f"report: {artifact_path}")

    if args.mdx_template:
        if not args.mdx_output:
            raise SystemExit("--mdx-template requires --mdx-output")
        mdx = report.resolve(Path(args.mdx_template).read_text(), facts)
        mdx_path = Path(args.mdx_output)
        mdx_path.parent.mkdir(parents=True, exist_ok=True)
        mdx_path.write_text(mdx)
        print(f"report: {mdx_path}")

    if args.figure_dest:
        destination = Path(args.figure_dest)
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for figure in sorted(figure_dir.glob("*.png")):
            shutil.copy2(figure, destination / figure.name)
            copied += 1
        print(f"figures: {copied} copied to {destination}")
    return 0
