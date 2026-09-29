"""Resolve publication documents and assemble the September 8 depth-study release.

Run after ``render-study``. This step reads the completed fact base and figures;
it does not recompute results or modify frozen analysis runs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from . import report


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def archive(path: Path, files: dict[str, Path]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        checks = []
        for name, source in sorted(files.items()):
            out.write(source, name)
            checks.append(f"{digest(source)}  {name}\n")
        out.writestr("SHA256SUMS.txt", "".join(checks))
    if path.stat().st_size >= 95_000_000:
        raise ValueError(f"Archive exceeds the publication file-size budget: {path}")
    print(f"{path.name}: {path.stat().st_size:,} bytes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--website-root", type=Path, required=True)
    parser.add_argument("--legacy-september9", action="store_true",
                        help="Explicitly regenerate the archived September 9 release")
    args = parser.parse_args()
    if not args.legacy_september9:
        parser.error("Use concise_release with --progress and a new --output for revised reports. "
                     "Archived release regeneration requires --legacy-september9.")
    base = args.study_dir.resolve()
    publication = base / "publication"
    website = args.website_root.resolve()
    source = Path(__file__).resolve().parents[2]
    templates = Path(__file__).parent / "templates/archive/20260909"
    facts = json.loads((publication / "study_facts.json").read_text())
    assert facts["primary"]["coverage"]["paired_annotations"] == 3_334_708_099
    matched = [facts["arms"][key] for key in (
        "B_seeds_rs10_rs13", "C_rs10_matched", "D_rs13_matched")]
    assert all(arm["coverage"]["paired_annotations"] == 1_536_316_689 for arm in matched)
    assert all(arm["finality"]["status"] == "provisional" for arm in facts["arms"].values())
    assert len(list((publication / "figures").glob("*.png"))) == 16
    assert len(list((publication / "figures").glob("*.pdf"))) == 16
    boundaries = json.loads((base / "verification/boundary_impact.json").read_text())
    assert all(r["maps_checked"] == r["expected_maps"] and not r["whole_chunk_groups"]
               for r in boundaries)

    stem = "OpenSpliceAI_vs_SpliceAI_genome_wide"
    markdown = report.resolve((templates / f"{stem}.md").read_text(), facts)
    (publication / f"{stem}.md").write_text(markdown)
    for suffix, renderer in (("html", report.render_html),
                             ("artifact.html", report.render_artifact_html)):
        (publication / f"{stem}.{suffix}").write_text(
            renderer(markdown, publication / "figures", facts["study"]["title"]))
    for slug in ("full-snv-scoring-technical-report", "openspliceai-technical-report"):
        content = report.resolve((templates / f"{slug}.mdx").read_text(), facts)
        assert "PILOT PREVIEW" not in content and "INTERPRET" not in content
        (website / "src/content/reports" / f"{slug}.mdx").write_text(content)
        (publication / f"{slug}.mdx").write_text(content)
    figure_dest = website / "src/assets/reports/full-snv-scoring-technical-report"
    figure_dest.mkdir(parents=True, exist_ok=True)
    for figure in (publication / "figures").glob("*.png"):
        shutil.copy2(figure, figure_dest / figure.name)

    downloads = website / "public/downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    prefix = "full-snv-concordance-20260909"
    # The hosted reading copy follows the site's prohibition on data: URLs.
    # Self-contained HTML with embedded figures remains available in the archive.
    public_html = (publication / f"{stem}.html").read_text()
    public_figures = downloads / prefix / "figures"
    public_figures.mkdir(parents=True, exist_ok=True)
    for figure in (publication / "figures").glob("*.png"):
        embedded = 'src="data:image/png;base64,' + base64.b64encode(figure.read_bytes()).decode() + '"'
        if embedded not in public_html:
            raise ValueError(f"Standalone document is missing {figure.name}")
        public_html = public_html.replace(embedded, f'src="/downloads/{prefix}/figures/{figure.name}"')
        shutil.copy2(figure, public_figures / figure.name)
    (downloads / f"{prefix}.html").write_text(public_html)
    files = {p.name: p for p in publication.iterdir() if p.is_file()}
    files.update({str(p.relative_to(publication)): p
                  for p in (publication / "figures").iterdir() if p.is_file()})
    shared_csv = publication / "tables/seed_versus_method_strata.csv"
    files["tables/seed_versus_method_strata.csv"] = shared_csv
    for name in ("receipt_primary.json", "receipt_matched.json", "boundary_impact.json",
                 "production_contracts.json", "frozen_code_equivalence.json",
                 "production_slurm.tsv", "independent_headlines.json"):
        files[f"verification/{name}"] = base / "verification" / name
    files["study_meta.json"] = base / "study_meta.json"
    for package in ("concordance_study", "full_snv_concordance"):
        for p in (source / "validation" / package).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".mdx", ".sh", ".sbatch", ".json"):
                files["analysis_source/" + str(p.relative_to(source))] = p
    files["analysis_source/validation/__init__.py"] = source / "validation/__init__.py"
    files["analysis_source/LICENSE"] = source / "LICENSE"
    for name in ("test_full_snv_concordance.py", "test_full_snv_concordance_depth.py",
                 "test_site_distance.py", "test_concordance_study.py",
                 "test_concordance_depth_pass.py", "test_crosscheck_depth.py"):
        files[f"analysis_source/tests/unit/{name}"] = source / "tests/unit" / name
    for name in ("A_rs10_genomewide", "matched"):
        run = base / "production" / name
        for p in run.rglob("*"):
            if p.is_file() and (p.parent == run and p.name in (
                    "checks.sha256", "worker.sh", "launch.json", "pairs.tsv", "sites.tsv")
                    or "code" in p.relative_to(run).parts and p.suffix != ".pyc"):
                files[f"execution/{name}/" + str(p.relative_to(run))] = p
    archive(downloads / f"{prefix}.zip", files)
    for directory in sorted((publication / "tables").iterdir()):
        if directory.is_dir():
            archive(downloads / f"{prefix}-{directory.name}.zip",
                    {str(p.relative_to(publication)): p for p in directory.glob("*.csv")})
    (downloads / f"{prefix}-SHA256SUMS.txt").write_text("".join(
        f"{digest(p)}  {p.name}\n" for p in sorted(downloads.glob(f"{prefix}*"))
        if p.suffix in (".zip", ".html")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
