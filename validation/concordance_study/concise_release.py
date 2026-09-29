"""Generate a concise revision from frozen facts and a separate progress snapshot.

Never writes to the original publication directory or its dated downloads.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import date as calendar_date
import html
import json
from pathlib import Path
import re
import shutil

import yaml

from . import concise_figures, report
from .loading import EVENTS
from .prepare_release import archive, digest

TEMPLATES = Path(__file__).parent / "templates"
SLUGS = {"main": "full-snv-scoring-technical-report", "supplement": "full-snv-scoring-supplement"}
AUDIT = [
    (1, "Coverage", "Different counting units require explicit denominators.", "S1", "Retain as methods support."),
    (2, "Conditional scores", "Loss scores agree more closely; donor gain differs most.", "2", "Four larger event panels; omit MAX from the main figure."),
    (3, "Signed tails", "Gain differences are directionally asymmetric.", "S2", "Retain secondary distribution evidence."),
    (4, "Threshold agreement", "Loss overlap is higher; donor gain has fewer shared calls.", "3", "Focus on Jaccard and relative call counts."),
    (5, "Threshold transfer", "Matching call counts need not match membership.", "S3", "Retain; clarify comparator-based objectives."),
    (6, "Output precision", "Score differences remain after a 0.005 allowance.", "S4", "Retain as a robustness check."),
    (7, "Dominant events", "Largest-score labels compress nonexclusive event scores.", "S5", "Retain ties/zeros and limit biological interpretation."),
    (8, "Selected positions", "Shared gain calls usually select the same position.", "5", "Show exact rates and eligible counts; explain loss-mask constraint."),
    (9, "Boundary distance", "Call rates and gain–loss differences vary with proximity.", "4A–D", "Pool counts before division; distinguish variant and event position."),
    (10, "Context distributions", "Proximity groups have different score distributions.", "S6", "Retain supporting survival curves."),
    (11, "Conditional context", "Gain score agreement is closer near boundaries.", "4E–F; S7", "Highlight gain medians; retain all four events in supplement."),
    (12, "Context positions", "Agreement depends on a jointly called, sometimes small subset.", "S8", "Retain denominators and undefined groups."),
    (13, "Seed contrast", "Gain method differences exceed the observed seed contrast.", "6", "Compare four events on the identical three-way domain."),
    (14, "Seed contrast by gene", "The gain–loss distinction extends across retained genes.", "S9", "Retain; distinguish gene and annotation weighting."),
    (15, "Chromosomes/blocks", "Regional MAX means vary, including extreme retained blocks.", "S10", "Redraw with full vertical range; retain all eligible blocks."),
    (16, "Gene divergence", "Gene-level MAX means differ across retained genes.", "S11", "Retain descriptive candidates without enrichment claims."),
]


def figure_audit():
    return "| Original figure | Question | Key result | Revised figure | Decision |\n|---|---|---|---|---|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in AUDIT)


def template_context(facts, snapshot, release_id=None, revision_date=None):
    progress = []
    for seed in ("rs10", "rs13"):
        data = snapshot["seeds"][seed]
        counts, total = data["current_counts"], data["total_chunks"]
        if counts is None or set(counts) != {"valid", "missing", "invalid", "unverified"} or any(
                not isinstance(n, int) or n < 0 for n in counts.values()) or sum(counts.values()) != total or total <= 0:
            raise ValueError("invalid or unchecked progress categories")
        progress.append(f"{seed} has {counts['valid']:,} of {total:,} chunks ({counts['valid']/total:.3%}) "
                        f"with an audited-valid state and unchanged metadata; {counts['missing']:,} are missing, "
                        f"{counts['invalid']:,} invalid, and {counts['unverified']:,} changed or unverified")
    rows = []
    for event in EVENTS:
        threshold = facts["primary"]["thresholds"][event]["0.5"]
        rows.append(f"| {event} | {facts['primary']['agreement'][event]['pearson_r']:.3f} | "
                    f"{threshold['jaccard']:.3f} | {threshold['call_rate_ratio_right_over_left']:.3f} |")
    date = revision_date or snapshot["observed_at_utc"][:10]
    calendar_date.fromisoformat(date)
    release_id = release_id or date.replace("-", "")
    if not re.fullmatch(r"[0-9]{8}(?:-[a-z0-9]+)*", release_id):
        raise ValueError("invalid release ID")
    revision = dict(date=date, observed_at=snapshot["observed_at_utc"].replace("T", " "),
                    prefix="full-snv-concordance-" + release_id,
                    progress_summary="At the current metadata check, " + "; ".join(progress) + ". "
                    "These counts carry forward the " + snapshot["seeds"]["rs10"]["audit_at_utc"][:10] +
                    " content audit; they do not constitute a new VCF content audit. Figure 1 records the separately observed scheduler state.",
                    event_table="| Event | Pearson r | Jaccard at 0.5 | OpenSpliceAI / SpliceAI calls |\n|---|---:|---:|---:|\n" + "\n".join(rows),
                    dp={e: {"exact": facts["primary"]["dp"][e]["0.5"]["within_0bp"]} for e in EVENTS},
                    figure_audit=figure_audit())
    revision["call_changes"] = {e: abs(facts["primary"]["thresholds"][e]["0.5"]["call_rate_ratio_right_over_left"]-1) for e in EVENTS}
    revision["boundary_ratios"] = {
        e: {("far" if r["distance"] == ">500" else r["distance"]): r["ratio"] for r in concise_figures.pool_site_rates(
            facts["primary"].get("site_event_table", []), e, .5)} for e in EVENTS}
    return {**facts, "revision": revision}


def expand_figures(body, figures, document, mode):
    selected = {f["id"]: f for f in figures if f["document"] == document}
    found = re.findall(r"<!-- figure:([a-zA-Z0-9_]+) -->", body)
    if Counter(found) != Counter({name: 1 for name in selected}):
        raise ValueError(f"figure manifest and {document} document disagree: {found}")

    def replace(match):
        f = selected[match[1]]
        number = ("S" if document == "supplement" else "") + str(f["number"])
        caption = f"Figure {number}. {f['title']}. {f['caption']}"
        if mode == "mdx":
            return (f'<ZoomFigure src={{{f["id"]}}} alt={json.dumps(f["alt"])} variant="wide">\n'
                    f'  <p>{html.escape(caption, quote=False)}</p>\n</ZoomFigure>')
        return (f'<figure><img src="figures/{f["id"]}.png" alt="{html.escape(f["alt"], quote=True)}" />'
                f'<figcaption>{html.escape(caption, quote=False)}</figcaption></figure>')
    return re.sub(r"<!-- figure:([a-zA-Z0-9_]+) -->", replace, body)


def reading_document(resolved, figures, document):
    _, front, body = resolved.split("---", 2)
    metadata = yaml.safe_load(front)
    mdx_body = expand_figures(body, figures, document, "mdx")
    imports = "import '../../styles/snvReport.css';\nimport ZoomFigure from '../../components/ZoomFigure.astro';\n"
    imports += "\n".join(f"import {f['id']} from '../../assets/reports/full-snv-scoring-technical-report/{f['id']}.png';"
                         for f in figures if f["document"] == document)
    mdx = "---" + front + "---\n" + imports + "\n" + mdx_body
    markdown = "# " + metadata["title"] + "\n\n"
    if metadata.get("abstract"):
        markdown += "## Abstract\n\n" + metadata["abstract"] + "\n\n"
    markdown += expand_figures(body, figures, document, "markdown")
    if metadata.get("references"):
        markdown += "\n\n## References\n\n" + "\n\n".join(
            f"{i}. [{r['text']}]({r['doi']})" for i, r in enumerate(metadata["references"], 1))
    prose = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    return mdx, markdown, metadata["title"], len(prose.split())


def render_revision_figures(facts, snapshot, study, out, skip_scientific=False):
    if skip_scientific:
        # Progress is independent of the scientific cache and must always be fresh.
        concise_figures.style.apply_style()
        (out / "figures").mkdir(parents=True, exist_ok=True)
        concise_figures.progress_figure(facts, snapshot, out / "figures")
    else:
        concise_figures.render(facts, snapshot, study, out)


def protect_release(downloads, prefix):
    """Refuse a reused identity before changing any release files."""
    targets = [downloads / (prefix + suffix) for suffix in
               ("", ".html", "-supplement.html", ".zip", "-SHA256SUMS.txt", "-figure-audit.md")]
    if any(path.exists() for path in targets):
        raise ValueError("release identity already exists; choose a new --release-id")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--website-root", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-id", required=True, help="Unique download identity, e.g. 20260910-r2")
    parser.add_argument("--revision-date", required=True, help="Editorial date, YYYY-MM-DD; independent of progress")
    parser.add_argument("--skip-figures", action="store_true", help="Reuse scientific figures; always redraw progress")
    args = parser.parse_args(argv)
    study, out, website = args.study_dir.resolve(), args.output.resolve(), args.website_root.resolve()
    if out == study / "publication" or study / "publication" in out.parents:
        raise ValueError("revision output must not overwrite the frozen publication")
    frozen = study / "publication"
    before = digest(frozen / "study_facts.json")
    facts = json.loads((frozen / "study_facts.json").read_text())
    snapshot = json.loads(args.progress.read_text())
    context = template_context(facts, snapshot, args.release_id, args.revision_date)
    prefix = context["revision"]["prefix"]
    if prefix == "full-snv-concordance-20260909":
        raise ValueError("revision must use a new download prefix")
    downloads = website / "public/downloads"
    protect_release(downloads, prefix)
    out.mkdir(parents=True, exist_ok=True)
    figures = json.loads((TEMPLATES / "concise-figures.json").read_text())
    render_revision_figures(facts, snapshot, study, out, args.skip_figures)
    assets = website / "src/assets/reports/full-snv-scoring-technical-report"
    public_figures = downloads / prefix / "figures"
    public_figures.mkdir(parents=True, exist_ok=True)
    for f in figures:
        for suffix in (".png", ".pdf"):
            path = out / "figures" / (f["id"] + suffix)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"missing figure {path}")
            shutil.copy2(path, public_figures / path.name)
            if suffix == ".png":
                shutil.copy2(path, assets / path.name)
    (out / "figure_manifest.json").write_text(json.dumps(figures, indent=2) + "\n")
    audit = TEMPLATES / "scientific-review.md"
    (out / "figure_audit.md").write_text(audit.read_text() + "\n\n## Original-to-current figure mapping\n\n" + figure_audit() + "\n")
    shutil.copy2(out / "figure_audit.md", downloads / (prefix + "-figure-audit.md"))
    shutil.copy2(args.progress, out / "progress_snapshot.json")
    shutil.copy2(frozen / "study_facts.json", out / "study_facts.json")
    word_counts = {}
    for document, slug in SLUGS.items():
        resolved = report.resolve((TEMPLATES / (slug + ".mdx")).read_text(), context)
        mdx, markdown, title, words = reading_document(resolved, figures, document)
        word_counts[document] = words
        (website / "src/content/reports" / (slug + ".mdx")).write_text(mdx)
        (out / (slug + ".mdx")).write_text(mdx)
        (out / (slug + ".md")).write_text(markdown)
        standalone = report.render_html(markdown, out / "figures", title)
        # Absolute site links work when the bundled HTML is opened from disk.
        standalone = standalone.replace('href="/', 'href="https://khchao.com/')
        (out / (slug + ".html")).write_text(standalone)
        hosted = standalone
        for f in figures:
            image = out / "figures" / f["file"]
            embedded = 'src="data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode() + '"'
            hosted = hosted.replace(embedded, f'src="/downloads/{prefix}/figures/{image.name}"')
        (downloads / (prefix + ("-supplement" if document == "supplement" else "") + ".html")).write_text(hosted)
    # The reading copies are figures-only: an introduction plus one heading per figure, with the
    # explanation carried by the captions. The bound catches both an accidental return of the
    # removed prose and a lost introduction.
    if not 150 <= word_counts["main"] <= 1200:
        raise ValueError(f"main prose outside approved word budget: {word_counts['main']}")
    companion = "openspliceai-technical-report.mdx"
    companion_text = report.resolve((TEMPLATES / companion).read_text(), context)
    (website / "src/content/reports" / companion).write_text(companion_text)
    (out / companion).write_text(companion_text)
    # Keep plotting tables small; full arm tables retain their existing archives.
    for f in figures:
        for name in f["sources"]:
            source = frozen / name
            if name.startswith("tables/") and source.is_file():
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    (out / "README.md").write_text(
        "# Concise report revision\n\nScientific facts and original arm archives remain frozen at September 9. "
        "Progress is a separate metadata/scheduler observation. See figure_manifest.json for the data used by each plot. "
        "The tables/ directory contains selected plotting tables; complete tables remain in the four September 9 arm archives.\n\n"
        "Rebuild from a study workspace with: `python -m validation.concordance_study.concise_release "
        "--study-dir STUDY --website-root WEBSITE --progress SNAPSHOT.json --output NEW_OUTPUT --release-id UNIQUE_ID --revision-date YYYY-MM-DD`. "
        "The first display-histogram extraction reads the frozen primary summary; its cache is bound to SHA-256. "
        "That command requires the original study workspace. To redraw main figures from this bundle alone, "
        "load study_facts.json and progress_snapshot.json with json.load and display_histograms.npz with "
        "numpy.load(allow_pickle=False), then call the individual functions in concise_figures.py "
        "(score_figure, context_figure, position_figure, seed_figure, progress_figure); "
        "threshold_figure additionally accepts rows from tables/A_rs10_genomewide/agreement_curves.csv "
        "with cutoff, jaccard and call_rate_ratio_right_over_left converted to floats. "
        "S6/S7 include MAX arrays and left/right marginal counts. The full render entry point "
        "also needs the original September 9 figures for the retained supplementary images. "
        "Do not use prepare_release for this revision; it is the archived September 9 assembler.\n")
    (out / "verification.json").write_text(json.dumps(dict(frozen_facts_sha256=before, word_counts=word_counts,
        figure_counts=dict(Counter(f["document"] for f in figures)), progress_observed_at=snapshot["observed_at_utc"], release_id=args.release_id, revision_date=args.revision_date), indent=2) + "\n")
    evidence = out.parent / "verification"
    for name in ("scientific-checks.json", "claim-sources.csv", "numerical-checks.csv", "position_exception_records.csv", "position-exception-context.csv", "position_exception_maps.json"):
        source = evidence / name
        if source.is_file():
            target = out / "verification" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    files = {str(p.relative_to(out)): p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"}
    source_root = Path(__file__).resolve().parents[2]
    for package in ("concordance_study", "full_snv_concordance"):
        for p in (source_root / "validation" / package).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".mdx", ".json", ".sh", ".sbatch"):
                files["analysis_source/" + str(p.relative_to(source_root))] = p
    for p in (source_root / "tests/unit").glob("test_conc*.py"):
        files["analysis_source/" + str(p.relative_to(source_root))] = p
    files["analysis_source/validation/__init__.py"] = source_root / "validation/__init__.py"
    files["analysis_source/LICENSE"] = source_root / "LICENSE"
    archive(downloads / (prefix + ".zip"), files)
    (downloads / (prefix + "-SHA256SUMS.txt")).write_text("".join(
        f"{digest(p)}  {p.relative_to(downloads)}\n" for p in sorted(downloads.glob(prefix + "*")) if p.suffix in (".html", ".zip", ".md")))
    if digest(frozen / "study_facts.json") != before or digest(out / "study_facts.json") != before:
        raise ValueError("frozen scientific facts changed")
    print(json.dumps(dict(word_counts=word_counts, figures=len(figures), frozen_sha256=before), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
