"""Fact base, tables, and document rendering for the synthesis report.

The narrative is written as a template containing **no literal numbers**: every
quantity appears as a ``{{path.to.fact|format}}`` placeholder resolved against
``study_facts.json``. An unresolved or misspelled placeholder is a hard error, so
a number can never drift away from the run that produced it, and the prose cannot
quietly outlive a re-run.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import html as _html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from . import deeper, derive, markdown_tables
from .loading import EVENTS, SCORE_LABELS, Run

PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.:\[\]-]+)(?:\|([^}]+))?\}\}")

COVERAGE_FIELDS = (
    "source_rows", "variant_groups", "annotation_observations",
    "left_valid_annotations", "right_valid_annotations", "paired_annotations",
    "left_only_annotations", "right_only_annotations",
    "groups_without_right_prediction", "duplicate_annotations",
    "invalid_annotations", "left_conflicts", "right_conflicts",
    "excluded_incomplete_edge_fragments",
    "variant_collapsed_pairs", "variant_collapsed_left_only",
)


# --------------------------------------------------------------------------
# Fact base
# --------------------------------------------------------------------------
def _binning_artifact(run: Run) -> Dict[str, Any]:
    """Describe support without misclassifying quantization gaps as a bug.

    A 0.005 histogram of two-decimal scores naturally has unoccupied bins.
    Their neighbouring counts cannot establish how many values were misbinned.
    """
    bins = run.score_bins
    left = run.score_hist("MAX")["left"]
    empty = [index for index, count in enumerate(left) if count == 0]
    return {
        "bins": bins,
        "width": 1 / bins,
        "policy": "epsilon-floor-v1" if "depth" in run.raw else "unversioned",
        "empty_left_bins": empty,
        "empty_left_bin_count": len(empty),
        "support_note": "Empty bins can reflect the score grid or lack of support; they do not measure a binning defect.",
    }


def _run_facts(run: Run, sites=None) -> Dict[str, Any]:
    # Coverage is a Counter on the reducer side: a key that never fired is absent
    # rather than zero, so every read here is defensive and zero-filled.
    coverage = {key: 0 for key in COVERAGE_FIELDS}
    coverage.update(run.coverage)
    paired = coverage["paired_annotations"]
    source_rows = coverage["source_rows"]
    facts: Dict[str, Any] = {
        "arm": run.arm,
        "role": run.role,
        "comparison": run.comparison,
        "left_label": run.left_label,
        "right_label": run.right_label,
        "kind": run.kind,
        "run_label": run.run_label,
        "chunks": run.chunk_count,
        "finality": dict(run.finality),
        "provenance": dict(run.provenance),
        "coverage": coverage,
        "coverage_rates": {
            "right_annotation_rate": (coverage["right_valid_annotations"] / coverage["left_valid_annotations"])
            if coverage["left_valid_annotations"] else float("nan"),
            "paired_share_of_source_rows": (paired / source_rows) if source_rows else float("nan"),
            "paired_share_of_left_annotations": (paired / coverage["left_valid_annotations"])
            if coverage["left_valid_annotations"] else float("nan"),
            "groups_without_prediction_share": (coverage["groups_without_right_prediction"] / coverage["variant_groups"])
            if coverage["variant_groups"] else float("nan"),
        },
        "agreement": {row["label"]: row for row in derive.agreement_table(run)},
        "thresholds": {},
        "signal_subsets": {row["label"]: {} for row in derive.signal_subset_table(run)},
        "quantization": {row["label"]: row for row in derive.quantization_profile(run)},
        # the 2-dp threshold tables, so the report can show that output precision does
        # not explain the call-rate difference
        "right_rounded_2dp_thresholds": run.metrics["right_rounded_2dp"]["thresholds"],
        "dp": {},
        "histogram_binning": _binning_artifact(run),
        "bootstrap": run.metrics["cluster_bootstrap_95ci"],
        "approximation": run.metrics["approximation_metadata"],
        "estimand": run.metrics["estimand"],
    }
    if run.is_concordance:
        facts["coverage_rates"]["paired_share_of_spliceai_annotations"] = facts["coverage_rates"]["paired_share_of_left_annotations"]
    for row in derive.threshold_table(run):
        facts["thresholds"].setdefault(row["label"], {})[f"{row['threshold']:g}"] = row
    for row in derive.signal_subset_table(run):
        facts["signal_subsets"][row["label"]][row["subset"]] = row
    for row in derive.dp_table(run):
        facts["dp"].setdefault(row["event"], {})[f"{row['threshold']:g}"] = row

    dominant_rows, dominant_summary = derive.dominant_table(run)
    facts["dominant"] = {"matrix": dominant_rows, "summary": dominant_summary,
                         "row_normalized": run.metrics["dominant_normalized"]["row_normalized"]}

    # Analyses added for the deeper study. `site_distance` is present only when the
    # run carried the opt-in stratum, so downstream templates must tolerate its absence.
    site_rows = derive.site_distance_table(run)
    if site_rows:
        facts["site_distance"] = {
            "pooled": {row["distance"]: row for row in site_rows},
            "order": [row["distance"] for row in site_rows],
            "by_type": {
                site_type: {row["distance"]: row
                            for row in derive.site_distance_table(run, site_type=site_type)}
                for site_type in derive.SITE_TYPES
            },
        }
    facts["tail_asymmetry"] = {
        label: derive.tail_asymmetry(run, label) for label in SCORE_LABELS
    }
    facts["discordance"] = derive.discordance_taxonomy(run) if run.is_concordance else {}
    landscape = derive.genomic_landscape(run)
    facts["landscape"] = {
        "blocks": len(landscape),
        "most_negative": sorted(landscape, key=lambda r: r["bias"])[:15],
        "dispersion": derive.stratum_dispersion(landscape, "bias"),
    }
    curve = derive.same_cutoff_curve(run)
    defined = [p for p in curve["points"] if p["mcc"] is not None]
    best = max(defined, key=lambda p: p["mcc"]) if defined else None
    facts["agreement_curve"] = {
        "points": len(curve["points"]),
        "best_mcc": best,
        "crossover_cutoff": next(
            (p["cutoff"] for p in curve["points"]
             if (p["call_rate_ratio_right_over_left"] or 0) >= 1.0),
            None,
        ),
    }

    facts["operating_point"] = {
        f"{row['spliceai_threshold']:g}": row
        for row in derive.operating_point_transfer(run)
    } if run.is_concordance else {}
    facts["operating_point_by_event"] = deeper.operating_points(run) if run.is_concordance else {}
    facts["collapsed"] = deeper.collapsed_table(run)
    facts["configured_thresholds"] = run.raw["config"]["thresholds"]
    facts["site_event_table"] = deeper.site_event_table(run)
    facts["site_dp_table"] = deeper.site_dp_table(run)
    facts["discordance_depth"] = deeper.discordance(run, sites) if sites is not None else {}
    facts["matched_domain"] = run.summary.get("matched_domain")
    facts["dominant_pair_strata"] = derive.stratum_rows(run, "dominant_pair")

    facts["strata"] = {}
    for dimension in ("chrom", "gene", "block_1mb", "substitution", "dominant_pair", "site_event"):
        if dimension not in run.metrics["strata"]:
            continue
        rows = derive.stratum_rows(run, dimension)
        facts["strata"][dimension] = {
            "count": len(rows),
            "dispersion_mae": derive.stratum_dispersion(rows, "mae"),
            "dispersion_bias": derive.stratum_dispersion(rows, "bias"),
            "top_by_mae": sorted(rows, key=lambda r: -r["mae"])[:25],
            "largest": rows[:25],
        }
    return facts


def build_facts(runs: Mapping[str, Run], *, primary: str, seed_arm: str | None,
                model_arms: Sequence[str], study_meta: Mapping[str, Any], sites=None) -> Dict[str, Any]:
    facts: Dict[str, Any] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "study": dict(study_meta),
        "arms": {arm: _run_facts(run, sites) for arm, run in runs.items()},
        "primary_arm": primary,
    }
    if seed_arm and all(arm in runs for arm in model_arms):
        facts["seed_versus_model"] = derive.seed_versus_model(
            runs[seed_arm], [runs[arm] for arm in model_arms]
        )
        rows = deeper.matched_strata(runs[seed_arm], [runs[arm] for arm in model_arms])
        facts["seed_versus_model"]["strata"] = rows
        facts["seed_versus_model"]["stratum_summary"] = deeper.stratum_summary(rows)
    if primary in runs and model_arms and model_arms[0] in runs:
        facts["generalization"] = derive.generalization(runs[primary], runs[model_arms[0]])
    facts["primary"] = facts["arms"][primary]
    return facts


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_strict(value), sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list)) else _strict(value)
                for key, value in row.items()
            })


def write_tables(runs: Mapping[str, Run], out_dir: Path, sites=None) -> List[str]:
    written: List[str] = []
    for arm, run in runs.items():
        base = out_dir / arm
        jobs = {
            "agreement.csv": derive.agreement_table(run),
            "thresholds.csv": derive.threshold_table(run),
            "signal_subsets.csv": derive.signal_subset_table(run),
            "dp_agreement.csv": derive.dp_table(run),
            "quantization.csv": derive.quantization_profile(run),
            "dominant_event.csv": derive.dominant_table(run)[0],
            "site_event.csv": deeper.site_event_table(run),
            "site_dp.csv": deeper.site_dp_table(run),
            "variant_collapsed.csv": deeper.collapsed_table(run),
            "agreement_curves.csv": [{"event": e, "left_label": run.left_label,
                                      "right_label": run.right_label, **r} for e in SCORE_LABELS
                                      for r in derive.same_cutoff_curve(run, e)["points"]],
        }
        for dimension in run.metrics.get("strata", {}):
            jobs[f"stratum_{dimension}.csv"] = derive.stratum_rows(run, dimension)
        for dimension in run.metrics.get("event_strata", {}):
            jobs[f"event_stratum_{dimension}.csv"] = derive.event_stratum_rows(run, dimension)
        if sites is not None:
            taxonomy = deeper.discordance(run, sites)
            jobs["discordance_taxonomy.csv"] = taxonomy["strata"]
            jobs["discordance_records.csv"] = taxonomy["record_table"]
        if run.is_concordance:
            flat = []
            for event in SCORE_LABELS:
                for row in derive.operating_point_transfer(run, event):
                    for variant in ("identical_cutoff", "rate_matched", "agreement_optimal"):
                        flat.append({"event": event, "left_label": run.left_label,
                                     "right_label": run.right_label,
                                     "spliceai_threshold": row["spliceai_threshold"],
                                     "strategy": variant, **row[variant]})
            jobs["operating_point_transfer.csv"] = flat
        for name, rows in jobs.items():
            _write_csv(base / name, rows)
            written.append(str((base / name).relative_to(out_dir)))
    return written


# --------------------------------------------------------------------------
# Placeholder resolution
# --------------------------------------------------------------------------
_PATH_TOKEN = re.compile(r"\[([^\]]+)\]|([^.\[\]]+)")


def _path_tokens(dotted: str) -> List[str]:
    """Split a fact path, allowing bracketed segments for keys containing dots.

    Threshold and tolerance keys are literally ``"0.5"`` and ``"0.01"``, so a
    plain dotted path cannot reach them. ``a.b[0.5].c`` can.
    """
    return [bracketed or plain for bracketed, plain in _PATH_TOKEN.findall(dotted)]


def _lookup(facts: Mapping, dotted: str) -> Any:
    node: Any = facts
    for part in _path_tokens(dotted):
        if isinstance(node, Mapping):
            if part not in node:
                raise KeyError(f"unknown fact path: {dotted} (missing {part!r})")
            node = node[part]
        elif isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            raise KeyError(f"unknown fact path: {dotted} (cannot descend into {type(node).__name__})")
    return node


def _format(value: Any, spec: str | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if spec is None:
        if isinstance(value, float):
            return f"{value:,.4g}"
        if isinstance(value, int):
            return f"{value:,d}"
        return str(value)
    if spec == "pct1":
        return f"{value * 100:.1f}%"
    if spec == "pct2":
        return f"{value * 100:.2f}%"
    if spec == "pct3":
        return f"{value * 100:.3f}%"
    if isinstance(value, float) and spec.endswith("d"):
        value = int(round(value))
    return format(value, spec)


def resolve(text: str, facts: Mapping) -> str:
    """Substitute every ``{{fact.path|format}}`` and ``{{table:name}}``.

    Fails closed: an unknown fact path or table name raises rather than leaving a
    visible placeholder in a published document.
    """
    if re.search(r"<!--\s*INTERPRET:", text):
        raise ValueError("report contains unfinished INTERPRET sections")
    missing: List[str] = []

    def _substitute(match: re.Match) -> str:
        path, spec = match.group(1), match.group(2)
        try:
            if path.startswith("table:"):
                return markdown_tables.render(path[len("table:"):], facts)
            return _format(_lookup(facts, path), spec)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            missing.append(f"{path} ({exc})")
            return match.group(0)

    rendered = PLACEHOLDER.sub(_substitute, text)
    missing.extend("unsupported placeholder syntax: " + token for token in re.findall(r"\{\{[^{}]*\}\}", rendered))
    if missing:
        raise KeyError("unresolved placeholders:\n  " + "\n  ".join(missing))
    return rendered


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------
_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap");

/* Palette: cool-biased neutrals (the accent is blue, so the greys lean blue rather
   than sitting at a pure mid-grey), with the two accents taken directly from the
   figures' validated categorical slots -- SpliceAI blue and OpenSpliceAI orange --
   so the prose, the tables and the charts read as one system. */
:root{
  color-scheme: light;
  --ground:#f7f8fa; --card:#ffffff; --raised:#eef1f6;
  --ink:#12151a; --ink-2:#49515f; --ink-3:#838b99;
  --rule:#e0e4ea; --rule-strong:#c9cfd9;
  --accent:#2a78d6; --accent-2:#eb6834;
  --serif:"Spectral",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --measure:68ch;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#14171c; --card:#1b1f26; --raised:#232832;
    --ink:#f1f3f7; --ink-2:#b2b9c6; --ink-3:#7b8393;
    --rule:#2b303a; --rule-strong:#3b424f;
    --accent:#5a9bea; --accent-2:#ef7f52;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#14171c; --card:#1b1f26; --raised:#232832;
  --ink:#f1f3f7; --ink-2:#b2b9c6; --ink-3:#7b8393;
  --rule:#2b303a; --rule-strong:#3b424f;
  --accent:#5a9bea; --accent-2:#ef7f52;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.68;
  -webkit-font-smoothing:antialiased;
}
main{
  /* A narrow measure for prose; tables and figures break out to the full width,
     because dense numeric data is scanned, not read at 68 characters. */
  max-width:78rem; margin:0 auto; padding:4rem 2rem 7rem;
  display:flex; flex-direction:column; gap:0;
}
main > p, main > ul, main > ol, main > blockquote{max-width:var(--measure)}
main > h1, main > h2, main > h3, main > h4{max-width:var(--measure)}

h1{
  font-weight:700; font-size:clamp(1.9rem,3.4vw,2.7rem); line-height:1.14;
  letter-spacing:-.018em; text-wrap:balance; margin:0 0 1.1rem;
}
h2{
  font-weight:600; font-size:1.52rem; line-height:1.25; letter-spacing:-.012em;
  text-wrap:balance; margin:3.6rem 0 .5rem; padding-top:1.5rem;
  border-top:1px solid var(--rule);
}
h3{font-weight:600; font-size:1.18rem; letter-spacing:-.006em; text-wrap:balance; margin:2.3rem 0 .4rem}
h4{font-family:var(--sans); font-weight:600; font-size:.86rem; letter-spacing:.06em;
   text-transform:uppercase; color:var(--ink-2); margin:1.7rem 0 .35rem}
p{margin:0 0 1.05rem}
strong{font-weight:600}
a{color:var(--accent); text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:3px; border-radius:3px}
ul,ol{margin:0 0 1.1rem; padding-left:1.35rem}
li{margin:.3rem 0}
li::marker{color:var(--ink-3)}

blockquote{
  margin:1.5rem 0; padding:.85rem 1.15rem; background:var(--raised);
  border-left:2px solid var(--accent); border-radius:0 3px 3px 0;
  font-style:italic; color:var(--ink-2);
}

code{font-family:var(--mono); font-size:.84em; background:var(--raised);
     padding:.1em .34em; border-radius:3px; word-break:break-word}
pre{font-family:var(--mono); background:var(--raised); color:var(--ink);
    padding:1rem 1.15rem; border-radius:5px; overflow-x:auto; font-size:.82rem;
    line-height:1.55; border:1px solid var(--rule); margin:1.3rem 0; max-width:100%}
pre code{background:none; padding:0; font-size:1em}

/* Tables are the substance of this document, so they get the utility face,
   lining figures and a scroll container of their own. */
.tablewrap{overflow-x:auto; margin:1.5rem 0; border:1px solid var(--rule);
           border-radius:5px; background:var(--card)}
table{border-collapse:collapse; width:100%; font-family:var(--sans);
      font-size:.815rem; line-height:1.45; font-variant-numeric:tabular-nums lining-nums}
th,td{padding:.5rem .8rem; text-align:right; white-space:nowrap;
      border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left; white-space:normal; min-width:11rem}
thead th{background:var(--raised); color:var(--ink-2); font-weight:600;
         font-size:.74rem; letter-spacing:.045em; text-transform:uppercase;
         border-bottom:1px solid var(--rule-strong); position:sticky; top:0; z-index:1}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--raised)}
td code{font-size:.78em}

figure{margin:2rem 0}
/* Charts are rendered on the light chart surface. In dark mode they sit on a light
   card rather than being inverted: an automatic flip is not a selected palette. */
figure img{display:block; width:100%; height:auto; background:#fcfcfb;
           border:1px solid var(--rule); border-radius:5px; padding:.6rem}
figcaption{margin-top:.6rem; font-family:var(--sans); font-size:.8rem;
           line-height:1.5; color:var(--ink-2); max-width:var(--measure)}

small{font-family:var(--sans); font-size:.78rem; color:var(--ink-3)}
hr{border:none; border-top:1px solid var(--rule); margin:2.6rem 0}

/* Masthead: the one place the page spends any visual weight. */
.masthead{margin:0 0 2.2rem; padding:0 0 1.6rem; border-bottom:2px solid var(--rule-strong)}
.masthead .eyebrow{font-family:var(--sans); font-size:.74rem; font-weight:600;
  letter-spacing:.13em; text-transform:uppercase; color:var(--accent); margin:0 0 .7rem}
.masthead h1{margin:0 0 .8rem; max-width:22ch}
.masthead .dek{font-size:1.06rem; line-height:1.55; color:var(--ink-2);
  max-width:var(--measure); margin:0 0 1.5rem}
.status{display:inline-flex; align-items:center; gap:.42rem; font-family:var(--sans);
  font-size:.71rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  padding:.26rem .6rem .26rem .5rem; border-radius:3px;
  background:var(--accent-2); color:#fff}
.status::before{content:""; width:.4rem; height:.4rem; border-radius:50%;
  background:#fff; opacity:.85}
.meta{display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  gap:1rem 1.6rem; margin:1.4rem 0 0; padding:0; list-style:none;
  font-family:var(--sans)}
.meta div{display:flex; flex-direction:column; gap:.16rem}
.meta dt{font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:var(--ink-3)}
.meta dd{margin:0; font-size:.95rem; font-weight:500; color:var(--ink);
  font-variant-numeric:tabular-nums lining-nums}
.meta dd.small{font-family:var(--mono); font-size:.72rem; font-weight:400;
  color:var(--ink-2); word-break:break-all}

/* Table of contents, emitted by the markdown toc extension. */
.toc{max-width:var(--measure); margin:2rem 0 0; padding:1.1rem 1.3rem;
     background:var(--card); border:1px solid var(--rule); border-radius:5px}
.toc > ul{margin:0; padding-left:1.1rem; font-family:var(--sans); font-size:.85rem}
.toc ul ul{padding-left:1.05rem}
.toc li{margin:.16rem 0}
.toc a{color:var(--ink-2); text-decoration:none}
.toc a:hover{color:var(--accent); text-decoration:underline}

@media (max-width:760px){
  body{font-size:16px}
  main{padding:2.5rem 1.1rem 4rem}
  th,td{padding:.42rem .6rem}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def _markdown_to_html(text: str) -> str:
    try:
        import markdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HTML report rendering requires Markdown; install openspliceai[analysis] or the dev extra"
        ) from exc
    return markdown.markdown(
        text,
        extensions=["tables", "toc", "attr_list", "fenced_code", "md_in_html"],
        extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
    )


def _embed_figures(body: str, figures_dir: Path) -> str:
    """Inline every referenced PNG so the document is self-contained.

    A missing figure is an error, not something to paper over: leaving the
    original ``src`` in place would publish a document with a broken image and a
    path that only resolves on the machine that built it.
    """
    missing: List[str] = []

    def _replace(match: re.Match) -> str:
        src = match.group(1)
        candidate = figures_dir / Path(src).name
        if not candidate.is_file():
            missing.append(src)
            return match.group(0)
        encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    rendered = re.sub(r'src="([^"]+\.png)"', _replace, body)
    if missing:
        raise FileNotFoundError(
            "report references figures that were not rendered: " + ", ".join(sorted(set(missing)))
        )
    return rendered


def render_html(markdown_text: str, figures_dir: Path, title: str) -> str:
    body = _embed_figures(_markdown_to_html(markdown_text), figures_dir)
    body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )


def _strict(value: Any) -> Any:
    """Replace every non-finite float with ``None`` before serialising.

    ``NaN`` and ``Infinity`` are not valid JSON. Emitting them would produce a
    fact file that some readers accept and others reject, so an undefined value
    is written as ``null`` and the encoder is run with ``allow_nan=False`` to
    guarantee nothing slipped through.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def render_artifact_html(markdown_text: str, figures_dir: Path, title: str) -> str:
    """The same document, shaped for the Artifact publisher.

    The publisher supplies the ``<!doctype>``/``<html>``/``<head>``/``<body>``
    skeleton and a small reset, so this emits only the title, the stylesheet and
    the page content. The palette blocks are unchanged: light values on bare
    ``:root``, redefined under both ``prefers-color-scheme`` and an explicit
    ``[data-theme]`` stamp, so the viewer's toggle wins in both directions.
    """
    body = _embed_figures(_markdown_to_html(markdown_text), figures_dir)
    body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    return (
        f"<title>{_html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        f"<main>\n{body}\n</main>\n"
    )


def write_facts(facts: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_strict(facts), indent=2, sort_keys=True, allow_nan=False))


__all__ = ["build_facts", "write_tables", "resolve", "render_html",
           "render_artifact_html", "write_facts",
           "EVENTS", "SCORE_LABELS"]
