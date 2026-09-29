# Full-SNV technical reports: release handoff

The CPU analysis, figures, reports, downloads and local release validation are
complete. The two isolated release branches are prepared for commit, push and
deployment. No release commit, push or deployment has been performed.

This implements Claude's `i-would-like-to-steady-pony.md` plan through the release
preparation stage. The frozen September 8 scoring snapshot is retained. Report
dates remain September 9, 2026; final validation completed September 10 UTC.

## What was completed

| Plan component | Implementation and outcome |
| --- | --- |
| Richer analysis | 200 score bins; exact thresholds 0.05/0.1/0.2/0.5/0.8; AG/AL/DG/DL/MAX statistics across six stratum dimensions; 5,000 uniform samples and 30,000 top discrepancies. |
| Annotated-site context | Strand-aware internal splice boundaries from the scoring annotation, with a digest in the merge contract; distance-conditioned scores, calibration and predicted-position agreement. |
| Matched seed/method comparison | One three-way variant/gene intersection feeds all three comparisons after duplicate resolution and chunk-boundary handling. |
| CPU production | Primary: 99,283 chunks and 3,334,708,099 pairs. Matched: 45,589 chunks and 1,536,316,689 common observations. Both reductions completed. |
| Independent verification | Separate parser, site derivation and arithmetic passed 11,195 primary and 40,020 matched checks; 19 independently recomputed headline statistics also agree. |
| Reporting | Per-event operating-point transfer, threshold curves, tail asymmetry, discordance taxonomy, collapsed view and matched stratum comparisons; 104 CSV tables. |
| Figures | 16 reviewed figures in PNG and vector PDF; sequential captions, improved chromosome placement and distinguishable overlapping series. |
| Documents | New full-SNV report, condensed companion architecture/benchmark report, standalone HTML/Markdown and embeddable artifact, all resolved from one fact base. |
| Website release | Public/non-Scholar metadata, sitemap/PDF/robots allowlists, working figure zoom, readable screen/print tables and five checksummed archives. |

The plan's suggested sensitivity/specificity proxies were not adopted: annotated
proximity is descriptive context, not experimental truth. Symmetric 1–2 bp bins
are not essential-dinucleotide labels, and a distance above 500 bp does not prove
an intronic location. The site index pools internal boundaries across genes;
the scorer masks against the nearest per-gene boundary, including transcript
ends. Both distinctions are documented. Genome-wide coverage refers to the
published genic SNV collection, not every possible GRCh38 substitution.

## Validation evidence

Evidence is under `results/full_snv_concordance/study_20260908_depth/verification/`.

| Check | Result | Evidence |
| --- | --- | --- |
| Isolated analysis checkout | 110 tests passed | `isolated-source-tests.log`, job 30768733 |
| Focused production/report suites | 81 tests passed; Ruff clean | `final-python-tests.log`, `final-ruff.log` |
| Website content/type checks | 0 errors, 0 warnings; 101 existing hints | `website-check.log` |
| Website tests | 63 files passed; 4,080 tests passed, 350 skipped | `website-tests.log` |
| Build and PDFs | Passed; full report 24 pages / 2,472,913 bytes, companion 13 pages / 820,938 bytes | `website-build.log` |
| Indexing, links, references, datasets, ML curriculum, security | All passed | `release-gates.tsv`, job 30768732 |
| Browser and print layout | Both reports passed desktop/light, phone/light, phone/dark, image decode, zoom/close, metadata and overflow checks; all 23 print tables fit | `report-browser-review.json`, `screenshots/` |
| Scientific/report review | No unresolved critical or important findings | `final-release-review.md` |
| Downloads | Every archive member and download manifest checksum verified | `archive_verification.json` |
| Production boundary audit | Every map checked; no triggering whole-chunk edge groups | `boundary_impact.json` |

CI additionally exercises the existing terminal, cell background, playground and
deep-dive UI in Chromium/WebKit. Those broader CI smoke suites have not been run
locally for this release. The report-specific local browser checks use Chromium.

## Release workspaces and artifacts

All paths below are relative to the OpenSpliceAI repository root.

- Source: `results/full_snv_concordance/study_20260908_depth/analysis-release/`,
  branch `codex/snv-depth-release`, base `074d3e27d9950f48e7276ed69f0f0205f10ef223`.
  Scope: both validation packages and nine relevant test modules.
- Website: `results/full_snv_concordance/study_20260908_depth/website-release/`,
  branch `codex/full-snv-depth-release`, base
  `69395807331e2c2660da0cb85ef3a60f367c0d5e`.
- Built site and report PDFs:
  `results/full_snv_concordance/study_20260908_depth/build-30768732/`.
- Standalone study, fact base, tables and figures:
  `results/full_snv_concordance/study_20260908_depth/publication/`.
- Hosted HTML, five ZIPs, figures and the download checksum manifest:
  `website-release/public/downloads/` beneath the study directory.

The main ZIP includes facts, all figures, the shared-stratum CSV, reports,
analysis source, frozen execution inputs and numerical verification receipts.
Four companion ZIPs contain per-arm CSVs. Every archive is below 60 MB. Offline
HTML embeds its figures; the hosted reading copy uses ordinary image URLs.

The root checkout's pre-existing `openspliceai/variant/utils.py` and `variant.py`
edits are preserved and excluded from the source release. Raw score VCFs, frozen
execution code and summaries remain unchanged. No GPU scoring was submitted.

## Commit, push and deployment sequence

Review each staged diff in its named worktree, then commit separately:

```bash
# In analysis-release/
git diff --cached --check
git diff --cached --stat
git commit -m "Add verified full-SNV depth analysis and report generation"
git push -u origin codex/snv-depth-release

# In website-release/
git diff --cached --check
git diff --cached --stat
git commit -m "Publish full-SNV concordance reports and reproducible study downloads"
git push -u origin codex/full-snv-depth-release
```

Merge the source branch through the repository's normal review process. For the
website, merge the release branch into `main` to trigger GitHub Pages. Recheck
remote `main` first; if it changed, integrate it and repeat affected checks.
Do not force-push over concurrent work. If publishing by direct fast-forward is
preferred, `git push origin HEAD:main` from the website release branch triggers
the same workflow when remote `main` is still an ancestor.

Watch `.github/workflows/deploy.yml` through successful build and deploy jobs.
Then verify the two report URLs, their PDF links, both cross-links, all 16 new
figures, the standalone HTML and download checksum manifest, sitemap/robots
visibility and absence of Scholar citation tags. Expected report paths are
`/reports/full-snv-scoring-technical-report/` and
`/reports/openspliceai-technical-report/` on `https://khchao.com`.

Record the source/website commit IDs, Actions URL and live verification outcome
after publication. A passing local build is not a deployed release.

## Remaining scientific status

All results remain **provisional** because the scoring campaign is incomplete.
Legacy VCFs are structurally audited but lack embedded model receipts; the audit
does not cryptographically establish their checkpoint identity. SpliceAI is a
comparator, and the study does not establish biological accuracy or isolate
implementation effects from training, ensembling and precision differences.
