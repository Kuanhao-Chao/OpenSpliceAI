> Latest revision: see `results/full_snv_concordance/study_20260908_depth/revision_20260910_r2/progress.md`. Use `concise_release` with an explicit unique release ID for current documents. Instructions below describe historical releases; `prepare_release` is the archived September 9 assembler.

# September 8 study continuation

## Scope and verified starting point

Continue Claude's September 8 plan, `i-would-like-to-steady-pony.md`: complete the
four-arm CPU analysis, annotated-site arm, redesigned figures, standalone study,
and new website technical report. The user explicitly chose the complete
event-level analysis, including an additional CPU pass. Finishing the GPU
scoring campaign is outside this continuation.

Verified September 8, 2026: the `study_20260908` run has 947 successful maps,
four successful reducers, and four successful renderers (Slurm exit 0:0).
Arm A contains 99,283 chunks / 3,334,708,099 exact-gene pairs; B/C/D contain
45,589 chunks each and 1,536,318,266 / 1,536,317,500 / 1,536,318,192 pairs.
All are provisional. The focused baseline is 54 passing tests and clean Ruff.
The frozen package digest is
`23ab219e25cb3bcf66786403848658ecd78fbe4f2dff51583c9aff974ca41e69`;
the annotation digest is
`e1aad08cf529cbf1310eac0f20344c23a4af7f54b0608fdbe94f2ce1af42d3b4`.

## Work sequence

1. Preserve existing runs and the pre-existing scorer changes. Add a versioned
   depth pass using the frozen A and B pair lists and existing audit/boundary
   contracts. Process B/C/D together on the exact three-way variant/gene
   intersection; equal chunk domains alone do not guarantee equal observations.
2. Add AG/AL/DG/DL statistics within chromosome, gene, block, substitution,
   dominant-event and site-distance strata; add site-conditioned score
   histograms and DP agreement. Keep 200 bins, thresholds
   0.05/0.1/0.2/0.5/0.8, 5,000 hash samples and 30,000 top discrepancies.
3. Test parsing, merges, boundaries, provenance and three-way membership; run
   a real-data pilot before CPU production. Freeze the core and depth code
   before submission and record jobs and resource measurements.
4. Complete dynamic-threshold reporting, per-event operating-point transfer,
   discordance distance taxonomy, collapsed-view analysis and matched stratum
   comparisons. Correct obsolete histogram diagnostics and seed-arm labels.
5. Independently recompute the 40-chunk validation window and new distance/event
   statistics. Reconcile exact counts and numerical tolerances before writing
   conclusions.
6. Render and visually inspect all figures, including the missing distance
   distributions and conditioned calibration/DP panels. Write both documents
   from one fact base; reject unresolved placeholders and INTERPRET markers.
7. Update the website's existing report and add the public, non-Scholar full-SNV
   report, run the website gates, deploy and verify the resulting pages.

## Interpretation contracts

SpliceAI is a comparator. Proximity is descriptive annotation context, not a
truth label or measured sensitivity/specificity. A symmetric 1–2 bp bin is not
an essential-dinucleotide label, and >500 bp from a site does not establish an
intronic location. The distance index pools internal splice boundaries across
genes on each contig, excludes transcript ends, and therefore differs from the
scorer's per-gene mask that includes transcript ends. Both limitations must be
visible. The top-k taxonomy describes selected extreme disagreements, not
their genome-wide prevalence. Retain legacy provenance classifications.

## Progress

- [x] Recover latest plan and session handoff; inspect code and run artifacts.
- [x] Confirm Slurm completion, code/annotation digests, focused tests and lint.
- [x] Implement and test complete depth aggregation (81 focused tests; final release rerun recorded separately).
- [x] Pilot, submit, and validate the additional CPU pass.
- [x] Complete derived analyses and visually review all 16 figures.
- [x] Independently verify expanded results (11,195 primary + 40,020 matched checks).
- [x] Finish both reports and standalone artifacts; pass all local release gates.
- [ ] Commit, push, deploy and verify the public release (prepared, not performed).

## September 9 implementation checkpoint

The original September 8 analysis is retained. The additional depth pass now has a
completed primary production reduction: 795 maps over 99,283 chunks, yielding the
same 3,334,708,099 primary pairs. Both 40-chunk verification reductions completed.
Matched production is complete: 365 maps over 45,589 shared chunks, with
1,536,316,689 exact three-way observations in each B/C/D comparison. Do not launch
a duplicate run. Jobs: primary 30738286/30738287; matched 30738288/30738289.

Independent verifier v2 uses separate parsing, grouping, site derivation and
arithmetic. It checks every moment sum and exact count, global/site marginal and
joint histograms, DP counts, and AG/AL/DG/DL/MAX across six stratum dimensions.
Primary: 1,373,220 observations, 11,195 checks passed; matched: 1,373,355 common
observations, 40,020 checks passed. Floating tolerances are relative 1e-9 and
absolute 1e-10; maximum differences in raw sums are 1.57e-9 and 1.04e-8. Identity
checks bind cached data to requested inputs, kind, verifier code and configuration,
and bind the result to the compared summary digest. Empty/incomplete checks fail.

Review found an inherited reducer edge case: a variant filling a whole chunk at
a coverage gap could survive through its next fragment. The implementation now
excludes all fragments of an incomplete variant. Regression tests cover both
sides. All 795 primary and 365 matched maps were inspected, along with both verification
windows: no whole-chunk groups occur. This defect does not affect any frozen result
used in the final report. See verification/boundary_impact.json.

Reporting now carries explicit left/right labels for seed exports, all configured
thresholds, 0.005-precise cutoffs and n/a for undefined DP rates. Every event stratum
has a companion CSV. The complete standalone and new MDX templates are drafted;
the existing report is condensed in a third template. All are resolved from one
fact base. Scientific interpretation distinguishes variant proximity, selected DP
position, per-gene masking, pooled internal sites, and experimental truth.

Scoped analysis changes are synchronized back to this workspace, with a pre-sync
snapshot under the plan ledger. Pre-existing scorer edits remain untouched. Frozen
run code is never edited. The earlier /tmp worktrees were on another cluster node;
use the durable release workspace below.

## Release preparation

The full production fact base and all standalone documents are in
`results/full_snv_concordance/study_20260908_depth/publication/`.
All 16 figures have PNG and vector PDF copies; all 104 CSV tables are present.
The existing architecture report and new full-SNV report are resolved from the
same fact base, including full figure captions. The new report is public and
non-Scholar; its sitemap, PDF and robots allowlists are prepared together.

Visual review corrected block placement within chromosome panels, moved a
quantization legend off the bars and distinguished overlapping series by line
style. The annotation-context analysis deliberately avoids calling symmetric
1–2 bp bins essential dinucleotides, distant variants deep-intronic, or call
rates sensitivity/specificity. Those claims in the initial plan are not supported
by the available annotation and lack of experimental truth labels.

Run `python -m validation.concordance_study.prepare_release --study-dir <study>
--website-root <website>` after rendering to regenerate both MDX documents,
standalone HTML and checksummed downloads. The main archive includes facts,
verification receipts, frozen execution inputs, source, shared-stratum CSV and
all figures. Four companion archives contain per-arm CSVs. Raw score VCFs and
multi-gigabyte reduced summaries remain on cluster storage.

Durable website worktree:
`results/full_snv_concordance/study_20260908_depth/website-release/`, branch
`codex/full-snv-depth-release`, based on website main `69395807`.
Final release gates run as Slurm CPU job **30768732** (8 CPUs, 32 GB, no GPU) because
long-running login-node processes were terminated. Progress and logs are in the
study's `verification/release-gates.tsv` and `verification/*.log`.
The script `release-gates.sh` runs focused Python tests, Ruff, packaging, website
check/tests/build/PDF, indexing, links, references, datasets, ML curriculum and
security audits. All gates, including browser review, passed at 2026-09-10
00:30:42 UTC. The resulting build is `build-30768732/`. A separate CPU job,
30768733, ran all nine source test modules from the isolated checkout:
110 tests passed in 35.38 seconds. See `RELEASE_READY.md` for the release handoff.


The release review corrected source-universe and deterministic-runtime wording:
coverage refers to the published genic SNV collection, not every possible GRCh38
substitution, and the runtime settings do not promise cross-hardware equality.
All 16 display captions are sequential. Browser checks passed for both reports
in desktop/light, phone/light and phone/dark views, including figure decoding,
zoom/close, metadata, horizontal overflow and print table widths. The final
job repeated these checks on the fully regenerated release and passed. Earlier builds found
and resolved a shared-mount npm execution issue, embedded hosted image URLs
rejected by security policy, and relative URLs inconsistent with the link audit.
Offline standalone copies retain embedded images inside the checksummed ZIP.

The analysis release is isolated on `codex/snv-depth-release`, based on
OpenSpliceAI `074d3e2`, at the study's `analysis-release/` worktree. It includes
both validation packages and nine relevant test files. The six focused suites
contain 81 passing tests; the three companion package suites add 29 passing tests.
Source VCFs, frozen execution code, and the root workspace's existing scorer
edits are excluded from these release-branch changes.
