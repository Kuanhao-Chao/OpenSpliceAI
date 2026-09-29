# Scientific and editorial audit: September 10, revision 2

This audit accompanies the six-figure main report and eleven-figure supplement. The scientific
snapshot remains September 9. This revision changes presentation and interpretation without
rerunning scoring or replacing frozen scientific aggregates.

## Baseline and scientific domains

The latest inspected Claude Code session is `4223e752-ebda-4ac9-95a6-e3f399aa19d8`.
It records website commit `8da4d712` and a successful September 10 publication. The website repository
contains that commit and its saved build gates passed. The earlier execution ledger incorrectly
still described publication as pending; it is superseded by the revision ledger.

Primary observations are 3,334,708,099 paired variant–gene annotations from 99,283 rs10 chunks.
The three matched contrasts use exactly 1,536,316,689 observations from 45,589 common chunks.
The frozen facts SHA-256 is `51c6ae93c09e0652e39004a846258b8b7a0bf1e890123e698373d397f46c8c06`.
A VCF row, variant group and gene annotation are different counting units. Unpaired values are not
zeros. Legacy files have structural audit evidence, but not embedded checkpoint provenance.

## Confirmed corrections and their effect

- The 1–2-bp AL call ratio is 1,005,465 / 1,032,871 = 0.973466..., displayed as 0.973. The prose now derives boundary ratios directly from pooled counters, avoiding double rounding.

- Loss-position agreement is not literally perfect. At 0.5, AL has 1,441,729 exact matches among
  1,441,730 jointly called annotations; DL has 1,630,983 among 1,630,987. The report now exposes
  the one and four mismatches instead of rounding them to 100.00%. These exceptions do not alter
  the broader observation of near-complete loss-position agreement.
- Context pooling includes ambiguous boundary types as well as acceptors and donors. Counts are
  pooled before division, preserving the shared population denominator for both models.
- The quantization panel is a ratio of residual MAE to original MAE. It is not the fraction of
  variants whose disagreement survives rounding and does not identify rounding's causal contribution.
- Absent jointly called distance groups have undefined position agreement. Fixed x categories
  preserve the same distance layout across events; absent groups are labelled explicitly.
- Gene-level ratio panels disclose eligibility and exclusions. Very small seed denominators can
  generate extreme ratios. The gene-mean scatter contains mathematical coupling between its axes;
  its distribution does not by itself establish a biological trend or exclude outliers.

## Source-record investigation of the five loss-position exceptions

All 795 primary map files were checked. Maps 660 and 775 contain the four DL and one AL mismatches,
respectively; the other maps contain none at cutoff 0.5. All 125 prediction VCFs referenced by each
of those maps were read and checked against their frozen SHA-256 receipts. Original SpliceAI VCFs
for the two affected chunks were also checksummed and their exact gene annotations verified.
Identical repeated records were collapsed. The five resulting records agree with the global and
context-specific exception counts.

| Variant | Gene | Event | SpliceAI score / DP | OpenSpliceAI score / DP |
|---|---|---|---|---|
| chr17:15440211 C>A | TVP23C-CDRT4 | DL | 0.82 / 0 | 0.59874 / −3 |
| chr17:15440220 T>A | TVP23C-CDRT4 | DL | 0.53 / −9 | 0.67779 / −12 |
| chr17:15440241 T>A | TVP23C-CDRT4 | DL | 0.72 / −30 | 0.68441 / −33 |
| chr17:15440242 C>A | TVP23C-CDRT4 | DL | 0.55 / −31 | 0.54908 / −34 |
| chrX:52866551 G>C | XAGE3 | AL | 0.58 / −15 | 0.86068 / −13 |

All four DL records select chr17:15,440,211 in SpliceAI and chr17:15,440,208 in OpenSpliceAI.
The AL record selects chrX:52,866,536 and chrX:52,866,538, respectively. OpenSpliceAI's selected
positions equal the nearest gene boundaries in the campaign annotation (`data/grch38_chr.txt`);
the SpliceAI positions do not. The position-exception CSV records annotation and VCF hashes.
This is direct evidence of localized discrepancies in retained loss positions. It does not identify
the historical processing step responsible and does not prove which position is biologically used.
Universal claims of strictly identical masking behavior have been removed from both reports and
the companion's campaign summary.

## Figure-by-figure evidence and decisions

| Figure | Source and calculation | Supported result | Decision and limit |
|---|---|---|---|
| 1 | Progress manifests, current file metadata, dated scheduler query; frozen domain counts | rs10 and rs13 scoring remain incomplete | Retain; dates and state categories separate; no completion forecast |
| 2 | Stored joint histograms; exact moments in agreement.csv | Loss correlations exceed gain correlations; DG is lowest | Retain four event panels; distribution bands are not confidence intervals |
| 3 | thresholds.csv and agreement_curves.csv | At 0.5, losses overlap more and OpenSpliceAI calls more losses/fewer gains | Retain; common cutoff, exact 0.5 markers, approximate curves; no accuracy ranking |
| 4 | site_event.csv pooled over all site types; near/far histograms | Loss ratios differ by distance; distant DG call reduction exceeds AG | Clarify boundary sets and denominators; variant proximity is not intronic status |
| 5 | dp_agreement.csv integer counts and rates | Shared gains usually select the same site; loss agreement has five exceptions | Correct rounded perfection; retain exact counts and conditional eligibility |
| 6 | Identical B/C/D domains and MAEs | Gain method/seed ratios exceed loss ratios | Retain; two checkpoints give one observed contrast, not training variance |
| S1 | coverage counters | Counting units and pairing exclusions differ | Redraw with model names and separate units |
| S2 | Signed-difference histogram tails | Directional differences depend on event and magnitude | Retain original data/figure; describe tails without inferring true effects |
| S3 | operating_point_transfer.csv | Equal call volume does not ensure equal membership | Retain; MCC uses SpliceAI as comparator, not experimental truth |
| S4 | quantization.csv; residual MAE / raw MAE | Exact matches depend on precision; residual score differences remain | Redraw 0–1 axes; define operational allowance explicitly |
| S5 | Integer dominant-event matrix divided by row totals | Dominant-label agreement varies across categories | Redraw with row counts; blank labels mean <0.005, not zero |
| S6 | Near/far marginal histograms | Event score distributions differ with context | Redraw on shared log scale; show population sizes and zero behavior |
| S7 | Near/far conditional histograms | Gain medians are closer to equality near boundaries over much of the range | Redraw with shared axes and population sizes; empty bins remain gaps |
| S8 | site_dp.csv; pooled integer eligible/within counts | Position agreement is conditional on shared events | Redraw fixed categories, exact mismatch counts and absent-group labels |
| S9 | seed_versus_method_strata.csv; genes n≥1,000 and positive defined ratios | Gain ratios exceed one for almost all plotted genes; loss ratios are mixed | Redraw shared log axes, counts and exclusion summary |
| S10 | stratum_block_1mb.csv; n≥10,000 | Regional mean MAX differences span negative and positive values | Retain full-range display; block spacing is not genomic distance |
| S11 | stratum_gene.csv; n≥1,000 | Most retained genes have lower mean OpenSpliceAI MAX | Redraw descriptive title and MAX axes; explain algebraic coupling and label selection |

The figure manifest identifies all plotting inputs. Display tables contain exact denominators and
unrounded values. The original sixteen-figure release remains available in its original bundle.

## Claim review by section

Every main-report and supplement paragraph was reviewed against the following evidence boundaries.
Template-derived quantities have a machine-readable `claim-sources.csv` ledger; arithmetic and
cross-table checks are recorded in `numerical-checks.csv`. Displayed boundary ratios and relative call-count changes are generated directly from counters. Other rounded examples below are independently recoverable from the named tables. The companion report retains its distinct
architecture/benchmark purpose and the primary versus matched domain distinction.

| Text or claim | Evidence | Required qualification |
|---|---|---|
| Abstract and gain–loss conclusions | Exact correlations, threshold counts and matched MAEs | These are predicted effects and agreement, not comparative biological accuracy |
| Study design, model labels and provenance | Campaign settings, audit manifests, coverage and matched-domain receipts | Single rs10/rs13 checkpoints versus five-model ensemble; MANE training differs from scoring annotation |
| Score-distribution paragraphs | Joint histograms, exact moments, quantization table | Shared zeros affect global averages; quantiles are binned approximations |
| Call-count changes of +29.1%, +16.4%, −21.2%, −47.8% | Exact 0.5 call ratios minus one for AL/DL/AG/DG | Threshold-specific changes; do not imply sensitivity or specificity |
| Near-boundary ratios 0.973/0.984; 11–50-bp ratios 2.44/5.10 | Pooled AL/DL counts in site_event.csv | Same denominator; distance uses pooled internal boundaries |
| Far-gain ratios 0.797/0.439 | Pooled >500-bp AG/DG calls at 0.5 | Not every threshold, variant or transcript-specific context |
| Position paragraphs and exceptions | dp_agreement.csv and site_dp.csv; targeted source investigation | Masking is a recorded rule, not proof of identical behavior in legacy files |
| Gene-level fractions and medians | Positive defined gene ratios after n≥1,000 filtering | Equal gene weights; excluded ratios are not infinite effects |
| 19,063 genes; 18,881 negative means; median −0.00302 | stratum_gene.csv with n≥1,000 | MAX summary; mathematical coupling; no disease or enrichment claim |
| 2,571 blocks and range −0.02608 to +0.00182 | stratum_block_1mb.csv with n≥10,000 | Display eligibility and composition; not a regional mechanism |
| eLife comparison and writing | https://elifesciences.org/articles/107454, version of record October 30, 2025 | Selected-site mutagenesis and ensemble results differ from this masked variant-score experiment |
| Verification and data availability | Numerical check receipts, build gates, manifest and checksums | Tested real-data windows and aggregate consistency do not validate every prediction experimentally |

## Interpretation and writing standard

Each results paragraph states the comparison, quantitative observation, and its supported
interpretation. Repeated editorial history was moved out of the scientific narrative. The eLife
paper's direct question–comparison–result sequence guides the prose; its distinct experimental
results are neither transplanted nor treated as equivalent metrics. AG/AL/DG/DL are nonexclusive
predicted events; MAX is a summary. rs10 and rs13 are seeds rather than folds.

## Reproducibility and remaining limits

The release uses an explicit immutable ID, separate from the scientific and progress dates.
Previously published download identities cannot be reused. Tests cover rare mismatch visibility,
undefined distance groups, count pooling and download preservation. Numerical checks independently
recalculate fractions and ratios from exported counters, while baseline real-data-window checks
remain available in the September 9 bundle. These checks establish internal numerical consistency
within their scope; no experimental truth labels are available to rank biological accuracy.
