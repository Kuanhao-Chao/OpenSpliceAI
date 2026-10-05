<header class="masthead" markdown="1">
<p class="eyebrow">OpenSpliceAI validation study</p>

# Genome-wide concordance of OpenSpliceAI and SpliceAI variant delta scores

<p class="dek">Score agreement, seed variation and annotated-site context on audited human SNV outputs.</p>
<p><span class="status">Provisional</span></p>

<dl class="meta">
<div><dt>Paired annotations</dt><dd>{{primary.coverage.paired_annotations|,d}}</dd></div>
<div><dt>Audited chunks</dt><dd>{{primary.chunks|,d}} of {{study.source_chunks|,d}}</dd></div>
<div><dt>Generated</dt><dd>{{generated_at}}</dd></div>
</dl>
</header>

{{study.campaign_status}} This continuation completes CPU analysis of the frozen outputs;
it does not complete the GPU scoring campaign.

[TOC]

## Abstract

We compare {{primary.coverage.paired_annotations|,d}} exact-gene paired annotations from
{{primary.chunks|,d}} audited chunks. The recorded scoring setup matches masking, gene
annotation, distance and reference. MAX Pearson correlation is
{{primary.agreement.MAX.pearson_r|.4f}}, Lin concordance is
{{primary.agreement.MAX.lin_ccc|.4f}}, and positive-call Jaccard at 0.5 is
{{primary.thresholds.MAX[0.5].jaccard|.4f}}. Loss and gain channels have different agreement
profiles. A joint rs10/rs13/reference pass measures seed and method differences on exactly
{{arms.B_seeds_rs10_rs13.coverage.paired_annotations|,d}} common variant/gene observations.
Event-level statistics, threshold transfer, genomic strata, selected extreme discrepancies
and internal-site context describe the disagreement. None of these analyses uses experimental
truth labels or ranks biological accuracy. All source outputs remain legacy-unprovenanced
and all results remain provisional.

## 1. Introduction

SpliceAI and OpenSpliceAI predict acceptor gain, acceptor loss, donor gain and donor loss
scores with the same interpretation and range. Numerical agreement on those scores is a
separate question from splice-site prediction accuracy on held-out sequence. It matters
when comparing workflows, transferring a cutoff, or prioritizing discrepancies for follow-up.

An earlier comparison used {{study.prior_analysis_n|,d}} variants with different masking
settings. The intended common scoring setup here removes that known confound. It does not
isolate implementation from training, ensembling or output precision. A second independently
seeded OpenSpliceAI checkpoint provides a reference scale for training-run variation.

## 2. Scoring inputs

The source is {{study.source_collection}}, partitioned into {{study.source_chunks|,d}}
headered VCF files. The source annotation format is
`ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`. OpenSpliceAI preserves
source records and adds an annotation in the same format.

The published collection contains genic SNVs across the human genome; it does not
include every genomic base or every possible GRCh38 SNV. Genome-wide describes
the collection's genomic extent, and coverage percentages refer to this collection.

| Property | Campaign record |
|:---|:---|
| Reference | {{study.reference_assembly}} |
| Scoring annotation | {{study.scoring_annotation}} |
| Masking | {{study.masking}} |
| Maximum event distance | {{study.distance|d}} nt |
| Flanking context | {{study.flanking_size|,d}} nt |
| Comparator | {{study.spliceai_model}} |
| OpenSpliceAI models | {{study.openspliceai_model}} |
| Decimal precision | comparator {{study.source_precision_dp|d}}; OpenSpliceAI {{study.openspliceai_precision_dp|d}} |
| Runtime settings | {{study.deterministic_runtime}} |

These settings are supported by campaign records. Because the analyzed files predate embedded
model receipts, their checkpoint identity is not cryptographically established by the file audit.

## 3. Analysis methods

The primary unit is `(CHROM, POS, REF, ALT, gene)`. Identical repeated annotations are
collapsed; conflicting values within a method/gene are excluded. Missing annotations are
counted as unpaired rather than imputed as zero. A secondary allele-level view takes event
maxima across genes and explicitly changes the estimand.

Primary arm A compares SpliceAI and rs10 on the full frozen rs10 pair list. The additional
matched pass joins reference, rs10 and rs13, then selects valid nonconflicting genes present
on all three sides. It sends this identical membership to B (rs10 versus rs13), C (reference
versus rs10) and D (reference versus rs13). This excludes the small denominator differences
in the archived initial pairwise C/D analyses.

Streaming maps retain additive moments, configured-threshold contingency tables, fixed-width
histograms and bounded deterministic samples. Boundary groups are deferred until reduction,
where adjacent selected chunks are joined and incomplete groups are excluded. Each worker
verifies an immutable manifest binding its code, pair list and annotation. Reduction rejects
missing or duplicate tasks/chunks, mismatched configurations, changed inputs or annotation
digests, and unexpected provenance. An incomplete chunk domain cannot be labelled final.

The depth pass adds event-level moments and call tables within chromosome, gene, one-megabase
block, substitution, dominant-event pair and site-distance strata. Site-conditioned joint
histograms and DP counters support the added distribution and positional panels. The
configuration uses {{primary.histogram_binning.bins|d}} score bins and exact thresholds
{{primary.configured_thresholds}}. The bounded top-discrepancy set is selected by magnitude,
while the hash sample is uniform over paired observations.

Metrics include bias (right minus left), MAE, RMSE, Pearson correlation, Lin concordance,
exact/tolerance agreement, positive-call Jaccard, kappa, MCC, call-rate ratios and predicted
position agreement. Spearman and distribution distances are histogram approximations.
Configured-threshold counts are exact integers; moment sums are floating-point calculations.
Gene and block bootstraps describe sensitivity to resampling those clusters, without assuming
that nearby variants or overlapping genes are biologically independent.

Internal splice boundaries are derived from the scoring gene table, with starts+1, ends
unchanged and donor/acceptor roles assigned by strand. The index excludes transcript ends,
pools across genes on each contig, and retains ambiguous roles. This differs from the scorer's
per-gene mask; annotation proximity is context rather than a truth label.

## 4. Audit and independent verification

{{table:provenance}}

An independent standard-library implementation rereads the VCFs with its own parser,
grouping, conflict handling, three-way intersection, site derivation and arithmetic.
The primary window contains {{study.verification.primary.chunks|d}} contiguous chunks and
{{study.verification.primary.paired_annotations|,d}} paired observations; the matched window
contains {{study.verification.matched.chunks|d}} chunks and
{{study.verification.matched.paired_annotations|,d}} common observations.

The primary check is **{{study.verification.primary.status}}** across
{{study.verification.primary.checks|,d}} comparisons; the matched check is
**{{study.verification.matched.status}}** across {{study.verification.matched.checks|,d}}.
Integer counts, call tables, joint and marginal histograms, and DP counts must agree exactly.
The check also recomputes event statistics in every requested stratum. Continuous statistics
are compared with relative tolerance {{study.verification.relative_tolerance}} and absolute
tolerance {{study.verification.absolute_tolerance}}. The maximum observed absolute errors are
{{study.verification.primary.max_error}} and {{study.verification.matched.max_error}},
respectively. These are implementation checks on real windows, not independent biological
validation or an exhaustive reread of the full production domain.

The verification artifacts bind the requested inputs, annotation, comparison kind, verifier
configuration and compared summary digest. Cached independent data are accepted only if their
identity matches the requested run; absent expected comparisons cannot count as a passing check.

## 5. Results

### 5.1 Coverage and the comparison domain

{{table:runs_overview}}

The primary arm uses {{primary.chunks|,d}} audited chunks and
{{primary.coverage.paired_annotations|,d}} exact-gene paired annotations. Source rows,
alleles and gene annotations are different counting units; the coverage diagram records
that distinction. Missing predictions remain missing, and conflicting annotations are
excluded rather than replaced by zero.

<figure><img src="figures/f01_coverage.png" alt="Processing flow distinguishing source rows, variant groups and exact-gene paired annotations"><figcaption>Figure 1. Coverage flow with the unit labelled at every stage. Multiple genes can annotate one variant group, so annotation counts are not a decreasing sequence of row counts. Arrows describe processing; their widths do not encode quantities.</figcaption></figure>

{{table:coverage}}

The seed comparison and both restricted method comparisons use the **same three-way
variant/gene intersection**, containing {{arms.B_seeds_rs10_rs13.coverage.paired_annotations|,d}}
observations. The intersection is taken after duplicate resolution and chunk-boundary joining.
This is stricter than the earlier September 8 analysis, which matched chunk lists but paired
genes separately in each comparison. Its slightly different pairwise denominators are
preserved in the archived baseline and are not mixed into these seed-to-method ratios.
The restricted region is {{study.rs13_genomic_extent}}; it is not a random genomic sample.

### 5.2 Score agreement and the gain–loss distinction

{{table:agreement}}

MAX Pearson correlation is {{primary.agreement.MAX.pearson_r|.4f}} and Lin concordance is
{{primary.agreement.MAX.lin_ccc|.4f}}. Loss-event correlations are
{{primary.agreement.AL.pearson_r|.4f}} for AL and {{primary.agreement.DL.pearson_r|.4f}} for DL;
gain-event correlations are {{primary.agreement.AG.pearson_r|.4f}} for AG and
{{primary.agreement.DG.pearson_r|.4f}} for DG. The signed bias is defined throughout as
OpenSpliceAI minus SpliceAI. Correlation measures covariation; it does not establish equal
values, equal calls, or biological correctness.

<figure><img src="figures/f02_joint_density.png" alt="Paired score distributions and conditional medians for all four events and MAX"><figcaption>Figure 2. Calibration: the distribution of the OpenSpliceAI score conditional on the SpliceAI score. Orange dots mark the conditional median; vertical intervals are the central 50% and 90%. A perfectly agreeing pair would track the dashed line. Empty comparator bins remain gaps; the two-decimal score grid naturally leaves some bins unoccupied.</figcaption></figure>

{{table:equivalence}}

Shared zeros dominate the full population. The following subsets condition on either
system carrying signal, making their changed denominator explicit. They are descriptive
subsets selected using the scores themselves, not independent validation sets.

{{table:signal_subsets}}

The tail curves show the fraction of paired annotations at each magnitude of positive or
negative score difference. They summarize the direction as well as the size of disagreement.
These tails use differences rounded to histogram centres and are approximate.

<figure><img src="figures/f03_difference_distributions.png" alt="Negative and positive score-difference tail probabilities by event"><figcaption>Figure 3. Tail asymmetry of the signed score difference. Left: the probability that each predictor exceeds the other by at least x. Right: their ratio, where 1.0 would mean the two disagree in both directions equally often. Difference magnitudes are rounded to histogram centres, so these tail estimates are approximate.</figcaption></figure>

{{table:tail_asymmetry}}

### 5.3 Threshold agreement and operating-point transfer

Configured exact thresholds are {{primary.configured_thresholds}}. The joint histogram
adds {{primary.agreement_curve.points|d}} interior cutoff points at
{{primary.histogram_binning.width|.3f}} spacing. A cutoff of one cannot be isolated from
the final histogram bin and is excluded from the transfer search.

{{table:thresholds_max}}

<figure><img src="figures/f04_threshold_agreement.png" alt="Jaccard, kappa, MCC and call-rate ratio across the score cutoff grid"><figcaption>Figure 4. Chance-corrected and positive-class agreement, and the ratio of call volumes, as continuous functions of a cutoff applied to both predictors. The dashed line on the right marks equal call volume. Values below one mean fewer OpenSpliceAI calls, without implying greater specificity.</figcaption></figure>

At a MAX cutoff of 0.5, positive-call Jaccard is
{{primary.thresholds.MAX[0.5].jaccard|.4f}} and MCC is
{{primary.thresholds.MAX[0.5].mcc|.4f}}. Overall agreement is
{{primary.thresholds.MAX[0.5].overall_agreement|pct3}} and is dominated by shared negatives.
For this reason the report emphasizes positive-call overlap and chance-corrected agreement.

The event-specific snapshot below uses the same 0.5 cutoff:

{{table:thresholds_by_event}}

Rate matching chooses the OpenSpliceAI cutoff whose number of calls is closest to SpliceAI's;
MCC optimization chooses the cutoff with greatest agreement with SpliceAI. Neither procedure
optimizes agreement with a biological outcome. A matched number of calls need not identify
the same variants. These cutoffs were selected and evaluated on the same population and are
descriptive, without a held-out transfer guarantee.

<figure><img src="figures/f05_operating_point_transfer.png" alt="Per-event OpenSpliceAI cutoffs chosen by call-rate matching and MCC optimization"><figcaption>Figure 5. Per-event operating-point transfer at every configured threshold. Matching call volume and maximising MCC answer different questions; neither optimises biological accuracy. Candidate cutoffs use the histogram grid and exclude the unresolved endpoint at one.</figcaption></figure>

{{table:operating_points_by_event}}

### 5.4 Output precision

The comparator carries {{study.source_precision_dp|d}} decimal places and OpenSpliceAI
{{study.openspliceai_precision_dp|d}}. Rounding the latter to the comparator's precision
changes exact-value agreement. The residual MAE after subtracting a half-grid allowance
bounds the part that cannot be attributed to this quantization allowance alone; it does
not recover the unpublished full-precision comparator scores.

{{table:quantization}}

<figure><img src="figures/f11_quantization.png" alt="MAE and exact-match rates before and after matching output precision"><figcaption>Figure 6. SpliceAI publishes two decimals. Left: how much exact agreement is recovered by rounding OpenSpliceAI to the same grid. Right: the fraction of mean absolute difference that cannot be explained by that grid.</figcaption></figure>

### 5.5 Event identity, selected discrepancies and collapsed annotations

{{table:dominant}}

<figure><img src="figures/f06_dominant_event.png" alt="Row-normalized dominant-event confusion with NONE and ties retained"><figcaption>Figure 7. Which event each predictor considers dominant, normalised within SpliceAI's category. TIE and NONE are explicit categories, not discarded.</figcaption></figure>

Dominant-event categories describe the largest score within each predictor. The next table
conditions MAX statistics on the pair of dominant categories; this is a score-selected
stratification and is not an independent test of the event labels.

{{table:dominant_pair_strata}}

The taxonomy below describes the selected
{{primary.discordance_depth.records|,d}} largest absolute discrepancies, grouped by the
event driving the difference, its direction and variant proximity. Equal largest differences
are labelled TIE. These are extreme records selected from the population; their category
frequencies are **not genome-wide prevalence estimates**. The complete record and gene
tables are distributed with the study.

{{table:discordance_depth}}

As an annotation sensitivity check, the primary comparison also takes each predictor's
maximum per event across its own genes, then pairs on the variant allele. This changes both
the unit and the gene assignment: signal assigned to different genes may now be paired.
It should be interpreted alongside exact-gene matching rather than replacing it. In the
matched B/C/D pass, collapsing occurs within the common gene set so those arms retain their
shared domain.

{{table:collapsed_view}}

### 5.6 Predicted-position agreement

For each event, positional agreement is conditioned on both scores meeting the stated
threshold. A zero offset is a valid predicted position. The table below uses 0.5; companion
tables contain every configured cutoff and tolerance. An empty eligible group is n/a.

{{table:dp_agreement}}

<figure><img src="figures/f07_dp_agreement.png" alt="Agreement of predicted event offsets among jointly called annotations"><figcaption>Figure 8. Predicted-position agreement among variants both predictors call at a given score threshold. Every configured cutoff is shown. Denominators are jointly-called pairs; undefined groups are gaps and a measured zero remains zero.</figcaption></figure>

This measures agreement about the selected position among jointly called annotations.
It does not cover calls unique to one predictor, nor establish that the shared position
is used in vivo.

### 5.7 Context around annotated internal splice boundaries

The index contains {{study.sites_count|,d}} unique internal boundary coordinates derived
from the exact scoring gene table. Exon starts are converted from zero-based to one-based
coordinates; exon ends remain unchanged. Donor and acceptor roles follow strand. Transcript
ends are excluded, and coordinates with conflicting roles or equidistant nearest sites of
different type are labelled ambiguous.

The index pools boundaries across genes on a contig. The scorer's mask instead uses a
per-gene boundary set that includes transcript ends. Consequently proximity in this figure
is descriptive context and is **not identical to the masking rule**. The plotted distance is
from the variant position, whereas the mask applies to the position selected by each event's
DP offset. A symmetric 1–2 bp bin is not an essential-dinucleotide annotation; more than
500 bp from these boundaries is not an intron annotation.

<figure><img src="figures/f12_site_distance.png" alt="Event-specific call rates at 0.5 by distance from the variant to the nearest internal boundary"><figcaption>Figure 9. Event-specific call rates at threshold 0.5, pooled over nearest-site types. Zeros are omitted from the logarithmic axis and retained in the accompanying table. Distance is annotation context, not an experimental truth label.</figcaption></figure>

{{table:site_events_depth}}

The score distributions compare variants at or within 2 bp of an internal boundary with
variants more than 500 bp away. Both groups retain their own denominator and include all
nearest-site types. Zero call rates are retained in the tables even when omitted on a log axis.

<figure><img src="figures/f13_site_score_distributions.png" alt="Per-event score survival curves near boundaries and more than 500 bp away"><figcaption>Figure 10. Score survival distributions near internal splice boundaries versus variants more than 500 bp away, for each event. Curves use histogram edges at 0.005 resolution. The distant group includes any genomic context present in the input; it is not labelled deep intron.</figcaption></figure>

<figure><img src="figures/f14_site_conditioned_calibration.png" alt="Conditional median and interquartile range of OpenSpliceAI scores by annotation proximity"><figcaption>Figure 11. Median and interquartile range of the right score conditional on the left score, separated by annotation proximity. Empty conditioning bins remain gaps. This is agreement calibration against a comparator, not calibration against observed splice outcomes.</figcaption></figure>

Conditional medians and interquartile ranges show score agreement within each comparator
score bin. Empty bins remain gaps. This is calibration against another predictor, not
against measured probabilities of altered splicing.

<figure><img src="figures/f15_site_conditioned_dp.png" alt="Exact and within-two-base predicted-position agreement by variant proximity"><figcaption>Figure 12. Predicted-position agreement by variant proximity. Both event scores must be at least 0.5; DP=0 is eligible. Empty eligible groups are gaps, and the table retains all five tolerances.</figcaption></figure>

Distance-conditioned DP agreement retains the requirement that both event scores are at
least 0.5. Proximity can locate disagreements for follow-up, but none of these distance
panels measures sensitivity, specificity or false-positive rate.

### 5.8 Seed variation and regional heterogeneity

{{table:seed_versus_model_events}}

On the common observations, the rs10 method-to-seed MAE ratios are
{{seed_versus_model.events.AG.ratios.C_rs10_matched.mae|.3f}} for AG,
{{seed_versus_model.events.AL.ratios.C_rs10_matched.mae|.3f}} for AL,
{{seed_versus_model.events.DG.ratios.C_rs10_matched.mae|.3f}} for DG and
{{seed_versus_model.events.DL.ratios.C_rs10_matched.mae|.3f}} for DL.
The seed contrast uses two checkpoints, so it measures this pair's variability and does not
estimate the full distribution across training runs. The method contrast also includes the
published ensemble-versus-single-checkpoint difference.

<figure><img src="figures/f10_seed_versus_model.png" alt="Method disagreement compared with seed disagreement on the same variant and gene observations"><figcaption>Figure 13. Left: mean absolute difference between two training seeds of OpenSpliceAI, and between SpliceAI and each seed, on identical three-way variant/gene observations. Right: rs10 method MAE divided by seed MAE, per event. A value near 1 indicates similar average score differences for these contrasts; it does not establish interchangeability.</figcaption></figure>

<figure><img src="figures/f16_seed_method_by_gene.png" alt="Distribution across genes of method-to-seed MAE ratios for each event"><figcaption>Figure 14. Distribution across genes of method-to-seed MAE ratios on the exact three-way intersection, restricted in the figure to genes with at least 1,000 paired observations and positive, defined ratios. The complete table retains excluded and undefined entries and all genomic strata.</figcaption></figure>

The gene figure includes genes with at least 1,000 paired observations and positive, defined
ratios. The full exports retain zero and undefined cases. The following summary gives equal
weight to strata, which differs from the observation-weighted global MAE ratio. A ratio is
undefined when the seed MAE is zero. Dominant-pair categories are excluded from matched-stratum
ratios because their membership depends on the comparison itself.

{{table:seed_method_strata}}

<figure><img src="figures/f08_chromosome.png" alt="MAX score bias across genomic one-megabase blocks"><figcaption>Figure 15. Mean signed MAX difference in each 1-Mb block with at least 10,000 pairs. Blocks are ordered and evenly spaced within each chromosome; gaps are not drawn to scale. Point area scales with the number of paired annotations in the block. The dashed line is the genome-wide mean; blocks below it are where OpenSpliceAI scores relatively lower still.</figcaption></figure>

<figure><img src="figures/f09_gene_divergence.png" alt="Distribution of gene-level divergence with labelled outliers"><figcaption>Figure 16. Per-gene divergence against how much splice signal the gene carries. Point area is the number of paired annotations; the labelled genes are the most negative. The histogram shows the same values as a distribution.</figcaption></figure>

The primary arm contains {{primary.landscape.blocks|,d}} one-megabase blocks containing at least 10,000 paired annotations.
The figure orders retained blocks within equal-width chromosome panels; genomic gaps are not
drawn to scale.
Chromosome, gene, block and substitution exports include MAX and all four events. A local
outlier is a prioritization lead, not evidence of a causal effect of gene length or density.
Comparing the restricted rs10 arm with the broad primary arm describes sensitivity to the
regional domain and the common-gene restriction:

{{table:generalization}}

These aggregate comparisons cannot guarantee that the seed pattern transfers to unscored
regions or to genes excluded by the three-way intersection.

## 6. Discussion

The comparison separates several questions that are easy to conflate: numerical equality,
call overlap, agreement on the dominant event, and agreement on the selected position.
Shared zeros explain much of the apparent all-annotation agreement. The event-specific
statistics and signal-conditioned summaries reveal the difference hidden by that denominator.

The gain–loss contrast persists under the intended common masking, annotation, distance and
reference settings. Its cause is not isolated by this experiment. The two score collections
also differ in training, checkpoint identity, implementation and ensembling, and the published
comparator is rounded. The two-seed contrast provides a useful scale for comparison without
separating these causes.

Annotated-site proximity adds a way to describe where disagreement occurs. It does not resolve
whether lower gain scores reflect fewer false positives or more missed true sites. Evaluating
that distinction requires an independently measured splicing outcome, with a sampling scheme
that includes both shared and predictor-specific calls.

Threshold transfer should be guided by the application and independently validated. The
rate-matched and MCC-optimal cutoffs here describe agreement with SpliceAI on this population;
they are not calibrated clinical decision thresholds. Likewise, whether an OpenSpliceAI ensemble
would reduce the observed method gap remains an experiment, not a result of this study.

## 7. Limitations

- **Concordance only.** No experimental or clinical truth labels enter the analysis. Neither
  global agreement nor annotation proximity ranks biological accuracy.
- **Provisional coverage.** The primary arm contains {{primary.chunks|,d}} of
  {{study.source_chunks|,d}} source chunks. Incomplete outer groups are excluded. The matched
  arms cover {{study.rs13_genomic_extent}}, with an additional common-gene restriction.
- **Legacy provenance.** The scored files are structurally audited and hash-pinned but remain
  `legacy_unprovenanced`. The audit does not cryptographically establish the checkpoint that
  created them. Scoring settings and model labels are campaign records, not embedded proof.
- **Ensemble and seed scope.** The published comparator is an ensemble; each OpenSpliceAI arm is
  one checkpoint. Only two OpenSpliceAI seeds are compared. Architecture, training and ensemble
  contributions are not separately identified.
- **Annotation context.** The pooled internal-site index excludes transcript ends and differs
  from the per-gene mask. It gives neither a transcript-specific intronic annotation nor an
  experimental label. Variant distance and predicted-event distance are distinct quantities.
- **Quantization and approximations.** Distribution summaries use
  {{primary.histogram_binning.bins|d}} bins of width {{primary.histogram_binning.width|.3f}}.
  Epsilon-corrected floor binning prevents binary-edge underflow; empty bins can naturally
  reflect the comparator's decimal grid. Spearman and distribution distances are binned
  approximations, difference tails are rounded to bin centres, and transferred cutoffs are
  restricted to grid edges below one. Direct configured-threshold counts are exact; moment
  arithmetic has floating-point summation error.
- **Selection and dependence.** Top discrepancies are a selected extreme sample, signal subsets
  are conditioned on the scores, and adjacent variants and overlapping genes are dependent.
  Gene/block bootstrap intervals are resampling sensitivity summaries, not guarantees over
  independent biological observations. Large sample size does not remove these limitations.
- **Gene coverage.** Exact matching excludes unshared or conflicting gene annotations. The
  collapsed view changes the unit and can join signals assigned to different genes. Its
  agreement cannot be interpreted as recovering the missing exact-gene observations.

## 8. Reproducibility

The measured results in both reports are resolved from the same `study_facts.json` and
reduced summaries. Unknown fact paths and unfinished interpretation markers fail the build.
Configured cutoffs, schema constants and historical engineering settings are stated separately
from measured results. All analysis is provisional.

| Artifact | SHA-256 |
|:---|:---|
| Primary frozen pair list | `{{study.pair_lists.rs10_all}}` |
| Matched three-way source pair list | `{{study.pair_lists.rs10_rs13_overlap}}` |
| rs10 audit manifest | `{{study.rs10_manifest_sha256}}` |
| rs13 audit manifest | `{{study.rs13_manifest_sha256}}` |
| Internal-site source annotation | `{{study.sites_digest}}` |
| Primary frozen execution manifest | `{{study.depth_primary_checks_sha256}}` |
| Matched frozen execution manifest | `{{study.depth_matched_checks_sha256}}` |

The archived initial pairwise analysis also used `rs10_on_overlap.tsv`
(`{{study.pair_lists.rs10_on_overlap}}`) and `rs13_on_overlap.tsv`
(`{{study.pair_lists.rs13_on_overlap}}`). The new C/D results are derived together with B
from the three-way pass, rather than from those separate pairwise reductions.

Production jobs: primary {{study.slurm_jobs.A_rs10_genomewide}};
matched B/C/D {{study.slurm_jobs.matched}}. The execution manifests pin the frozen code,
annotation and pair files checked by each worker. They do not supply the missing model
receipts for legacy score files.

The campaign scoring command was:

```text
{{study.scoring_command}}
```

Machine-readable facts, per-arm CSV tables, verification evidence and figures in PNG and PDF
accompany the standalone study. The self-contained HTML version embeds its figures and can
be read without access to the cluster filesystem.
