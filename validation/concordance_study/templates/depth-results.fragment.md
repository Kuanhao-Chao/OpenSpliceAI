### 5.1 Coverage and the comparison domain

{{table:runs_overview}}

The primary arm uses {{primary.chunks|,d}} audited chunks and
{{primary.coverage.paired_annotations|,d}} exact-gene paired annotations. Source rows,
alleles and gene annotations are different counting units; the coverage diagram records
that distinction. Missing predictions remain missing, and conflicting annotations are
excluded rather than replaced by zero.

<figure><img src="figures/f01_coverage.png" alt="Processing flow distinguishing source rows, variant groups and exact-gene paired annotations"></figure>

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

<figure><img src="figures/f02_joint_density.png" alt="Paired score distributions and conditional medians for all four events and MAX"></figure>

{{table:equivalence}}

Shared zeros dominate the full population. The following subsets condition on either
system carrying signal, making their changed denominator explicit. They are descriptive
subsets selected using the scores themselves, not independent validation sets.

{{table:signal_subsets}}

The tail curves show the fraction of paired annotations at each magnitude of positive or
negative score difference. They summarize the direction as well as the size of disagreement.
These tails use differences rounded to histogram centres and are approximate.

<figure><img src="figures/f03_difference_distributions.png" alt="Negative and positive score-difference tail probabilities by event"></figure>

{{table:tail_asymmetry}}

### 5.3 Threshold agreement and operating-point transfer

Configured exact thresholds are {{primary.configured_thresholds}}. The joint histogram
adds {{primary.agreement_curve.points|d}} interior cutoff points at
{{primary.histogram_binning.width|.3f}} spacing. A cutoff of one cannot be isolated from
the final histogram bin and is excluded from the transfer search.

{{table:thresholds_max}}

<figure><img src="figures/f04_threshold_agreement.png" alt="Jaccard, kappa, MCC and call-rate ratio across the score cutoff grid"></figure>

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

<figure><img src="figures/f05_operating_point_transfer.png" alt="Per-event OpenSpliceAI cutoffs chosen by call-rate matching and MCC optimization"></figure>

{{table:operating_points_by_event}}

### 5.4 Output precision

The comparator carries {{study.source_precision_dp|d}} decimal places and OpenSpliceAI
{{study.openspliceai_precision_dp|d}}. Rounding the latter to the comparator's precision
changes exact-value agreement. The residual MAE after subtracting a half-grid allowance
bounds the part that cannot be attributed to this quantization allowance alone; it does
not recover the unpublished full-precision comparator scores.

{{table:quantization}}

<figure><img src="figures/f11_quantization.png" alt="MAE and exact-match rates before and after matching output precision"></figure>

### 5.5 Event identity, selected discrepancies and collapsed annotations

{{table:dominant}}

<figure><img src="figures/f06_dominant_event.png" alt="Row-normalized dominant-event confusion with NONE and ties retained"></figure>

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

<figure><img src="figures/f07_dp_agreement.png" alt="Agreement of predicted event offsets among jointly called annotations"></figure>

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

<figure><img src="figures/f12_site_distance.png" alt="Event-specific call rates at 0.5 by distance from the variant to the nearest internal boundary"></figure>

{{table:site_events_depth}}

The score distributions compare variants at or within 2 bp of an internal boundary with
variants more than 500 bp away. Both groups retain their own denominator and include all
nearest-site types. Zero call rates are retained in the tables even when omitted on a log axis.

<figure><img src="figures/f13_site_score_distributions.png" alt="Per-event score survival curves near boundaries and more than 500 bp away"></figure>

<figure><img src="figures/f14_site_conditioned_calibration.png" alt="Conditional median and interquartile range of OpenSpliceAI scores by annotation proximity"></figure>

Conditional medians and interquartile ranges show score agreement within each comparator
score bin. Empty bins remain gaps. This is calibration against another predictor, not
against measured probabilities of altered splicing.

<figure><img src="figures/f15_site_conditioned_dp.png" alt="Exact and within-two-base predicted-position agreement by variant proximity"></figure>

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

<figure><img src="figures/f10_seed_versus_model.png" alt="Method disagreement compared with seed disagreement on the same variant and gene observations"></figure>

<figure><img src="figures/f16_seed_method_by_gene.png" alt="Distribution across genes of method-to-seed MAE ratios for each event"></figure>

The gene figure includes genes with at least 1,000 paired observations and positive, defined
ratios. The full exports retain zero and undefined cases. The following summary gives equal
weight to strata, which differs from the observation-weighted global MAE ratio. A ratio is
undefined when the seed MAE is zero. Dominant-pair categories are excluded from matched-stratum
ratios because their membership depends on the comparison itself.

{{table:seed_method_strata}}

<figure><img src="figures/f08_chromosome.png" alt="MAX score bias across genomic one-megabase blocks"></figure>

<figure><img src="figures/f09_gene_divergence.png" alt="Distribution of gene-level divergence with labelled outliers"></figure>

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
