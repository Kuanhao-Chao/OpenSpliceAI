# OpenSpliceAI full-SNV concordance study protocol

## Scope and estimands

This study has three related genome-wide estimands. They must not be blended
into one headline number.

1. **Masked genome-wide predictor concordance (`M=1`)** compares the published
   two-decimal SpliceAI ensemble annotations already present in the 100,000
   source chunks with one OpenSpliceAI checkpoint at five-decimal precision.
   The primary unit is an exact `(CHROM, POS, REF, ALT, gene)` match. A
   per-variant maximum-across-genes view is a sensitivity analysis only.
2. **Masked seed reproducibility (`M=1`)** compares rs10 with rs13 on the exact
   intersection of audit-approved chunks and exact genes. It measures training
   seed sensitivity, not accuracy. Until both seeds have all 100,000 chunks,
   this analysis is explicitly provisional.
3. **Seed-versus-model discrepancy (`M=1`)** compares the rs10--rs13
   difference with each seed's SpliceAI difference using identical chunks,
   genes, events, thresholds, and score definitions. This separates
   training-seed variability from the broader SpliceAI/OpenSpliceAI system
   contrast without using an external truth set.

The genome-wide SpliceAI comparator is its published v1.3 five-model ensemble,
whereas rs10 and rs13 are individual 10,000-nt MANE-trained OpenSpliceAI
checkpoints. The OpenSpliceAI campaign uses the repository's GENCODE v24 GRCh38
canonical scoring annotation (`data/grch38_chr.txt`). Consequently, the primary
SpliceAI--OpenSpliceAI contrast is a method/system comparison, not an isolated
architecture effect: ensemble averaging, training realization and corpus,
implementation, annotation coverage, and output quantization can all
contribute. The rs10--rs13 contrast more narrowly measures random-training-seed
reproducibility under the shared scoring setup.

SpliceAI is a comparator, not ground truth. This study does not estimate
biological accuracy; it characterizes numerical, event, annotation, and
regional concordance between score collections.

## Prespecified questions

The analysis is designed to answer the following questions.

1. How complete and provenance-bound is each score collection? How many
   variant groups, annotations, exact gene pairs, missing genes, conflicts,
   malformed annotations, duplicate annotations, and incomplete boundary
   groups are present?
2. Do continuous scores agree in location and scale? Report bias, MAE, RMSE,
   Pearson correlation, Lin's concordance correlation coefficient, binned rank
   correlation, and practical-equivalence rates for AG, AL, DG, DL, and MAX.
3. Is apparent disagreement explained by output precision? Repeat the primary
   comparison after rounding OpenSpliceAI to two decimals and report the change
   in error and exact-match rates.
4. Do the predictors make the same calls at 0.1, 0.2, 0.5, and 0.8? Report the
   complete 2-by-2 call table, positive/negative agreement, Jaccard/Dice,
   kappa, MCC, and the ratio of positive call rates. Overall agreement alone is
   not sufficient because most genome-wide scores are zero or small.
5. Are disagreements concentrated in nonzero or decision-relevant variants?
   Repeat continuous summaries on the union-signal subsets `either > 0` and
   `either >= threshold`.
6. Do event identity and predicted position agree? Report dominant-event
   confusion including ties/none, and score-gated DP agreement within
   0/1/2/5/10 bases for each event.
7. Are discrepancies heterogeneous by chromosome, exact gene, 1-Mb block,
   REF>ALT substitution, dominant-event pair, or event category? These are
   descriptive strata; REF>ALT is not trinucleotide context.
8. How much variability is attributable to a random training seed? Apply the
   same continuous, threshold, event, DP, and stratum analysis to rs10 versus
   rs13. Compare seed disagreement with SpliceAI--OpenSpliceAI disagreement,
   without treating the two comparisons as independent samples.
9. Are rs10--rs13 differences small relative to the SpliceAI/OpenSpliceAI
   difference, and are those relationships stable across score thresholds,
   event types, genes, chromosomes, substitutions, and genomic blocks?
10. Which observed differences are likely attributable to two-decimal
    SpliceAI quantization, annotation/gene assignment, event identity, or
    model/system behavior? The report must label these as descriptive
    explanations, not causal proofs.

## Evidence required for conclusions

“Highly correlated” is supported only by a high correlation estimate and says
nothing by itself about calibration or interchangeability. “Practically
equivalent” additionally requires small bias/error, high Lin CCC, high
predeclared equivalence rates, stable threshold calls, and no important
gene/event/region failure mode. “Same biological accuracy” cannot be concluded
from concordance, and no external database or functional-validation analysis
is part of this study.

Threshold comparisons always show raw counts and denominators. Exact-gene
metrics are primary; per-variant collapsed metrics are secondary because they
can pair signal assigned to different genes. Seed comparisons must use exact
shared chunks and genes, not different coverage sets.

The full-genome report can be labelled **final** only when the frozen pair list
contains exactly chunk IDs 1 through 100,000 and every raw input/output hash
matches a current deep-audit manifest. Partial chunk intersections are useful
but must remain **provisional**. A legacy VCF without an embedded receipt can be
structurally audited and supported by historical logs, but its original model
identity is not cryptographically provable; reports must retain that
limitation. A structurally valid VCF carrying a malformed or mismatched receipt
is classified separately as `receipt_invalid`; it is never promoted to
receipt-bound provenance.

## Uncertainty and dependence

Genome-wide sample size is so large that naive row-level p-values would be
misleading. The primary presentation is effect size plus denominators.
Deterministic gene- and 1-Mb-block cluster bootstraps provide sensitivity
intervals for additive MAX-score metrics, but do not remove every dependency
created by overlapping genes, nearby variants, or shared training data.
All uncertainty summaries are descriptive cluster-bootstrap sensitivities over
genes or genomic blocks; they do not represent independent biological samples.

## Reproducibility contract

Every released result records immutable pair-list and code hashes, exact model,
annotation, reference and masking identities, deterministic runtime settings,
mapper shard hashes, and the deep-audit manifest hashes. Raw VCF hashes are
checked before mapping. Missing tasks, duplicate chunks, changed files, mixed
labels/configurations, or incomplete final chunk sets fail closed. Figures are
derived from reduced aggregate JSON rather than from an untracked notebook.
