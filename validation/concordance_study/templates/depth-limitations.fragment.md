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
