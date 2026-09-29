# Full-SNV file format and concordance analysis specification

## Files and units

The source directory contains 100,000 headered GRCh38 VCF chunks named
`snv.hg38_<chunk>.vcf`; the OpenSpliceAI directories contain corresponding
`snv.hg38_<chunk>_openspliceai.vcf` files. A source chunk is the atomic scoring
and audit unit. The deep audit records the exact total row count rather than
trusting a filename or directory count; as a concrete format check, source
chunk 50 contains 34,334 records in the current tree.

Each record has the eight VCF columns `CHROM`, `POS`, `ID`, `REF`, `ALT`,
`QUAL`, `FILTER`, and `INFO`. The source INFO field is:

```text
SpliceAI=ALT|GENE|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
```

`DS_*` are the four delta scores and `DP_*` are the four predicted positions.
The provided source scores are masked (`M=1`) and are displayed at two-decimal
precision. OpenSpliceAI preserves the seven fixed VCF fields and the original
SpliceAI value, then adds:

```text
OpenSpliceAI=ALT|GENE|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
```

OpenSpliceAI uses five-decimal output, `D=50`, a 10,000-nt flank, and the same
masked scoring intent. Unsupported annotation contigs must preserve the source
record without inventing a prediction. The prediction annotation is not required
to have the same gene symbol as the source row: the source can contain several
SpliceAI rows for overlapping genes, while the MANE-trained OpenSpliceAI output
can emit the selected MANE gene, or no `OpenSpliceAI` value for a row whose
source gene is outside that scoring annotation. In other words, a preserved VCF
row is a structural identity check, not a claim that both models scored the same
gene. The analysis therefore reports exact-gene coverage and a separate
annotation-agnostic per-allele sensitivity view rather than silently pairing
different genes.

The current chunk-50 spot check is useful for interpreting the eventual full
audit: both rs10 and rs13 preserve all 34,334 source rows and their source
fields, carry the two expected INFO declarations, and use DP values in
`[-50,50]`. The source rows include `CDK11A`, `CDK11B`, and `MMP23B`; the
OpenSpliceAI spot check emits the MANE `CDK11A` annotation where available and
leaves 78 rows without an `OpenSpliceAI` value because those rows have no
selected-MANE prediction. This is a sample-level observation, not a
genome-wide estimate; the audit and concordance coverage tables are the
authoritative counts.

### Current preflight snapshot (not a final audit)

The read-only inventory on 2026-08-08 found 100,000 source filenames, 99,307
rs10 output filenames, and 45,618 rs13 output filenames. These are filename
counts only. The last rs10 deep-audit sidecar (created 2026-08-01 with an old
campaign digest) recorded 99,283 structurally valid chunks, 693 missing outputs,
13 empty outputs, and 11 truncated outputs; all of its rows predate the current
campaign and are marked legacy/unprovenanced. There is no current rs13 manifest.
Those observations are retained for diagnosis, but they are deliberately not
used to build mapper inputs or submit a resume array: a fresh deep audit is
required first.

## Analysis identity

The primary analysis unit is the exact tuple
`(CHROM, POS, REF, ALT, gene)`. VCF rows are not assumed to be unique at that
level: an allele may occur once per overlapping gene, and a prediction may
repeat a complete list or may be absent for a source gene that is not in the
OpenSpliceAI scoring annotation. Identical repeated annotations are
deduplicated. Conflicting values for one exact key are counted and excluded
from paired numerical metrics; missing/gene-mismatched annotations remain in
the coverage denominators and are never treated as zero scores.

The three prespecified comparisons are:

1. SpliceAI versus OpenSpliceAI rs10.
2. SpliceAI versus OpenSpliceAI rs13, provisional until rs13 is complete.
3. OpenSpliceAI rs10 versus rs13, measuring seed reproducibility rather than
   biological accuracy.

SpliceAI is the comparator, not truth. Differences can reflect ensemble versus
single-checkpoint behavior, training realization, implementation, annotation,
masking, and two-decimal quantization. The GENCODE v24 GRCh38 scoring
annotation and exact model/reference hashes are retained in every result.

## Statistics and figures

For AG, AL, DG, DL, and MAX, the reducer reports exact additive agreement
statistics, two-decimal rounding sensitivity, union-signal subsets, threshold
tables at 0.1/0.2/0.5/0.8, dominant events, DP tolerances, and descriptive
chromosome/gene/block/substitution strata. Histogram-derived KS, Wasserstein,
Jensen–Shannon, and rank statistics are explicitly marked approximate. Gene and
1-Mb-block cluster bootstrap intervals are sensitivity intervals, not claims of
independent biological sampling.

The report includes score distributions, joint-score heatmaps, difference plots,
threshold agreement, DP agreement, dominant-event agreement,
chromosome/gene/block summaries, and quantization sensitivity. Every report is
published as both Markdown and a self-contained HTML document plus machine-
readable JSON/TSV outputs.

## Interpretation boundaries

High correlation alone does not establish calibration or interchangeability.
Practical equivalence requires small bias/error, strong Lin concordance, stable
threshold calls, and no important event/gene/region failure mode. A provisional
partial overlap is useful for debugging and exploratory analysis but cannot be
labelled final. Concordance alone cannot establish biological accuracy, and no
external database or functional-validation analysis is performed here.
