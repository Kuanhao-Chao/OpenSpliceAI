.. _Q&A:

Troubleshooting and edge cases
==============================

Empty data or no evaluation batches
-----------------------------------

Check paired X/Y shards, gene biotype attributes, assembly names and chromosome
splits. Very small input sets can leave validation empty. Use explicit validation
files; lower the batch size or disable training ``--drop-last`` for tiny shards.
Human chromosome splitting expects UCSC ``chr`` names. Metadata keys are ignored.

Checkpoint does not load
------------------------

Use complete raw state dictionaries or a versioned calibrated artifact at the
matching context. Directories are ensembles and every member must be valid.
Do not mix context sizes or include temperature-only files in an ensemble folder.
Partial student transfer initialization requires explicit opt-in and is unsuitable
for ordinary inference. Original Keras weights require their separate backend.

Few predicted sites near a gene boundary
----------------------------------------

Provide real genomic context. A FASTA consisting only of a short exon/gene must
use N padding, which changes its predictions. GFF extraction includes real flanks
by default. Without annotation, input orientation is used directly; a minus-strand
header describes an already reverse-complemented sequence.

Unscored variants
-----------------

Check the reference allele, contig aliases, transcript overlaps, complete reference
window and allele type. Symbolic/missing alleles are skipped, and REF length must
not exceed distance+1. Records retained without scores are coverage-only inputs.

Memory, disk and reproducibility
--------------------------------

Use streamed prediction for long sequences and calibrate with sufficient temporary
disk space. Training currently accumulates selected-split predictions for exact
ranking metrics; size datasets to available RAM. CUDA may differ numerically from
CPU even with the same seed. Benchmark representative windows on the actual
allocation and record TRES billing, elapsed time, source/environment/input hashes.
Validate optional TF32 settings before using them in a scientific campaign.
