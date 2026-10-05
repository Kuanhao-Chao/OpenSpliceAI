.. _variant_subcommand:

variant
=======

Annotate a VCF using the matching genome FASTA, transcript annotation and
complete model checkpoint or ensemble. ``--model`` is required. Built-in
``grch37``/``grch38`` tables are packaged; use an assembly-matched custom table
for other genomes. Custom TSV columns are ``#NAME, CHROM, STRAND, TX_START,
TX_END, EXON_START, EXON_END`` (tab-separated). Transcript and exon starts are
zero-based, ends are exclusive; exon lists are comma-separated. Strands are +/−,
exons are ordered and lie within their transcript. Empty annotation tables
produce pass-through records.

For each alternate allele and overlapping gene, reference and alternate windows
are scored within ``--distance`` (0–4999). AG/AL/DG/DL are maximum probability
changes for acceptor gain/loss and donor gain/loss. DP gives the maximizing offset
in the forward genomic frame for both strands. Ties use the first maximizing
position. ``--mask 1`` suppresses gains at annotated sites and losses away from
annotated sites. Missing/unsupported symbolic alleles, reference mismatches,
unavailable chromosome windows and REF spans longer than distance+1 are skipped;
the input record is retained. SNVs, insertions, deletions and supported MNV/delins
receive numeric scores. Original SpliceAI does not score the MNV/delins extension.

PyTorch ``--batch-size >1`` reuses reference windows and groups alternate windows
by width. FP32 is the default; ``OSAI_TF32=1`` and ``OSAI_CUDNN_BENCH=1`` opt into
other kernels and require workload-specific numerical validation.
``OSAI_DETERMINISTIC=1`` requests deterministic algorithms; configure
``CUBLAS_WORKSPACE_CONFIG`` before importing PyTorch for CUDA. Backend settings
are restored after library calls. Rounding affects DS only; DP is integer.

Filesystem output is validated then atomically published. ``.gz``/``.bgz`` paths
use BGZF. Failure preserves an existing output; stdin/stdout streams have no
atomic replacement. Diagnostics go to stderr. Reference contigs absent from
annotations pass through unchanged. For original Keras weights use an explicit
``.h5`` path/directory with ``-t keras -f 10000``; the repository-only ``SpliceAI``
preset also requires these flags and installed weights.

Example
-------

.. code-block:: bash

   openspliceai variant -R genome.fa -A annotation.tsv -I variants.vcf -O scored.vcf.gz --model model_80nt.pt --model-type pytorch -f 80 --batch-size 8 --precision 6

Complete options
----------------

.. include:: ../_generated/cli-variant.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
