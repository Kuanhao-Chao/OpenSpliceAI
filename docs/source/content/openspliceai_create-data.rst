.. _create-data_subcommand:

create-data
===========

Convert a genome FASTA and matching GFF3 annotation to train, validation and
test HDF5 files. Sequence names and assembly coordinates must agree. The
annotation needs genes, transcript children and exons, with ``gene_biotype``
(or ``gene_type``) identifying the selected biotype.

``canonical`` chooses the transcript with the longest genomic span per gene;
``all_isoforms`` creates independent transcript labels and keeps repeated gene
records together in the validation split. Protein-coding, non-coding and combined
``all`` inputs are supported. ``--canonical-only`` retains GT–AG, GC–AG and AT–AC
splice pairs. Ambiguous bases are encoded as zero vectors.

Random chromosome splitting uses ``--split-ratio``; the human split uses UCSC
names and holds out chr1, chr3, chr5, chr7 and chr9. ``--chr-split test`` produces
only test files. A validation fraction of the training genes is selected during
data creation, not during model training. Small inputs can yield an empty split;
training and calibration reject that split explicitly.

Datasets retain the legacy schema: ``Xn=(N,15000,4)`` and
``Yn=(1,N,5000,3)``. Stored context is 10000 bases for all four model contexts;
the training loader crops inputs to the chosen model and retains output labels.
Non-coding files use ``_ncRNA`` filenames. ``--remove-paralogs`` uses minimap2
identity and query-coverage thresholds to filter homologous test/validation genes.
This is a sequence-similarity filter, not a complete proof of biological independence.

Example
-------

.. code-block:: bash

   openspliceai create-data --genome-fasta genome.fa --annotation-gff annotation.gff3 --output-dir data --verify-h5

Complete options
----------------

.. include:: ../_generated/cli-create-data.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
