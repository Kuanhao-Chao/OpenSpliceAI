.. _predict_subcommand:

predict
=======

Predict acceptor/donor probabilities from FASTA using one PyTorch checkpoint
or an ensemble directory. Both raw and calibrated artifacts are supported, with
matching ``--flanking-size``. Every ensemble member must load successfully.

Unannotated FASTA entries are scored in their supplied orientation and produce
local coordinates on the entry ID. For genomic coordinates, provide GFF3 with
``--annotation-file`` or headers such as ``gene NC_0123.1:101-300(-)``.
A minus-strand genomic-header sequence must already be reverse-complemented.
GFF extraction does this automatically and includes ``--gene-flank`` real bases
on each side (default half the model context). Header spans must match sequence
lengths. Short sequences use N padding; that padding is not emitted as genomic
sequence. Supply real context when assessing junctions near sequence boundaries.

Long entries are split with inference halos. Each original position has one
owned output interval, so halo overlaps create no duplicate rows. Different
overlapping gene records can still describe the same genomic site.
``OSAI_CORE``, ``OSAI_ORIGIN`` and ``OSAI_LENGTH`` in intermediate headers record
ownership, source name and true length.

Default output is BED6 in ``SpliceAI_5000_<context>/``: contig, zero-based start,
half-open end, name, probability, strand. Acceptor p uses [p-1,p), donor p uses
[p,p+1), transformed to genomic orientation. Scores must exceed the threshold.
Four storage/execution routes share coordinate handling: HDF5 or PT intermediates,
each with streamed BED writing or ``--predict-all`` stored probability arrays.

Example
-------

.. code-block:: bash

   openspliceai predict --input-sequence genome.fa --model model_80nt.pt -f 80 --output-dir predictions

Complete options
----------------

.. include:: ../_generated/cli-predict.inc

See :doc:`output_explanation` for schemas and :doc:`migration` for changed behavior.
