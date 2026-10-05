.. _repository_walkthrough:

Repository walkthrough
======================

Installed package
-----------------

``openspliceai/openspliceai.py`` constructs a light parser and lazily dispatches
six workflows. ``model_config.py`` owns the four architecture schedules.
``train_base/openspliceai.py`` implements residual/skip units, cropping, probability
outputs and frozen BatchNorm behavior. ``train_base/utils.py`` shares losses,
metrics, loaders, split resolution, schedulers and training/evaluation loops.

``create_data/`` parses annotations, extracts oriented sequences, labels junctions,
splits genes, removes homologs and builds legacy HDF5 shards. ``train/`` initializes
new models; ``transfer/`` adds freezing, rehearsal, teachers and regularization.
``data_schema.py`` validates paired shards. ``checkpoints.py`` provides atomic saves
and versioned calibrated artifacts. ``calibrate/streaming.py`` caches logits on
disk and computes complete-split metrics with bounded plot samples.

``predict/`` prepares FASTA windows and writes owned genomic/local BED intervals.
``variant/`` loads transcripts/models, scores reference/alternate windows and
publishes validated VCFs. Built-in annotation tables are package data. Model weights,
genomes and research data are separate resources.

Research and campaigns
----------------------

``validation/full_snv_concordance/`` implements VCF alignment, score denominators,
external-source reconciliation and map/reduce workflows. ``validation/concordance_study/``
implements summaries, scientific checks, provenance and report generation.
Campaign automation coordinates Slurm and validates completed outputs; its tests
simulate schedulers and filesystems. It is not installed by the package wheel.
The staged recovery bundle is separate from live signed r13 controls.

Standalone historical scripts under ``openspliceai/scripts`` and ``openspliceai/test``
retain historical research context and external-data requirements. They are not
supported CLI workflows. The source census in ``verification/comprehensive/``
records maintained, research and historical definitions separately.

Tests and documentation
-----------------------

``tests/unit`` checks local contracts, ``tests/regression`` checks reproduced defects,
``tests/integration`` runs real tiny workflows, and ``tests/equivalence`` compares
optional backends. :doc:`development` explains checks and evidence limits.
:doc:`function_manual` links public definitions and their signatures to source.
