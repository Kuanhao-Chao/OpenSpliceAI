
|

Changelog
===========

v0.0.8.dev0 (unreleased)
------------------------

**New features**

- ``transfer``: added a suite of **catastrophic-forgetting mitigations** for fine-tuning on
  narrow datasets (all optional, default-off, so existing runs are unchanged):

  - ``--weight-decay`` exposes the AdamW weight decay (default ``0.01``); set ``0`` to stop
    decaying pretrained weights toward zero. ``--l2sp`` instead regularizes the trainable
    weights toward the **pretrained** weights (active with a distillation teacher).
  - ``--rehearsal-dataset`` / ``--rehearsal-shards`` interleave real genomic shards
    (experience replay) into training.
  - ``--distill-weight`` / ``--distill-teacher`` / ``--distill-shards`` /
    ``--distill-batch-size`` add a knowledge-distillation (Learning-without-Forgetting) loss
    against a frozen teacher on genomic anchor windows — no genomic labels required.
  - ``--genomic-eval-dataset`` logs a per-epoch **forgetting curve** (donor/acceptor AUPRC +
    top-k) under ``LOG/GENOMIC/`` so you can pick the checkpoint on the gain-vs-retention front.

  See the ``transfer`` docs ("Mitigating catastrophic forgetting") and
  ``examples/transfer/transfer_forgetting_cmd.sh``.

- ``variant``: **multi-nucleotide variants (MNVs / delins)** — records where REF and ALT are
  both multiple bases — are now scored instead of emitting a ``.|.|.|.`` placeholder
  (`#18 <https://github.com/Kuanhao-Chao/OpenSpliceAI/issues/18>`_, thanks to @bpow). Deletion,
  insertion, and MNV score-reshaping are unified into one expression, and the "ref too long"
  guard is tightened to ``len(REF) > dist_var + 1`` — the exact point beyond which the reshape
  can no longer realign within the score window (this also fixes a latent bug where large
  deletions previously produced silently misaligned scores). This enables scoring
  reference-anchored multi-variant windows as a single MNV record; see "Scoring custom
  sequences" in the ``variant`` docs (addresses
  `#15 <https://github.com/Kuanhao-Chao/OpenSpliceAI/issues/15>`_).

**Bug fixes**

- CLI: ``openspliceai`` (and ``openspliceai --help``) no longer imports the full dependency
  stack at startup. Each subcommand's heavy dependencies (torch, pandas, scikit-learn/scipy,
  biopython, pysam, ...) are now imported lazily inside dispatch, so an import-time failure in
  a single transitive dependency can no longer break the whole CLI
  (`#19 <https://github.com/Kuanhao-Chao/OpenSpliceAI/issues/19>`_:
  ``module 'numpy' has no attribute 'long'`` on a no-argument invocation).

**Dependencies**

- Floored ``numpy>=2.0`` (together with ``torch>=2.3`` and ``pandas>=2.2.2``). numpy removed
  ``np.long`` in 1.24 and re-added it in 2.0, so 1.24–1.26 is a gap where a numpy-2-era
  dependency reading ``np.long`` fails to import. OpenSpliceAI's own code is numpy-2.0 clean,
  so keeping the whole stack on the numpy-2 side of that gap avoids the failure (issue #19).

v0.0.7 (2026-06-23)
-------------------

**New features**

- ``variant``: added ``--batch-size``/``-b`` to enable **batched inference** on PyTorch
  models. With ``--batch-size > 1`` the subcommand buffers records and scores their
  reference/alternate windows in batched forward passes (deduplicating the shared reference
  window across a position's alternate alleles), for large speedups on many-variant VCFs.
  The default (``1``) preserves the exact original per-variant path bit-for-bit. Two optional
  environment knobs tune throughput: ``OSAI_CUDNN_BENCH`` (cuDNN autotuning) and ``OSAI_TF32``
  (A100 TF32 fast path; set ``OSAI_TF32=0`` for bit-reproducible output).
- ``predict``: added ``--gene-flank``. With ``-a/--annotation``, it includes real genomic
  flanking sequence on each side of every extracted gene so the model sees true context
  instead of ``N`` padding at gene boundaries (default ``-1`` uses ``flanking_size/2``; set
  ``0`` for the legacy bare-gene-body behavior). ``predict`` now also warns when an input
  sequence is shorter than the model's required context and clarifies strand handling when no
  annotation is supplied (closes
  `#16 <https://github.com/Kuanhao-Chao/OpenSpliceAI/issues/16>`_).

**Packaging & distribution**

- OpenSpliceAI is now installable from **Bioconda**:
  ``conda install -c conda-forge -c bioconda openspliceai``.
- Corrected the conda recipe to **GPL-3.0** built from the PyPI source tarball + checksum
  (it previously declared ``MIT`` and built from a git tag), and added GPLv3 license metadata
  and trove classifiers to ``setup.py``.
- Pruned three unused dependencies (``torchaudio``, ``torchvision``, ``matplotlib-inline``)
  from ``install_requires``.
- Added a regression test locking the new batched ``variant`` path to produce delta scores
  identical to the per-variant path (SNV / deletion / insertion / multi-allelic).

**Bug fixes**

- ``variant``: the reference one-hot encoder now folds every non-ACGT base (``N``, IUPAC
  ambiguity codes, gaps) to the all-zero row, matching its documented contract — previously
  only a literal ``N`` was handled and other characters were silently miscoded. ACGTN
  reference sequence is encoded bit-identically, so real-genome scores are unchanged.
- ``predict``: fixed a crash in the ``neg_strands`` reverse-complement path of
  ``get_sequences`` (it called a sequence-object method on a plain string).

**Testing & quality**

- Greatly expanded the pytest suite (now ~300 tests) to **~96% line coverage** of the packaged
  pipeline, including characterization tests that lock the cross-subcommand hyperparameter
  table, encode↔decode round-trips, and batched-vs-sequential variant equivalence.
- Added a one-command, reproducible test entrypoint (``Makefile``: ``make test`` / ``test-all``
  / ``coverage`` / ``lint``) with a coverage-floor **gate** (``--cov-fail-under``), a
  ``tests/README.md`` describing the taxonomy and coverage policy, and ``KNOWN_ISSUES.md``
  references to each locking test.

v0.0.6 (2026-06-13)
-------------------

**Validation**

- Audited the ``predict`` and ``variant`` subcommands step by step (see ``validation/``).
  ``variant --model-type keras --flanking-size 10000`` reproduces the original Illumina
  ``spliceai`` tool **exactly** (every delta score and position, across a mask/distance grid),
  and ``predict`` coordinates were verified correct on both strands.

**Bug fixes**

- Fixed a crash in the ``calibrate`` subcommand so it now runs end-to-end (fits a temperature,
  reports ECE/NLL/Brier, and writes calibration plots and ``temperature.pt``/``.txt``).
- Fixed ``predict`` checkpoint (``.pt``) path handling when loading a single model file.
- Fixed the ``variant`` built-in annotation paths: the ``grch37``/``grch38`` tables now **ship
  inside the package** (``openspliceai/variant/annotations/{grch37,grch38}.txt``) and are
  resolved via ``importlib.resources``, so they work regardless of the current working
  directory or install location.
- Fixed layer freezing in the ``transfer`` subcommand.
- Restricted and validated ``--flanking-size`` to ``{80, 400, 2000, 10000}`` across all
  subcommands, and hardened ``predict`` to raise on an unsupported value instead of silently
  defaulting to the 80 nt schedule.
- Fixed a crash in ``variant`` when ``--output-vcf`` was a bare filename (empty directory name).
- Fixed a path-handling bug in the ``--remove-paralogs`` (paralog removal) flow of
  ``create-data`` (the datafile/removed-paralog paths now use ``os.path.join``, so an
  ``--output-dir`` without a trailing slash works).
- Fixed a crash in ``create-data --verify-h5`` on small datasets: the verification step
  hardcoded a chunk index (``X3``) and raised ``KeyError`` for datasets with fewer than four
  chunks; it now visualizes the last available chunk.

**Packaging, testing & tooling**

- Stopped shipping the top-level ``tests`` package in the built wheel
  (``find_packages(include=['openspliceai', 'openspliceai.*'])``).
- Added a ``pytest`` test suite (~143 tests) under ``tests/`` (``unit``, ``integration``,
  ``regression``, ``equivalence`` layers with shared synthetic fixtures), designed to run
  CPU-only and cover every subcommand end-to-end — including a keras-vs-original-SpliceAI
  equivalence regression and predict/variant invariants.
- Added ``ruff`` linting (``ruff.toml``), ``pre-commit`` hooks (``.pre-commit-config.yaml``),
  and coverage configuration (``.coveragerc``), plus a ``dev`` install extra
  (``pip install -e '.[dev]'``). See :ref:`development_and_testing` for details.
- Condensed the README "Development & Testing" section.

Initial release
---------------

- Initial release of OpenSpliceAI, distributed on `PyPI
  <https://pypi.org/project/openspliceai/>`_ and `GitHub
  <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_.
- Released via the documentation (https://ccb.jhu.edu/openspliceai) and the paper
  (https://doi.org/10.7554/eLife.107454).


|
|
|
|
|



.. image:: ../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center

