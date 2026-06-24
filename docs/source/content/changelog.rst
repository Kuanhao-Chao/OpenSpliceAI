
|

Changelog
===========

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

