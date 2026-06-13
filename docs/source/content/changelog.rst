
|

Changelog
===========

v0.0.6 (2026-06-13)
-------------------

**Validation**

- Audited the ``predict`` and ``variant`` subcommands step by step (see ``validation/``).
  ``variant --model-type keras --flanking-size 10000`` reproduces the original Illumina
  ``spliceai`` tool **exactly** (every delta score and position, across a mask/distance grid),
  and ``predict`` coordinates were verified correct on both strands.

**Bug fixes**

- Fixed a crash in ``variant`` when ``--output-vcf`` was a bare filename (empty directory name).
- Hardened ``predict`` to raise on an unsupported ``--flanking-size`` instead of silently
  defaulting to the 80 nt schedule.

**Packaging & tooling**

- Stopped shipping the top-level ``tests`` package in the built wheel
  (``find_packages(include=['openspliceai', 'openspliceai.*'])``).
- Expanded the test suite to **143 tests** (added a keras-vs-original-SpliceAI equivalence
  regression and predict/variant invariants) and condensed the README "Development & Testing"
  section.

v1.0.1 (2026-06-13)
-------------------

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
  subcommands, including ``predict`` and ``variant``.
- Fixed a path-handling bug in the ``--remove-paralogs`` (paralog removal) flow of
  ``create-data`` (the datafile/removed-paralog paths now use ``os.path.join``, so an
  ``--output-dir`` without a trailing slash works).
- Fixed a crash in ``create-data --verify-h5`` on small datasets: the verification step
  hardcoded a chunk index (``X3``) and raised ``KeyError`` for datasets with fewer than four
  chunks; it now visualizes the last available chunk.

**Testing & tooling**

- Added a ``pytest`` test suite (123 tests) under ``tests/`` (``unit``, ``integration``,
  ``regression`` layers with shared synthetic fixtures), designed to run CPU-only, covering
  every subcommand end-to-end.
- Added ``ruff`` linting (``ruff.toml``), ``pre-commit`` hooks
  (``.pre-commit-config.yaml``), and coverage configuration (``.coveragerc``).
- Added a ``dev`` install extra (``pip install -e '.[dev]'``). See
  :ref:`development_and_testing` for details.

v1.0.0
-------

- Initial release of OpenSpliceAI
- Release via the documentation (http://ccb.jhu.edu/openspliceai)
- Released via the paper (https://doi.org/10.1101/2023.07.27.550754)


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

