
|


.. _development_and_testing:

Development & Testing
=====================

This page describes how to set up a development environment for OpenSpliceAI and how to run
its test suite and quality-control tooling. It is intended for contributors and for anyone who
wants to verify a local change before opening a pull request.

|

Installing the development dependencies
---------------------------------------

Install OpenSpliceAI in editable mode together with the optional ``dev`` extra. This pulls in
the test runner, linter, pre-commit, and coverage tooling on top of the runtime requirements:

.. code-block:: bash

   pip install -e '.[dev]'

The ``dev`` extra installs ``pytest`` (test runner), ``pytest-cov`` (coverage), ``ruff``
(linter), and ``pre-commit`` (git hooks). A lighter ``test`` extra is also available
(``pip install -e '.[test]'``) if you only need ``pytest`` and ``pytest-cov``.

.. note::

   ``tensorflow`` / ``keras`` are **not** part of the ``dev`` extra. They are only required for
   the small set of Keras-specific tests (and for scoring with the original Keras SpliceAI
   models in the ``variant`` subcommand). Tests that need them are auto-skipped when the
   packages are absent.

|

Running the test suite
-----------------------

The suite lives under ``tests/`` and contains **122 tests** (collected via
``pytest --collect-only``). It is designed to run **CPU-only** so it can execute on any
machine, including CI runners without a GPU.

Run the full suite from the repository root:

.. code-block:: bash

   pytest

For a fast inner-loop run, deselect the heavier end-to-end and long-running tests using the
markers (see below):

.. code-block:: bash

   # Fast: skip end-to-end integration and slow tests
   pytest -m "not integration and not slow"

   # Full: everything, including the tiny end-to-end flows
   pytest

You can also scope the run to a single layer or file:

.. code-block:: bash

   pytest tests/unit                      # unit tests only
   pytest tests/integration               # end-to-end smoke tests only
   pytest tests/unit/test_variant_utils.py

|

Test markers
~~~~~~~~~~~~~

Markers are declared in ``pytest.ini`` and enforced with ``--strict-markers`` (an unknown
marker is an error). The available markers are:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Marker
     - Meaning
   * - ``integration``
     - End-to-end flow exercised on tiny synthetic fixtures (CPU only).
   * - ``slow``
     - Long-running test; deselect for a quick inner-loop run.
   * - ``gpu``
     - Requires a CUDA device; skipped on CPU-only machines.
   * - ``keras``
     - Requires ``tensorflow`` / ``keras`` (heavy, optional; auto-skipped if absent).

Select or deselect markers with ``-m``, e.g. ``pytest -m integration`` to run only the
end-to-end flows, or ``pytest -m "not slow"`` to skip the long-running tests.

|

Layout of ``tests/``
--------------------

The suite is organized into three layers plus shared fixtures:

.. code-block:: text

   tests/
   ├── conftest.py            # shared pytest fixtures / configuration
   ├── fixtures/              # synthetic genomes, annotations, and datasets
   │   └── synthetic.py
   ├── unit/                  # fast, isolated tests of individual functions
   ├── integration/           # end-to-end smoke tests on tiny fixtures (CPU)
   └── regression/            # guards against previously fixed bugs

- **unit/** — fast, isolated checks of individual functions and classes (CLI argument
  parsing, the model forward pass, one-hot encoding utilities, temperature scaling, transfer
  layer freezing, variant delta-score math, etc.).
- **integration/** — minimal end-to-end runs of each subcommand
  (``create-data``, ``train``, ``transfer``, ``calibrate``, ``predict``, ``variant``) on the
  tiny synthetic fixtures, verifying that the pipelines run on CPU and produce the expected
  output files.
- **regression/** — targeted tests that lock in the behavior of previously fixed bugs (e.g.
  minus-strand labeling and the ``clip_datapoints`` cropping invariant).
- **fixtures/** — synthetic, deterministically generated inputs (a mini genome, GFF
  annotation, HDF5 datasets, and VCF/variant inputs) so the suite has no large external data
  dependencies.

|

Linting with ruff
------------------

OpenSpliceAI uses `ruff <https://docs.astral.sh/ruff/>`_ for linting, configured in
``ruff.toml``. The configuration focuses on real correctness issues (pyflakes ``F`` checks)
plus basic errors/warnings (``E``, ``W``); line-length (``E501``) and ambiguous single-letter
names (``E741``) are intentionally ignored, and legacy/experimental trees are excluded.

.. code-block:: bash

   ruff check .          # report lint issues
   ruff check --fix .    # auto-fix what ruff can fix safely

|

Pre-commit hooks
----------------

A `pre-commit <https://pre-commit.com/>`_ configuration (``.pre-commit-config.yaml``) runs
``ruff`` (with ``--fix``) and a couple of lightweight hygiene hooks (``check-yaml``,
``check-added-large-files``) on every commit. Install the git hook once, then optionally run
it across the whole repository:

.. code-block:: bash

   pre-commit install            # install the git hook
   pre-commit run --all-files    # run all hooks on every file

|

Coverage
--------

Coverage is configured in ``.coveragerc`` (measuring the ``openspliceai`` package and omitting
the legacy ``scripts``/``test`` trees). Collect coverage while running the suite via
``pytest-cov``:

.. code-block:: bash

   pytest --cov=openspliceai --cov-report=term-missing

Add ``--cov-report=html`` to generate a browsable HTML report under ``htmlcov/``.

|

Continuous validation
----------------------

There is no proprietary build system to run — validation is exactly the commands above:
``pytest`` for behavior, ``ruff`` for static checks, and ``pre-commit`` to wire both into the
commit workflow. Because the suite is CPU-only and self-contained (all inputs are synthetic
fixtures), it can be run anywhere without GPUs or large downloads.

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
