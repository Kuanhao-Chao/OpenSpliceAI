.. _development_and_testing:

Development and verification
============================

Install ``python -m pip install -e '.[dev]'`` in an isolated environment, then run:

.. code-block:: bash

   make lint test
   make test-cpu
   make coverage-branch
   python examples/tutorial/run.py --output-dir /tmp/osai-tutorial
   python -m pip install -r docs/requirements.txt
   make -C docs html SPHINXOPTS='-W --keep-going'
   python docs/check_links.py docs/build/html

Fast checks exclude integration/slow/optional backends; the full CPU suite includes
real train/transfer/calibration/prediction/variant and research-tool fixtures.
Release gates require at least 95% statement and 90% branch coverage separately,
meaningful numerical regressions, wheel/sdist installations away from source,
Sphinx warnings-as-errors, local paths/anchors and browser inspection.

``make test-keras`` and ``make test-gpu`` are optional local selectors. A release
backend runner first asserts dependency/weight/device availability and treats
skipped required backend cases as missing evidence. GPU checks cover all four
contexts, inference, scoring and calibration. CI also configures Python 3.9–3.14,
a representative minimum NumPy-2 stack, a modern stack and macOS. Configuration
alone does not establish that those jobs passed.

Executed results, environment manifests, resource charges and unresolved platform
checks are recorded in ``verification/comprehensive/summary.json`` and
``docs/development/comprehensive-audit.md``. The feature matrix maps contracts to
tests. Coverage measures execution; it cannot prove every edge case or biological
claim correct. Historical external-genome scripts are inventoried separately.
