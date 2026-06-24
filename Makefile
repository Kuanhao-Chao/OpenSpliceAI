# OpenSpliceAI — reproducible test & quality entrypoints.
#
# All targets pin the interpreter and force CPU-only, single-threaded-ish, headless
# execution so runs are deterministic and copy-pasteable across machines. Override the
# interpreter on the CLI, e.g.:  make test PYTHON=python
#
#   make test       fast unit/regression suite (integration + slow deselected)  ~10s
#   make test-all   the whole suite incl. integration/slow/keras                 ~3-4min
#   make coverage   full suite + coverage report + gate (fails under $(COV_MIN)%)
#   make lint       ruff over the package and the tests
#
# The coverage gate intentionally runs the FULL suite: integration tests exercise the
# create-data / train / predict / variant / calibrate pipelines end-to-end and contribute
# most of the covered lines. See tests/README.md for the coverage policy.

PYTHON ?= /home/kchao10/miniconda3/envs/pytorch_cuda/bin/python
RUFF   ?= /home/kchao10/miniconda3/envs/pytorch_cuda/bin/ruff
COV_MIN ?= 95

# CPU-only + headless + bounded BLAS threads => deterministic, prompt-free runs.
ENV := CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
PYTEST := $(ENV) $(PYTHON) -m pytest

.PHONY: help test test-all coverage lint clean

help:
	@echo "OpenSpliceAI make targets:"
	@echo "  make test       - fast unit/regression suite (integration + slow deselected)"
	@echo "  make test-all   - full suite (integration, slow, keras if TF present)"
	@echo "  make coverage   - full suite + term/html coverage + gate (>= $(COV_MIN)%)"
	@echo "  make lint       - ruff check over openspliceai/ and tests/"
	@echo "  make clean      - remove coverage/pytest caches"
	@echo "Override the interpreter with: make <target> PYTHON=/path/to/python"

test:
	$(PYTEST) -m "not integration and not slow" -q

test-all:
	$(PYTEST) -q

coverage:
	$(PYTEST) --cov=openspliceai --cov-report=term-missing --cov-report=html \
		--cov-fail-under=$(COV_MIN) -q

lint:
	$(ENV) $(RUFF) check openspliceai tests

clean:
	rm -rf .pytest_cache htmlcov .coverage .coverage.*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
