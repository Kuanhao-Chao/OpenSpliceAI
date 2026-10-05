# Portable CPU test entrypoints. Install with: python -m pip install -e '.[dev]'
PYTHON ?= python
RUFF ?= $(PYTHON) -m ruff
COV_MIN ?= 95
THREADS ?= 2
ENV := CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS) TF_NUM_INTRAOP_THREADS=$(THREADS) TF_NUM_INTEROP_THREADS=1 MPLBACKEND=Agg
PYTEST := $(ENV) $(PYTHON) -m pytest

.PHONY: help test test-all test-cpu test-keras test-gpu coverage coverage-branch lint package clean
help:
	@echo "test: fast unit/regression; test-cpu: full CPU; test-all: all available backends"
	@echo "test-keras/test-gpu: optional backends; coverage: 95% line gate; coverage-branch: branch report"
	@echo "lint: ruff; package: build sdist and wheel"
	@echo "Override PYTHON=/path/to/python and THREADS=2 as needed."
test:
	$(PYTEST) -m "not integration and not slow and not keras and not gpu" -q
test-cpu:
	$(PYTEST) -m "not keras and not gpu" -q
test-all:
	$(PYTEST) -q
test-keras:
	$(PYTEST) -m keras -q
test-gpu:
	OMP_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS) MPLBACKEND=Agg $(PYTHON) -m pytest -m gpu -q
coverage:
	$(PYTEST) --cov=openspliceai --cov-report=term-missing --cov-report=html --cov-fail-under=$(COV_MIN) -q
coverage-branch:
	$(PYTEST) --cov=openspliceai --cov-branch --cov-report=term-missing --cov-report=json:coverage-branch.json -q
	$(PYTHON) ci/check_coverage.py coverage-branch.json
lint:
	$(ENV) $(RUFF) check openspliceai tests
package:
	$(PYTHON) -m build
clean:
	rm -rf .pytest_cache htmlcov .coverage .coverage.* coverage-branch.json
