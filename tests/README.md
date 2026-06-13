# OpenSpliceAI test suite

## Environment

Use an interpreter with the pinned scientific stack (numpy 1.26, torch 2.2, scikit-learn, h5py,
pyfaidx, gffutils, biopython, pysam). On this machine that is the `pytorch_cuda` conda env. The base
`python` has a broken numpy/torch ABI and is missing dependencies.

```bash
ENV=/home/kchao10/miniconda3/envs/pytorch_cuda/bin
$ENV/pip install -e '.[test]'      # installs pytest + pytest-cov
```

## Running

```bash
# Fast, CPU-only unit + regression tests (no GPU, no TensorFlow):
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest tests/unit tests/regression \
    -m "not gpu and not keras and not integration" -q

# Full suite incl. end-to-end smoke tests (still CPU; keras-marked tests auto-skip without TF):
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest tests -q

# Coverage:
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest tests --cov=openspliceai --cov-report=term-missing -q
```

## Layout

- `tests/unit/` — pure-function tests (encoding, windowing, losses, metrics, model shape, BED
  coordinate math, chromosome splitting, argparse validation).
- `tests/regression/` — locks in two behaviors that an automated audit wrongly flagged as bugs
  (minus-strand labeling, `clip_datapoints` X/Y handling). See `../KNOWN_ISSUES.md`.
- `tests/integration/` — end-to-end smoke tests on tiny fixtures proving the bug fixes for
  `calibrate`, the `predict` non-HDF5 path, and the PyTorch `variant` path. Marked `integration`/`slow`.
- `tests/fixtures/synthetic.py` — builders for tiny HDF5 datasets and a mini gffutils annotation.

## Markers

`gpu` (needs CUDA), `keras` (needs TensorFlow), `slow`, `integration`. GPU/keras tests auto-skip when
the backend is unavailable.
