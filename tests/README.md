# OpenSpliceAI test suite

Strict, reproducible, CPU-only tests for the packaged `openspliceai` pipeline
(create-data → train / transfer → calibrate → predict / variant).

## Running

All commands are wrapped in the repo `Makefile`, which pins the interpreter and forces
CPU-only, headless, bounded-thread execution so runs are deterministic:

```bash
make test       # fast unit + regression suite (integration & slow deselected)  ~10s
make test-all   # the whole suite incl. integration / slow / keras              ~3-4min
make coverage   # full suite + term/html coverage + gate (fails under the floor)
make lint       # ruff over openspliceai/ and tests/
```

> **Interpreter gotcha (this machine):** the default `python` has a broken numpy/torch ABI and is
> missing deps. The Makefile uses `/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python` (numpy 1.26,
> torch 2.2.1, scikit-learn/h5py/pyfaidx/gffutils/biopython/pysam; TensorFlow present so keras tests
> run). Override with `make test PYTHON=/path/to/python`. Install dev deps with `pip install -e '.[dev]'`
> (or `'.[test]'` for just pytest + pytest-cov).

Determinism is enforced by the autouse `_seed_everything` fixture (`conftest.py`), which seeds
`random` / `numpy` / `torch` before every test.

## Layout & taxonomy

| Directory | Purpose |
|---|---|
| `tests/unit/` | Pure-function / single-component tests (encoding, windowing, losses, metrics, model shape, BED coordinate math, chromosome splitting, argparse, CLI dispatch, model loaders, temperature scaling, plotting smoke). Fast, no heavy I/O. The bulk of coverage. |
| `tests/integration/` | End-to-end flows on tiny synthetic fixtures (create-data, train/transfer, predict, variant, calibrate, merge). Marked `integration` (+ `slow` for the training runs). |
| `tests/regression/` | Characterization tests that **lock** specific conclusions: hyperparameter-table sync across the 5 call sites, encode↔decode round-trips, minus-strand labeling, the `clip_datapoints` invariant, and batched==sequential variant scoring. |
| `tests/equivalence/` | Bit-exact parity vs the original Illumina SpliceAI (Keras, flanking 10000). Marked `keras`+`slow`+`integration`. |
| `tests/fixtures/synthetic.py` | Builders for the tiny on-disk schemas (HDF5 shards, mini genome+GFF, variant ref/TSV/VCF) — they mirror the *real* formats so loaders are exercised, not stand-ins. |
| `tests/conftest.py` | Shared fixtures (`model_80nt`, `packaged_80nt_state/_dir`, dataset builders) + the keras/gpu auto-skip hook. |

## Markers (`pytest.ini`)

- `integration` — end-to-end on tiny fixtures (CPU). Deselected by `make test`.
- `slow` — long-running. Deselected by `make test`.
- `keras` — needs TensorFlow/Keras; **auto-skipped** when TF is absent.
- `gpu` — needs CUDA; **auto-skipped** when no CUDA device is present.

## Coverage policy

`make coverage` runs the **full** suite (integration contributes most of the covered lines) with
`--cov-fail-under` (the `COV_MIN` variable in the `Makefile`). Pure/logic/encoding modules sit at
≥95%; the heavy I/O modules (`predict/predict.py`, `train_base/utils.py`, `variant/utils.py`) at ≥90%.

A few lines are excluded with `# pragma: no cover` because they are unreachable from the packaged
PyTorch CLI; listed here so the exclusions stay auditable:

- `openspliceai/train_base/utils.py` — `process_batch`, `test_SpliceAI_Keras_model`: Keras-only
  (`model.predict`) evaluation helpers, reachable only via the disabled `test` subcommand.

`.coveragerc` additionally omits legacy/dead modules: `openspliceai/scripts/*`, `openspliceai/test/*`,
`create_data/gff_to_tsv.py`, `variant/get_anno.py`, `calibrate/temperature_scaling_site_only.py`.

See `../KNOWN_ISSUES.md` for the behavior-changing issues that are deliberately deferred (and pinned
by characterization tests) and the two bugs fixed during the v0.0.7 test-hardening pass.
