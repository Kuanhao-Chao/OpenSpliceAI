# OpenSpliceAI tests

Install an isolated environment with `python -m pip install -e '.[dev]'`.
The Makefile uses the active `python`; override `PYTHON=/path/to/python` if needed.
CPU targets bound BLAS/TensorFlow threads and use the headless plotting backend.

| Command | Scope |
|---|---|
| `make test` | Fast unit/regression checks; integration, slow and optional backends excluded |
| `make test-cpu` | All CPU workflows, including end-to-end integration |
| `make test-all` | All tests; unavailable optional backends report skips |
| `make test-keras` | Original SpliceAI comparison; requires TensorFlow, spliceai and weights |
| `make test-gpu` | CUDA numerical comparison; does not clear visible devices |
| `make coverage` | Full available suite, 95% line-coverage gate |
| `make coverage-branch` | Full CPU coverage with separate 95% statement and 90% branch gates |
| `make lint` | Ruff for package and tests |
| `make package` | Build sdist and wheel |

`tests/unit` checks components; `tests/integration` exercises the six commands on
synthetic files; `tests/regression` protects strand labels, context schedules,
checkpoint loading, scoring and evaluation contracts; `tests/equivalence` compares
optional backends. Research helpers under `validation/` have synthetic tests for
count pooling, thresholds, provenance, VCF alignment and simulated scheduler state.
No tests submit production jobs. Test count and runtime depend on environment; use
`python -m pytest --collect-only` for the current inventory.

Expected scientific values are independently specified for strand labels, one-hot
encoding, loss/metric formulas, probability normalization, BED coordinates, all
four variant events and masks, and calibration bin counts. The four predict
storage/execution combinations must agree on complete BED rows. The existing Keras
comparison checks formatted DS/DP fields against the original SpliceAI software;
it does not establish biological accuracy or prove equivalence of independently
trained PyTorch and Keras weights. CUDA checks use explicit numerical tolerances.

The autouse fixture seeds Python, NumPy and Torch. This ensures controlled test
inputs, not fully reproducible CLI training. Scientific defaults and known
limitations are recorded in `KNOWN_ISSUES.md`.

Software CI defines Python 3.9–3.14, macOS, minimum-stack, full CPU, wheel/sdist and required Keras checks.
Configured jobs are distinct from executed evidence. Optional backend tests must
report skips precisely; initialization failures with an available backend fail.

Coverage omits legacy `openspliceai/scripts`, the disabled `test` command and
standalone alternatives listed in `.coveragerc`. Exclusions are not evidence those
features work. Keep the separate 95% statement and 90% branch gates. Required
backend runners fail on missing dependencies/devices/weights or skipped cases.
See `docs/development/comprehensive-audit.md` and `verification/comprehensive/`
for current evidence, per-function execution, feature contracts and pending gates.
