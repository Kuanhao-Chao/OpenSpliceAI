# Reproduce and extend this verification

Start with [the audit report](../../docs/development/comprehensive-audit.md) and
[feature matrix](feature-matrix.md). `summary.json` records executed checks and
pending gates; raw coverage, JUnit, backend/environment and scheduler files provide
the evidence. `artifacts.json` binds recorded files to their SHA-256 checksums.

## Local CPU and documentation

Use an isolated environment with Python 3.11 and a tested scientific stack:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]' build -r docs/requirements.txt
python -m pip check
make lint
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLBACKEND=Agg python -m pytest -m 'not keras and not gpu' \
  --cov=openspliceai --cov-branch --cov-report=json:coverage.json \
  --junitxml=cpu-tests.xml -q
python ci/check_coverage.py coverage.json
python examples/tutorial/run.py --output-dir /tmp/osai-tutorial
python -m build
make -C docs html SPHINXOPTS='-W --keep-going'
python docs/check_links.py docs/build/html
```

Install each distribution into a separate environment and execute from outside
the source tree using `ci/smoke_distribution.py --expected-prefix ENVIRONMENT`.
It asserts installed module paths, annotation package data and real inference.
The recorded local environments copied a verified dependency stack; a complete
fresh dependency resolution is tested separately by CI.

Install Playwright and both browser engines for the optional local browser check:

```bash
python -m pip install playwright
python -m playwright install chromium firefox
python ci/check_docs_browser.py docs/build/html --output-dir /tmp/osai-docs-browser
```

## Required backends and scheduled jobs

Keras requires TensorFlow/tf-keras 2.18, SpliceAI 1.3.1, original weights and
`TF_USE_LEGACY_KERAS=1` set before startup. CUDA requires exactly one visible device.
`ci/required_backend.py BACKEND --output-dir OUTPUT` first asserts dependencies,
then fails if any required case skips. Selectors alone are not successful evidence.

The final Slurm jobs use frozen source-v3, shared audit environments and no `/tmp`
source dependencies. Its manifest covers 233 files and is verified before startup.
Do not edit an in-use snapshot. Create a new destination for any correction:

```bash
python ci/freeze_backend.py --destination /shared/audit/source-new \
  --weights-root /path/to/reference-checkout
```

Job `31628903` repeats Keras on final source. Job `31628943` runs nine CUDA cases
only after Keras success and completion of r13 GPU array `31494759`. Both request
12 CPUs, 32 GiB and 45 minutes; CUDA adds one A100. Shared outputs:

```text
/data/ssalzbe1/khchao/OpenSpliceAI/.audit/comprehensive-20261004/source-v3/verification/keras/
/data/ssalzbe1/khchao/OpenSpliceAI/.audit/comprehensive-20261004/source-v3/verification/gpu/
```

Read status and allocation-only accounting:

```bash
squeue -j 31628903,31628943
sacct -X -D -j 31606547,31628903,31628943 \
  --format=JobIDRaw,Account,State,ExitCode,Start,End,ElapsedRaw,AllocTRES -P
```

Use allocation billing TRES × elapsed seconds / 3,600, retaining every distinct
requeued allocation and excluding `.batch`/`.extern`. A100 occupancy is a separate
limit, not an additional duplicate charge. Include failed jobs in the 200-hour/
two-A100-hour audit budget. Do not resubmit after an ambiguous acknowledgement
until the existing job is reconciled.

When jobs finish, copy their summary/JUnit/logs into this directory, capture exact
accounting, update `summary.json` and the report, and regenerate `artifacts.json`.
Maintain the pending state if a required backend has not actually passed. Record
successful compatibility CI links and commit hashes before a release claim.

## Production boundary

The production source/checkpoints, held arrays and signed controls remain separate.
An inactive r13 controller still needs a reviewed recovery plan. Package changes
must not be imported by a running production worker or silently replace provenance.
Whole-genome final content validation remains a separate campaign completion gate.
