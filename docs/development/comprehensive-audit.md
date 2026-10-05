# Comprehensive repository audit — 5 October 2026

The corrected package and reorganized documentation are implemented on
`audit/comprehensive-20261004`, based on `d23e4413`. Scientific implementation
commit: `2ad31fea9a9eea929a4faa6f6cbf201a55851c70`. The staged version is
`0.1.0.dev0`. This is a review branch; no package or documentation release has
been published.

The final Linux CPU suite passed **702 tests**, with **96.256% statement coverage**
and **90.514% branch coverage** for the maintained installed package. The same
702 tests passed on the representative minimum scientific stack. Both built
distributions installed and ran real inference away from the source checkout.
The six-command synthetic tutorial and local documentation checks passed.
Required CUDA evidence is still pending; scheduled checks are not successful checks.

## Plan and current stage

| Step | Work | Status and evidence |
|---|---|---|
| 1 | Preserve production and isolate implementation | Worktree in `/tmp/openspliceai-comprehensive-20261004`; production remains at `074d3e27` with its two existing scorer patches. |
| 2 | Inventory source, functions and workflows | 245 Python files; 35 maintained package files with 217 definitions, 45 research/campaign files, and 59 historical files. See the source census and feature matrix. |
| 3 | Reproduce and correct scientific defects | Independent loss/gradient, scheduler, class-order, sample-retention, checkpoint and coordinate tests pass. |
| 4 | Exercise all six installed commands and boundaries | Full CPU suite, real tiny train/transfer/calibration/prediction/variant, data creation/merge and tutorial pass. |
| 5 | Test campaign recovery separately | Simulated filesystem/scheduler tests pass. New helpers are staged separately from live signed r13 controls. |
| 6 | Reorganize documentation | Generated CLI/API reference, concise workflow guides, migration notes and preserved historical URLs/anchors. All public maintained-package definitions have docstrings. |
| 7 | Verify installation and compatibility | Wheel/sdist and representative minimum-stack checks pass. Broader Python/macOS checks are configured in CI; record actual results separately. |
| 8 | Run required backends | Earlier frozen Keras run: 5 passed, no skips. Final frozen Keras job `31628903`: 5 passed, no skips. CUDA job `31628943` remains dependency-pending. |
| 9 | Record evidence and prepare review | Evidence in `verification/comprehensive/`; draft review must retain pending backend/platform gates. |
| 10 | Adopt or publish | Deferred until review and required checks pass; production transition needs validated r13 or an independently validated frozen runtime. |

## What changed and why

### Training and transfer

Focal alpha and gamma now affect values and gradients, with defaults 0.25 and 2.
MultiStepLR advances once per epoch; cosine restarts use progress across the
entire epoch. Python, NumPy and PyTorch are seeded before initialization.
Single-window shards and partial batches are retained unless training explicitly
selects `--drop-last`.

Validation uses the full selected split and resolves explicit validation filenames.
Metrics keep the background/acceptor/donor order when a class is absent.
Zero-label padding contributes no loss, gradient, calibration count or evaluation
observation. The historical metric named `accuracy` is macro class recall; its
name is retained and documented.

Transfer trains the output head plus the requested residual units. Frozen
BatchNorm buffers stay fixed. Students load strictly by default; partial student
initialization is explicit and teachers always load strictly. Distillation uses
the actual three-channel anchor labels.

### Calibration and checkpoints

Calibration caches logits once on disk and reads bounded chunks during fitting.
One full-objective gradient update is made per epoch; epochs, early stopping,
learning rate and memory options are honored. The best post-update state is
selected, including identity temperature as a candidate.

NLL, ECE and Brier statistics use all observed positions. Curves use occupied-bin
counts; bounded plotting samples are labeled separately. Portable calibrated
artifacts validate version, class order, context and three finite floating-point
temperatures. Raw and calibrated checkpoints work directly in prediction/variant.
Strict finite/shape checks and atomic saves prevent silent partial ensembles and
failed replacement of existing artifacts.

### Prediction and variant scoring

All four prediction routes share ownership and coordinates. Split segments emit
every source position once, exclude padding and map both strands correctly,
including accession contigs. Eight real split/unsplit comparisons cover text/HDF5
and turbo/predict-all routes: the same 2,001 owned BED rows, probabilities within
`1e-6`. Overlap between distinct annotated genes remains biologically meaningful.

Variant requires valid models/annotations and uses scoped FP32 settings, restoring
the caller's PyTorch flags on success and failure. Diagnostics go to stderr.
Atomic file publication validates finite numeric DS/DP fields and supports BGZF.
Library failures raise exceptions. Exact contig names take precedence over optional
`chr` aliases, preserving scoring on mixed-prefix references.

Original SpliceAI parity applies only to shared supported variants with the
original 10,000-base Keras weights. MNV/delins extensions and rejected boundaries
have separate tests; original placeholders are not numeric parity evidence.

### Data preparation and research tools

Gene-grouped validation prevents isoforms crossing splits, while isoforms retain
independent labels. Advertised biotypes, GFF/GTF transcript/gene-type fields,
coordinates, strands and ratios have explicit checks. Annotation databases store
source SHA-256 and rebuild stale/corrupt caches atomically. Paralogy filtering
uses query-span coverage and fails if minimap2 cannot initialize. HDF5 verification
examines every paired shard, including validation, metadata and empty files.
Merge validates encoding/context and publishes each file atomically. Resource
handles close on failures.

Staged campaign helpers retry bounded transient I/O, verify atomic JSON readback,
avoid redundant throttle writes and reconcile ambiguous submissions using a
durable identity. Accounting excludes batch/extern steps and duplicate generic/
typed GPU TRES. Tests use simulated services. **Live r13 is not restarted or changed.**

## Verification results

| Check | Executed result | Qualification |
|---|---|---|
| Final CPU | 702 passed, 14 deselected; 381.13 s | Python 3.11.9, Torch 2.4.1+cpu, NumPy 2.4.6. The 14 backend cases run separately. |
| Package statements | 3,214 / 3,339 = 96.2564% | Separate 95% gate passed. |
| Package branches | 916 / 1,012 = 90.5138% | Separate 90% gate passed; combined percentage is not substituted. |
| Representative minimum stack | 702 passed, 14 deselected; 274.02 s | Torch 2.3.0+cpu, NumPy 2.0.2, h5py 3.11, SciPy 1.13, sklearn 1.4.2, matplotlib 3.8.4. Lower-bound overlay, not a fresh full resolution. |
| Initial Keras | 5 passed, no skips; 117.12 s | Frozen source-v2, TensorFlow/tf-keras 2.18, SpliceAI 1.3.1, NumPy 2.0.2. Final current-source repeat passed all five cases in 88.12 s; no skips. |
| Wheel and sdist | Both installed; CLI, annotations and exact raw/calibrated inference passed | Isolated dependency copies, executed from `/tmp`; package paths asserted within each environment; `pip check` passed. |
| Tutorial | Six commands plus both calibrated inference commands passed | Tiny synthetic model/inputs; no biological performance claim. |
| Sphinx | Warnings-as-errors passed | 39 HTML pages. |
| Documentation references | 2,690 local paths/fragments resolved | Old page URLs and 215 historical section anchors retained. |
| Browsers | Chromium/Firefox: all pages/search and 32 responsive/theme checks passed | 1440/390 px, light/dark, visible logos, no key-page horizontal overflow. |
| Lint/formatting | Passed | Package, tests, CI tools, recovery module and `git diff --check`. |
| Research coverage | 62.5736% statements, 56.8140% branches | Separate external-data/unexecuted gaps; package coverage does not imply research coverage. |

Per-function coverage records execution and missing lines, not universal correctness.
Historical standalone research scripts are inventoried, not claimed fully tested.
Baseline/minimum runs had nine/211 warnings, including upstream deprecations and
scheduler warnings in tests mocking optimizer steps; no failures.

## Backend jobs and exact resource ledger

Audit limit: **200 Slurm billing-hours and two A100-hours**, including failures.
Only one audit allocation runs at a time. Each requests 12 CPUs and 32 GiB for
at most 45 minutes; CUDA also requests one typed A100. Frozen source-v3 verifies
233 files before execution. Manifest SHA-256:
`57e3b693c99e9b0eed01c1a25e01dd21203bd50596bdea8a6cabecd6e492226a`.

| Job | Account / partition | State at scheduling | Maximum / exact charge |
|---|---|---|---|
| 31606547 | ssalzbe1_bigmem / bigmem | Earlier Keras completed, exit 0:0 | Exact 171 s × billing 12 / 3,600 = **0.57 billing-hours**, zero GPU-hours. |
| 31628903 | ssalzbe1_bigmem / bigmem | Final Keras completed, exit 0:0 | Exact 127 s × billing 12 / 3,600 = **0.4233 billing-hours**, zero GPU-hours. |
| 31628943 | ssalzbe1_gpu / a100 | CUDA waiting on dependencies | At most **9 billing-hours and 0.75 A100-hours**. |

CUDA waits for **successful final Keras and completion of the entire r13 GPU array
31494759**, preserving all six production slots. Saved Slurm readback confirms
`afterok:31628903` and `afterany:31494759_*`. This also prevents simultaneous audit
allocations. Required-backend runners fail on missing dependencies/weights/devices
or skipped required tests.

Exact completed consumption is **0.9933 billing-hours** and zero A100-hours. The
pending CUDA job reserves at most nine more billing-hours and 0.75 GPU-hours. If
it uses its full limit, **190.0067 billing-hours and 1.25 A100-hours** remain.
These are credits, not USD; GPU occupancy is not billed a second time.

Captured bigmem/GPU balances exceed their 25% quarterly floors. Shared lab usage
can change balances. Keep caps of 100 running scoring/utility tasks, 1,200 CPU units,
six typed A100s and 14 bigmem allocations including any controller. No production
throttle was changed.

## Production r13 status and boundary

The refreshed scheduler snapshot around 2026-10-05 11:55 UTC recorded **99 running
tasks and 1,188 CPUs**, including six A100s. CPU lanes had 55/25/13 running tasks,
filling their ceilings. Each scorer requests 12 CPUs and 32 GiB; requested RAM is not
observed usage. Older arrays 29611225/29611226 remained held.

Controller `31496207` was absent from the running queue. Saved phase:
`needs_attention`; `OSError(5, 'Input/output error')` recorded October 2.
Workers continue, but controller-driven repair/adaptation/final-audit scheduling
does not run. The October 3 forecast of October 10–12 validated completion was
conditional on recovery; it is a dated estimate, not a new completion guarantee.
A fresh ETA requires a current publication census, successful timings and repair plan.

Production remains on `074d3e27` plus its two existing scorer patches. This audit
modifies no production source, frozen models, signed controls or published scores.
Shared `.audit/` environments/snapshots are separate. Staged controller changes
need their own reviewed, frozen transition.

## Remaining gates and next actions

1. Final Keras results and exact allocation consumption are recorded; all five cases passed.
2. Record all nine CUDA cases after its dependency frees capacity. A queued job
   has no trustworthy start time while the dependency remains unresolved.
3. Review the branch and record successful Python/macOS/minimum-install CI results.
   Configuration alone is not platform evidence. The local Conda recipe now matches
   the version/dependency floors, but a Conda build has not been executed.
4. Update ledger/manifests for any correction, include failed allocations, and never
   edit an in-use frozen snapshot.
5. Review r13 recovery separately and refresh the ETA before final repair/audit.
6. Merge or publish after required gates and release authorization. Historical
   reproduction uses earlier source/environment, without a buggy training mode.

See [feature contracts](../../verification/comprehensive/feature-matrix.md),
[source census](../../verification/comprehensive/source-inventory.json),
[summary](../../verification/comprehensive/summary.json),
[migration](../source/content/migration.rst), and
[runbook](../../verification/comprehensive/README.md).
