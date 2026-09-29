# Repository hardening audit — 29 September 2026

## Scope and provenance

The approved baseline is development commit `87ede55bbd8960bd40307526f56e0669ea7d9a62`,
merged with main `074d3e27d9950f48e7276ed69f0f0205f10ef223` in commit `3c5136a`.
Implementation is isolated on `audit/repo-hardening-20260929`; the production
checkout, inference jobs and shared Python environment are not modified.
Existing local deterministic-inference and atomic-VCF changes are carried forward.
Scientific defaults (loss, optimizer, scheduler, data layout and scoring) are preserved.

## Baseline evidence

The original checkout passed 440 tests in 772.58 seconds using Python 3.9,
PyTorch 2.2.1 and NumPy 1.26.4. Branch-aware coverage rounded to 94%; this is
not the same metric as the existing 95% line-coverage gate. Ruff found no violations.
This legacy environment has dependency conflicts and does not meet the development
branch's declared dependency floors; a clean environment is checked separately.
Logs: `/tmp/openspliceai-audit-20260929-tests.log`; coverage JSON:
`/tmp/openspliceai-audit-20260929-coverage.json`.

## Implementation checklist

- [x] Inspect baseline, local changes, six CLI workflows and existing tests.
- [x] Isolate development branch and merge current main documentation.
- [x] Reproduce and fix calibration count alignment and calibrated-count reuse.
- [x] Fail explicitly on invalid checkpoints and incomplete inference ensembles.
- [x] Test development MNV and transfer-learning features and scientific invariants.
- [x] Check evaluation/resource cleanup without changing numerical defaults.
- [x] Provide portable test commands, software CI and installation checks.
- [x] Run full coverage, optional backend checks, docs and distribution validation (available backends).
- [x] Record feature evidence and limitations; preserve changes on the isolated implementation branch.

Passing synthetic tests does not establish biological accuracy on unseen data or
prove every possible input correct. GPU execution and supported Python versions
must be reported separately from local CPU verification. Calibration reporting
corrections do not retroactively update published figures or production scores.

## Interim verification

- Focused defect tests reproduced 17 checkpoint/bin failures and 8 evaluation/resource
  failures before correction. The corresponding focused suites passed afterward.
- 146 configuration/regression tests passed in the clean Python 3.11 environment.
- Hand-specified AG/AL/DG/DL tests passed for both strands and masks; mock Keras
  profiles test the score algorithm independently of backend availability.
- All four prediction storage/execution modes produce identical complete BED rows
  for a 5kb-boundary crossing sequence and a short second entry.
- Compared with the original checkout, SNV/indel DS/DP fields agree at six decimals
  for both masks; model output arrays are bit-identical. Small CPU timing samples
  (0.193 and 0.113 seconds for ten forwards) are not controlled benchmarks.
- Clean installation uses Python 3.11.9, Torch 2.4.1+cpu and NumPy 2.4.6; pip check
  succeeds. Torch 2.3.0+cpu also passed a NumPy round-trip ABI probe on Linux.
- Source and wheel distributions build. Installed wheel runtime imports and a
  real checkpoint inference pass independently of the source import path.
- Final results supersede interim runs; the earlier 489-test run overlapped source
  edits, so its coverage output is discarded.

See `feature-matrix.md` for the inventory of executable checks. Python 3.10/3.12
CI and CUDA are not claimed as locally verified. Production main remains at
`074d3e27d9950f48e7276ed69f0f0205f10ef223`.

## Final verification

| Check | Executed result |
|---|---|
| Clean Python 3.11 / NumPy 2.4 / Torch 2.4.1 CPU suite | 533 passed, 6 optional tests deselected; 509.20 seconds |
| Representative NumPy 2.0.2 minimum scientific stack / Torch 2.3 | 505 fast tests passed; 160.79 seconds |
| Real Keras shared-allele parity plus loader error check | 4 passed; 591.81 seconds |
| Python 3.9 correction/cleanup/batching/report checks | 53 passed; 125.30 seconds |
| Clean line coverage | 96.11% (95% requirement retained) |
| Clean branch coverage | 89.15% (baseline 88.60%) |
| Ruff and whitespace check | Passed |
| Source and wheel builds | Passed; final installation checks are recorded below |

The first final runs exposed a malformed new teacher test fixture and missing
Markdown dependency. Both were corrected. The legacy full run's three MNV parity
failures exposed an actual contract difference between development OpenSpliceAI
and original SpliceAI; the revised shared-allele comparison passed. Those failed
runs are retained in the temporary logs and are not presented as green suites.

The development NumPy-2 dependency floor is now paired with h5py 3.11, scikit-learn
1.4.2, SciPy 1.13 and Matplotlib 3.8.4. The representative Linux scientific stack
is pinned in `ci/minimum-scientific.constraints` and tested separately. This does
not claim every version/platform combination is compatible. Upstream evidence:
[h5py 3.11 release notes](https://docs.h5py.org/en/3.14.0/whatsnew/3.11.html),
[scikit-learn 1.4.2 release notes](https://scikit-learn.org/1.4/whats_new/v1.4.html),
[SciPy 1.13 release notes](https://docs.scipy.org/doc/scipy-1.13.0/release/1.13.0-notes.html),
[NumPy ecosystem compatibility](https://github.com/numpy/numpy/issues/26191).

`verification/summary.json` records counts, coverage, package versions, artifact
checksums and unexecuted platforms. `verification/scoring-comparison.json` keeps
the representative DS/DP values. The feature matrix maps each capability to its
tests. Research/report helpers and their existing local tests are now versioned
with the implementation; generated genomes, score files and model symlinks are
excluded. No live scoring jobs were submitted or altered during this audit.

## Remaining limits

The Python 3.9 runtime was checked in a legacy environment with known dependency
conflicts; it is not a clean installation of the new NumPy-2 package requirements.
Python 3.10/3.12 and remote CI jobs are configured but not executed locally. CUDA
was unavailable. Real Keras tests used the existing Python 3.9 backend; a clean
NumPy-2 Keras installation is not claimed. The preserved focal loss, scheduler,
seed, freezing, HDF5 and BED-overlap behaviors are described in `KNOWN_ISSUES.md`.
The branch is ready for review; production adoption still requires a deliberate
transition away from the checkout used by signed running jobs.

Final distribution installation checks passed: seven help entrypoints (root plus
six commands), a nonzero invalid-command status, both built-in annotation resources,
six runtime module imports and real checkpoint inference from the installed wheel.
A wheel rebuilt from the final sdist installed and its entrypoint ran outside the
source tree. Sphinx 8.2.3 completed with warnings treated as errors; all 2,014 local
references in 37 pages resolve. The separate single-model real-Keras integration
check passed (39.62 seconds).

The existing analysis source and local report tests are preserved in commit
`87647b9`; core hardening, new tests, CI, dependency declarations and this audit
are recorded in the subsequent commit on `audit/repo-hardening-20260929`.
Standalone historical benchmark/figure scripts were inventoried but not executed
against external genomes. The optional Keras reference-weight symlink remains a
local test fixture and is not included in the commits or distributions.
