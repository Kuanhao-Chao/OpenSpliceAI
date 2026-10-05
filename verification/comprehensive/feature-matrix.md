# Feature contracts and verification

This matrix covers the maintained installed workflows and selected research
contracts. Paths name executable tests in this branch. The CPU suite executes
all 702 CPU cases; required Keras/CUDA cases are separate. A passing finite test
set is not a proof of universal correctness or biological performance.

| Area | Contract and edge cases | Test evidence |
|---|---|---|
| CLI | Six workflows, argument bounds, lazy imports, invalid paths/options, typed dispatch | `tests/unit/test_cli_argparse.py`, `test_cli_dispatch.py`; `tests/regression/test_lazy_cli_imports.py`, `test_public_contract_guards.py`; six-command tutorial |
| Architecture | Four contexts (80/400/2000/10000), channel order, output length, independent hyperparameter arrays | `tests/unit/test_model_forward.py`; `tests/regression/test_hyperparam_table_sync.py`, `test_clip_datapoints_invariant.py` |
| Labels | Plus/minus strand donor/acceptor positions and motifs; independent isoforms; encoding round trips | `test_minus_strand_labeling.py`, `test_encode_decode_roundtrip.py`, `test_dataset_boundaries.py` |
| Data creation | All advertised biotypes, GFF/GTF, transcript/gene type, valid strands/exons/coordinates, empty splits and invalid ratios | `tests/integration/test_create_data_pipeline.py`; `test_annotation_and_merge_edges.py`, `test_dataset_boundaries.py` |
| Data cache | Source-content fingerprints, changed/corrupt cache rebuild, atomic failure preservation | `test_data_integrity.py`, `test_annotation_and_merge_edges.py` |
| Paralogy | Query-span coverage with reference deletions, requested filtering on test/validation, empty split and index failure | `tests/unit/test_paralogs.py`; `test_data_integrity.py`, `test_annotation_and_merge_edges.py` |
| HDF5 schema | Paired noncontiguous canonical keys, metadata, binary encodings, observed/padding labels, invalid shape/context, every shard | `test_scientific_boundaries.py`, `test_public_contract_guards.py`, `test_data_integrity.py` |
| Merge | Numeric shard ordering, values, context, train/test/validation, missing validation, failure preservation | `tests/integration/test_merge_data.py`; `test_annotation_and_merge_edges.py` |
| Loss | Independent categorical focal formula and gradients, scalar/vector alpha, gamma, invalid/nonfinite values, padding invariance | `test_correct_scientific_defaults.py`, `test_public_contract_guards.py`, `test_scientific_boundaries.py` |
| Training | Seed-before-initialization, one-window/partial batches, full selected split, explicit validation, real epoch/checkpoints | `tests/unit/test_train_init.py`, `test_train_loop.py`; `tests/integration/test_train_smoke.py`; `test_correct_scientific_defaults.py` |
| Schedulers | Hand-computed epoch decay and fractional cosine values; optimizer/epoch boundaries | `test_scientific_boundaries.py`; `tests/unit/test_train_base_utils.py`, `test_train_loop.py` |
| Metrics | Fixed background/acceptor/donor labels, absent positives/classes, complete observed counts | `test_correct_scientific_defaults.py`; `tests/unit/test_train_base_utils_extra.py` |
| Transfer | Head-only and selected residual units, unchanged frozen BatchNorm buffers, strict students/teachers, partial opt-in, mitigations | `tests/unit/test_transfer_freeze.py`, `test_transfer_forgetting.py`; `test_scientific_boundaries.py`; `tests/integration/test_transfer_smoke.py` |
| Checkpoints | Missing/corrupt/nonfinite/wrong-context/partial ensemble, `.pt`/`.pth`, failed atomic replacement | `test_checkpoint_failures.py`, `test_portable_calibration.py`, `test_public_contract_guards.py` |
| Calibration optimization | Independent chunked/full NLL gradient, requested epochs/early stop, best post-update state, identity baseline | `test_portable_calibration.py`; `tests/unit/test_temperature_scaling.py`, `test_temperature_scaling_extra.py` |
| Calibration resources | Single disk-backed logit cache, bounded reads/sample, exact NLL/ECE/Brier counts, exclusion of padding, cleanup on failure | `test_portable_calibration.py`, `test_scientific_boundaries.py`, `test_evaluation_resources.py` |
| Calibration curves | Endpoint/quantile observations, occupied bins, separate calibrated counts and aligned plot samples | `test_calibration_bin_alignment.py`; `tests/unit/test_calibrate_visualization.py` |
| Calibration helper API | Independent Brier values with absent classes, legacy positions, context exposure and labeled preview metrics | `test_calibration_public_helpers.py` |
| Portable calibration | Version/context/class order/finite float temperatures, no nested artifact, CPU serialization, both actual inference loaders | `test_portable_calibration.py`, `test_public_contract_guards.py`; distribution smoke and tutorial |
| Prediction routes | Text/HDF5 × turbo/predict-all, ensemble, annotation extraction, actual identical rows/probabilities | `tests/integration/test_predict_modes.py`, `test_predict_mode_equivalence.py`, `test_split_prediction_equivalence.py` |
| Prediction ownership | Split/unsplit, plus/minus, accession contigs, padding exclusion, unique source positions, invalid spans | `test_prediction_ownership.py`, `test_data_integrity.py`; real eight-case split/unsplit integration comparison |
| Variant semantics | SNV/indel/multiallelic and MNV/delins boundaries, masks/strands, known DS/DP oracle, sequential/batch equivalence | `test_variant_event_semantics.py`, `test_variant_mnv_boundary.py`, `test_variant_batched_equivalence.py`; real variant integration |
| Contig aliases | Existing exact names first, optional existing `chr` alias, missing names and mixed accession/chromosome references | `test_mixed_contig_scoring.py` |
| Variant publication | Scoped FP32/restored flags, stdout purity, BGZF integrity, finite DS/integer DP, old file survives failed publication | `test_variant_library_contract.py`; `tests/unit/test_variant_output_atomic.py` |
| Lifecycle | HDF5/FASTA/checkpoint/cache failures close resources; libraries raise exceptions | `test_evaluation_resources.py`, `test_data_integrity.py`, `test_public_contract_guards.py` |
| Original Keras parity | Original weights, supported shared variants, reference outputs and masks | Five `keras` cases; final frozen job `31628903` passed with no skips |
| CUDA | Four contexts, CPU/batch equivalence, finite training gradients, CUDA calibration/portable state, both strands/masks | Nine `gpu` cases in `tests/equivalence/test_gpu_equivalence.py`; job `31628943` dependency-pending |
| Installed artifacts | Both archive formats, installed module paths, CLI, annotation package data, exact raw/calibrated inference | `ci/smoke_distribution.py`; wheel and sdist passed from `/tmp` |
| Documentation | Generated parser/API reference, old paths/anchors, build warnings, local references, browser search/themes/mobile | Sphinx, `docs/check_links.py`, `ci/check_docs_browser.py`; all executed local checks passed |
| Research scoring | Denominators, score alignment, missing/placeholder values, streaming summaries, external-source plans | `tests/unit/test_full_snv_concordance.py`, `test_full_snv_concordance_depth.py`, `test_external_evaluate.py`, `test_external_score_plan.py` |
| Research reports | Concordance studies/depth passes, annotations, provenance, report contracts, bounded scientific figures | `test_concordance_study.py`, `test_concordance_depth_pass.py`, `test_concise_report.py`, `test_validation_invariants.py` |
| Campaign recovery | Transient I/O/readback, stale redundant throttle, ambiguous acknowledgement, allocation accounting, no live service calls | `tests/unit/test_campaign_automation.py`, `test_campaign_reliability.py`; new helpers staged separately from signed r13 |

Bare `test_*.py` names above are under `tests/regression/` unless the row explicitly
names `tests/unit/` or `tests/integration/`. Research/campaign rows name unit files.

## Scope that remains explicit

The source inventory records 217 maintained definitions, including classes and
private helpers. All public maintained definitions have docstrings; generated
API signatures point to source. Per-function CPU execution is in
`function-coverage.json`, and raw per-file reports retain missing lines/branches.
Optional Keras entrypoints can be unexecuted in CPU coverage and are checked
separately. Coverage exclusions are recorded in `.coveragerc`.

Research/campaign coverage is 62.5736% statements and 56.8140% branches, below the
maintained-package coverage. External-data workflows and historical standalone
scripts require their own fixtures/resources before equivalent claims are possible.
No whole-genome rerun, published-model retraining, broad biological benchmark or
live controller mutation is part of this audit. macOS and additional Python
versions require successful CI evidence; configuration is insufficient.
