# Feature verification matrix

This inventory covers the packaged commands and current validation helpers.
Named files contain the executable evidence; final environment results and
remaining limitations are recorded in `repository-audit.md`.

| Feature | Evidence | Scope |
|---|---|---|
| CLI dispatch, help, required arguments, lazy imports | `test_cli_argparse.py`, `test_cli_dispatch.py`, `test_lazy_cli_imports.py`; isolated wheel help checks | Six commands; help needs no heavy backend |
| Architecture / probabilities | `test_model_forward.py`, `test_validation_invariants.py`, `test_hyperparam_table_sync.py` | Four context schedules, crop length, normalized probabilities, complete inference checkpoints |
| Encoding / strand labels | `test_encode_decode_roundtrip.py`, `test_minus_strand_labeling.py`, `test_create_datafile.py`, `test_create_data_utils.py` | Canonical motifs, both strands, ambiguity, padded boundaries and chromosome splits |
| Data creation and merging | `test_create_data_pipeline.py`, `test_merge_data.py`, `test_paralogs.py` | Real HDF5 round trips and minimap2 removal of identical synthetic sequences |
| Training | `test_train_base_utils.py`, `test_train_loop.py`, `test_train_smoke.py` | Loss/metrics, finite optimizer steps, scheduler choices, early stopping and checkpoints |
| Transfer | `test_transfer_freeze.py`, `test_transfer_forgetting.py`, `test_transfer_smoke.py` | Gradient freezing, weight decay, rehearsal, teacher/distillation, L2-SP, genomic evaluation, default-off compatibility |
| Calibration | `test_temperature_scaling.py`, `test_temperature_scaling_extra.py`, `test_calibration_bin_alignment.py`, `test_calibrate_smoke.py` | Class scaling, restored temperature, metrics/plots, occupied counts, endpoint/tie cases, separate scaled counts |
| Evaluation and cleanup | `test_evaluation_resources.py`, `test_invalid_training_inputs.py` | No gradients in validation, unchanged BN state in calibration, clear empty-input failures, file closure on errors |
| Prediction | `test_predict_utils.py`, `test_predict_pipeline_units.py`, `test_predict_modes.py`, `test_predict_mode_equivalence.py` | BED coordinates/thresholds, annotated FASTA, split overlap and multi-window input; all four storage/execution modes agree |
| Variant events | `test_variant_event_semantics.py`, `test_variant_delta.py`, `test_variant_utils_extra.py` | Independent AG/AL/DG/DL score and DP expectations, both strands, masks, supported/unsupported alleles and reference mismatches |
| Variant batching / MNV | `test_variant_batched_equivalence.py`, `test_variant_mnv_boundary.py`, `test_variant_batched_cli.py` | Sequential/batched agreement, indel/delins/MNV realignment boundary and CLI wiring |
| Checkpoint failure | `test_checkpoint_failures.py`, `test_keras_loader_contract.py` | Missing/corrupt/incompatible members fail; `.pth` works; Keras loader contracts use a fake backend |
| Atomic VCF output | `test_variant_output_atomic.py` | Successful same-directory replacement, structural validation, interrupted/failed writes preserve existing output |
| Original SpliceAI backend | `test_keras_equivalence.py` | Real optional Keras weights; formatted DS/DP equality; backend availability and results reported separately |
| CUDA backend | `test_gpu_equivalence.py` | Optional CPU/CUDA and single/batched numerical comparison with explicit tolerance |
| Research / report calculations | `test_full_snv_concordance*.py`, `test_concordance*.py`, `test_external*.py`, `test_concise*.py`, `test_campaign_automation.py` | Counts/denominators, thresholding, provenance, VCF alignment, release gates and simulated scheduling; no live job submission |
| Distribution and documentation | `tests.yml`, `docs.yml`; local installation/build logs | Wheel/sdist package data, CLI outside source, six runtime imports, Sphinx warnings-as-errors, local link checks |

Synthetic tests establish implementation behavior on controlled inputs. They do
not validate all real-world annotations, large-genome memory/runtime behavior or
biological accuracy. Test seeding does not prove complete CLI seed reproducibility.
The preserved scientific limitations are described in `KNOWN_ISSUES.md`.
