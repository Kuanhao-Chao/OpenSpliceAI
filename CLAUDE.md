# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenSpliceAI is a PyTorch reimplementation and extension of SpliceAI (Jaganathan et al., 2019) for splice-site
prediction. It is distributed as a single `openspliceai` console command with six subcommands that form an
end-to-end pipeline: **create-data → train / transfer → calibrate → predict / variant**.

## Install & run

```bash
pip install .                 # editable dev install: pip install -e .
# or from PyPI: pip install openspliceai ; or via conda-recipe/meta.yaml

openspliceai <subcommand> ...  # console entry point -> openspliceai/openspliceai.py:main
```

Requires Python ≥3.9 and PyTorch ≥2.2.1. `mappy` (minimap2) is only used for `--remove-paralogs` in create-data;
`tensorflow`/`keras` is only needed when scoring with original Keras SpliceAI models in the `variant` subcommand
(`--model-type keras`). The `test` subcommand and `openspliceai/test/test.py` are commented out / disabled.

### Testing & dev tooling
There **is** a pytest suite under `tests/` (`tests/{unit,regression,integration}`, with `tests/conftest.py`
fixtures and `tests/fixtures/synthetic.py` builders), plus `ruff` (`ruff.toml`), `pre-commit`
(`.pre-commit-config.yaml`), and coverage config (`.coveragerc`). Install dev deps with `pip install -e '.[dev]'`
(or `'.[test]'` for just pytest). Tests are **CPU-only**; GPU/keras-only tests are marked and auto-skip when those
backends are absent. Markers: `unit`-implicit, `integration`, `slow`, `gpu`, `keras`.

> **Interpreter gotcha (this machine):** the default `python` has a broken numpy/torch ABI and is missing deps. Run
> everything with `/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python` (numpy 1.26, torch 2.2.1, full bio-stack).
> TensorFlow **is** installed there, so `keras` tests run.

```bash
ENV=/home/kchao10/miniconda3/envs/pytorch_cuda/bin
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest -m "not integration and not slow" -q   # fast unit suite (~secs)
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest -q                                     # full suite incl. e2e
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest --cov=openspliceai --cov-report=term-missing -q
$ENV/ruff check openspliceai tests          # lint (clean)
```
`KNOWN_ISSUES.md` documents deferred behavior-changing issues and two audit *false positives* locked by regression
tests. There is no CI workflow (lint/tests run locally / via pre-commit).

## Architecture

### CLI dispatch
`openspliceai/openspliceai.py` defines all argparse subparsers (`parse_args_*`) and `main()` dispatches to each
subcommand's package. `create-data` runs **two stages back-to-back** in one invocation: `create_datafile.create_datafile`
then `create_dataset.create_dataset` (+ optional `verify_h5_file`).

### The model (one definition, reused everywhere)
`openspliceai/train_base/openspliceai.py` defines the `SpliceAI` `nn.Module` — a deep residual dilated 1-D CNN:
`ResidualUnit` (BatchNorm → LeakyReLU(0.1) → Conv1d, twice, with a residual add), a `Skip` connection inserted after
every 4 residual units, `Cropping1D` to trim context, and a final 1×1 conv to **3 channels** (non-splice / acceptor /
donor) with softmax. Input is **4-channel one-hot DNA** (A,C,G,T). `train`, `transfer`, `predict`, `variant`, and
`calibrate` all build this same class with hyperparameters keyed off `--flanking-size`.

**Flanking size is the central parameter and must match between the dataset, the model, and the checkpoint.** Only
`{80, 400, 2000, 10000}` are valid. Each maps to a fixed `(W=conv-window, AR=atrous/dilation-rate)` schedule of
4/8/12/16 residual units, and the resulting context length `CL = 2·Σ(AR·(W−1))` equals the flanking size exactly
(e.g. 10000 flank → 16 units → CL=10000). This `(L=32, W, AR, BATCH_SIZE)` table is **duplicated** in
`train/train.py`, `transfer/transfer.py`, `predict/predict.py`, and `variant/utils.py` — keep them in sync when changing.

Global constants live in `openspliceai/constants.py`: `CL_max=10000` (max total context, padded onto both ends),
`SL=5000` (model output/prediction window length). Inputs to the model are windows of length `SL + CL_max`.

> **Gotcha:** `N_GPUS=2` is hardcoded in the hyperparameter tables but the code does **not** wrap the model in
> `DataParallel` — it runs on a single device. `N_GPUS` only scales `BATCH_SIZE` (e.g. 18×2) and drives the
> remainder-trimming in `clip_datapoints`. `setup_device()` picks cuda → mps (macOS) → cpu.

### Data representation
- **One-hot maps** (`create_data/utils.py`): nucleotides A,C,G,T → ints 1,2,3,4, N/padding → 0 via `IN_MAP`; labels
  0,1,2 (no-splice, acceptor, donor) via `OUT_MAP`.
- **`datafile_*.h5`** (stage 1 output): per-gene string datasets `SEQ`, `LABEL`, `STRAND`, `TX_START`, `TX_END`.
  Genes are extracted from the FASTA, **reverse-complemented on the minus strand**, and labels mark donor=2 /
  acceptor=1 positions (optionally gated to canonical GT-AG / GC-AG / AT-AC motifs via `--canonical-only`).
- **`dataset_*.h5`** (stage 2 output): one-hot encoded, windowed tensors stored as chunked datasets `X0,X1,…` and
  `Y0,Y1,…` (100 genes per chunk). Training reads these shards and **transposes to `(N, channels, length)`** before
  the model. Splits produced: `train` / `validation` / `test` (train is further split 90:10 into train/val).
- **Chromosome split** (`split_chromosomes`): `--split-method human` uses SpliceAI's fixed train/test chromosome
  assignment (test = chr1,3,5,7,9); `random` splits by length to `--split-ratio`.

### Training / transfer (`train/`, `transfer/`, shared `train_base/utils.py`)
Both build the model, then call the shared `train_model` loop. Data is loaded shard-by-shard from HDF5
(`load_data_from_shard`), and every batch is cropped by `clip_datapoints` to the model's required context. Loss is
either `categorical_crossentropy_2d` or `focal_loss` (`--loss`). Optimizer is AdamW (lr 1e-3 train / 1e-4 transfer);
schedulers are `MultiStepLR` or `CosineAnnealingWarmRestarts`. Checkpoints are saved as **plain `state_dict`s**
(`model_{epoch}.pt`, `model_best.pt`). Metrics (top-k accuracy + AUPRC for donor & acceptor, per-class
precision/recall/F1) are appended to per-metric `.txt` files. Output layout:
`{output_dir}/SpliceAI_{project}_{flank}_{exp}_rs{seed}/{exp}/{models,LOG/{TRAIN,VAL,TEST}}/`.
`transfer` additionally loads a pretrained checkpoint (filtering size-mismatched keys) and can freeze all but the last
`--unfreeze` residual units (`--unfreeze-all` is the default).

### Predict (`predict/predict.py`)
Multi-stage, designed to scale to whole genomes: extract sequences (optionally just gene regions when `-a/--annotation`
GFF is given) → split FASTA entries longer than `--split-threshold` (default 1.5 Mb) **with flanking overlap so
predictions are seamless** → one-hot encode to `dataset.h5`/`.pt` → load model(s) → infer → write `donor_predictions.bed`
and `acceptor_predictions.bed`. Two modes: default **turbo** (`predict_and_write`, streamed, no intermediate file) vs
`--predict-all` (writes `predict.h5` then `generate_bed`). **If `--model` is a directory, all checkpoints in it are
ensembled by averaging predictions.** Checkpoint `.pt` files are state_dicts loaded into a freshly-built `SpliceAI`;
a flanking-size mismatch surfaces as a size-mismatch warning.

### Variant (`variant/variant.py`, `variant/utils.py`)
Annotates a VCF with splicing delta scores. The `Annotator` loads the reference genome (pyfaidx), a gene-annotation
table (`grch37`/`grch38` builtins or a custom TSV), and SpliceAI model(s) — PyTorch (default) or Keras. `get_delta_scores`
computes, within `--distance` of each variant, delta **scores** (DS) and **positions** (DP) for acceptor gain/loss and
donor gain/loss, written to the `OpenSpliceAI` INFO field with format
`ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`. Reads stdin / writes stdout by default.

### Calibrate (`calibrate/`)
Post-hoc **temperature scaling** (`ModelWithTemperature`) of a trained model. Fits a temperature on the validation
set, reports ECE/NLL and Brier scores, writes calibration-curve plots, and saves `temperature.pt`/`.txt` and a full
`calibrated_model.pt`.

## Pretrained models (`models/`)
`models/openspliceai-{mane,mouse,zebrafish,arabidopsis,honeybee}/{80,400,2000,10000}nt/model_*nt_rs{10..14}.pt` — five
random-seed checkpoints per (species, flanking) combo, intended to be ensembled (point `predict`/`variant` at the
directory). `mane` is the human GRCh38/MANE model. `models/spliceai/` holds the original Keras SpliceAI weights;
`models/prev/` is older/gitignored.

## Conventions & gotchas
- The installed pipeline lives entirely in the subcommand packages (`create_data`, `train`, `transfer`, `calibrate`,
  `predict`, `variant`, `train_base`). **`openspliceai/scripts/` is legacy research/experimentation code** (much of it
  gitignored) and is not part of the packaged flow — don't treat it as the source of truth.
- `*.h5`, `*.bed`, `*.fa`, `*.db`, `*.log`, `*.txt`, `results/`, and `/data/` are gitignored; generated `gff_to_tsv`
  databases (`*.gff_db`) and `examples/data/*` genomes are large local artifacts, not committed.
- The hyperparameter table duplication (see Architecture) is the most common source of subtle bugs — a change in one
  subcommand's `(W, AR, BATCH_SIZE)` must be mirrored in the others.
- Full user docs (Sphinx) are in `docs/source/` and hosted at https://ccb.jhu.edu/openspliceai/.
