<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/main/logo/logo_white.png">
  <img alt="OpenSpliceAI" src="https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/main/logo/logo_black.png" width="75%">
</picture>
</p>

OpenSpliceAI trains and runs splice-site models in PyTorch and annotates genetic
variants with splice gain/loss scores. Released models cover human, mouse,
Arabidopsis, honeybee and zebrafish, with contexts of 80, 400, 2,000 and 10,000 bases.

[Documentation](https://khchao.com/OpenSpliceAI/) ·
[Quick start](https://khchao.com/OpenSpliceAI/content/quick_start_guide/index.html) ·
[Model downloads](https://khchao.com/OpenSpliceAI/content/pretrained_models/index.html) ·
[GPLv3 license](LICENSE)

This checkout stages **0.1.0.dev0**. Read the
[migration notes](docs/source/content/migration.rst) before reproducing an older
training run: focal loss, scheduler timing, seeds, partial batches, transfer
freezing, calibration and prediction output have corrected behavior. Published
releases and existing signed production campaigns use their recorded source.

## Install

Use Python 3.9 or newer in a separate environment. Install a PyTorch wheel for
your CPU/GPU, then install OpenSpliceAI and check the scientific stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openspliceai
python -m pip check
openspliceai --help
```

For this development checkout, use `python -m pip install -e '.[dev]'`.
NumPy 2 requires compatible h5py, SciPy, scikit-learn and Matplotlib wheels;
[installation details](https://khchao.com/OpenSpliceAI/content/installation.html)
cover native dependencies and the separate optional TensorFlow/Keras profile.
PyTorch selects CUDA, then available macOS MPS, then CPU. On Linux,
`CUDA_VISIBLE_DEVICES=''` forces CPU.

## Commands

| Command | Input | Result |
|---|---|---|
| `create-data` | Assembly FASTA and GFF/GTF | Oriented sequence/label files and train/validation/test HDF5 shards |
| `train` | Three distinct HDF5 splits | Raw model checkpoints and observed-position metrics |
| `transfer` | Raw checkpoint and new splits | Fine-tuned head/residual units, with optional rehearsal/distillation |
| `calibrate` | Raw checkpoint, validation and test splits | Portable calibrated checkpoint and complete-split metrics |
| `predict` | FASTA and model; optional annotation | Acceptor/donor BED6 with strand and owned coordinates |
| `variant` | VCF, reference, transcript annotation and model | Validated VCF/BGZF with `OpenSpliceAI` DS/DP annotations |

Each command has complete `--help` and a
[worked example](https://khchao.com/OpenSpliceAI/content/quick_start_guide/index.html).
Variant model context must match the checkpoint. Original SpliceAI `.h5` models
use `--model-type keras --flanking-size 10000`; PyTorch requires its own checkpoint.
Raw checkpoints remain supported. Versioned calibrated artifacts work in both
prediction and variant scoring.

From a clone, run all six workflows on synthetic data without downloading a genome:

```bash
python examples/tutorial/run.py --output-dir /tmp/osai-tutorial
```

The driver checks output schemas and calibrated inference. Its small synthetic
model verifies mechanics and carries no biological accuracy claim.

## Develop and verify

```bash
python -m pip install -e '.[dev]'
make lint test
make test-cpu
make coverage-branch
python -m pip install -r docs/requirements.txt
make -C docs html SPHINXOPTS='-W --keep-going'
python docs/check_links.py docs/build/html
```

Coverage gates require 95% statements and 90% branches separately. Optional
`make test-keras` and `make test-gpu` require real dependencies, weights and devices;
missing backends are missing evidence. CI configures Python 3.9–3.14, Linux/macOS,
a minimum scientific stack, distribution installation and original Keras parity.
Configured jobs are distinguished from executed results in the
[audit record](docs/development/comprehensive-audit.md).

Installed code lives in `openspliceai/`; tests in `tests/`; scientific/campaign tools
in `validation/`. Historical scripts are inventoried separately and depend on
external datasets. See the [repository walkthrough](docs/source/content/repository.rst)
and [known limitations](KNOWN_ISSUES.md).

## Cite

Kuan-Hao Chao, Alan Mao, Anqi Liu, Mihaela Pertea and Steven L. Salzberg.
[OpenSpliceAI provides an efficient modular implementation of SpliceAI enabling easy
retraining across nonhuman species](https://doi.org/10.7554/eLife.107454.3).
Also cite [Jaganathan et al., SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015)
when using its architecture or original weights.
