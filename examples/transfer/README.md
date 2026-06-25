# OpenSpliceAI `transfer` examples

The `transfer` subcommand **fine-tunes a pretrained OpenSpliceAI model** on new data — a new species,
or a narrow dataset — instead of training from scratch. This folder walks through the full workflow,
from building datasets to a baseline fine-tune to fine-tuning with **catastrophic-forgetting
mitigation**.

All scripts resolve the repo root automatically and run from anywhere. They use the **80 nt** model so
the demo is fast (the 80 nt checkpoints are ~400 KB each); for production use `--flanking-size 10000`
with the `10000nt/` models.

## What's here

| Script | Purpose |
|---|---|
| [`create-data_bee_example.sh`](./create-data_bee_example.sh) | Build `dataset_{train,validation,test}.h5` from a genome + annotation (honey bee sample). |
| [`transfer_example.sh`](./transfer_example.sh) | **Baseline** fine-tune of the pretrained MANE model on the new data. |
| [`transfer_forgetting_cmd.sh`](./transfer_forgetting_cmd.sh) | Fine-tune **with the forgetting-mitigation flags** (see below). |

## Prerequisites

- OpenSpliceAI installed (`pip install -e .` from the repo root, or the devel build).
- A pretrained checkpoint — the repo ships `models/openspliceai-mane/80nt/model_80nt_rs10.pt`.
- An example genome + annotation — `examples/data/honeybee/HAv3.1_genomic_10_sample.{fna,gff}` (small).

## Step 1 — Build the datasets

```bash
bash examples/transfer/create-data_bee_example.sh
```

Produces `examples/transfer/results/dataset_{train,validation,test}.h5` (one-hot encoded sequences +
labels) at `--flanking-size 80`. The two following scripts read those files.

## Step 2 — Baseline transfer

```bash
bash examples/transfer/transfer_example.sh
```

Fine-tunes `model_80nt_rs10.pt` on the new data and writes checkpoints + logs (see **Outputs**). This
is the plain fine-tune with **no** forgetting mitigation — a useful reference point.

## Step 3 — Transfer with catastrophic-forgetting mitigation

When you fine-tune on a **narrow** dataset (a single locus, an assay covering a few genes), a fully
unfrozen model can **catastrophically forget** canonical splice sites elsewhere in the genome —
donor/acceptor scores at well-annotated junctions collapse toward zero even though those sites are
unchanged. `transfer` has four optional, default-off controls (added in v0.0.8.dev0):

| Flag | What it does |
|---|---|
| `--weight-decay 0` | AdamW's default `0.01` decays every weight toward **zero** each step, eroding pretrained splice features. Set `0` to disable. |
| `--l2sp 0.1` | Instead of decaying toward zero, penalize drift away from the **pretrained** weights (active only with a distillation teacher). |
| `--unfreeze 2` | Freeze most layers (train only the last 2 residual units), preserving general motif detectors. Prefer **progressive unfreezing** on narrow data (see Tips). |
| `--rehearsal-dataset` / `--rehearsal-shards` | **Changes training**: interleave real genomic shards (with true labels) into the batch stream — experience replay. |
| `--distill-weight` / `--distill-shards` / `--distill-teacher` | Knowledge distillation (Learning without Forgetting): add `λ · cross_entropy(teacher_soft_targets, student)` on genomic anchor windows scored by a frozen copy of the pretrained model. **No genomic labels needed.** |
| `--genomic-eval-dataset` | **Pure measurement** (no effect on training): log a per-epoch **forgetting curve** (donor/acceptor AUPRC + top-k) under `LOG/GENOMIC/`. |

```bash
bash examples/transfer/transfer_forgetting_cmd.sh
```

The demo reuses the test set as the genomic source so it runs out of the box. **In a real run**, point
`--genomic-eval-dataset` / `--rehearsal-dataset` / `--distill-shards` at **held-out shards of the
distribution you don't want to forget** (e.g. MANE human test chromosomes), created with `create-data`
at the **same flanking size** as your fine-tuning data (per-batch context clipping assumes a fixed
`SL + CL_max` width).

## Step 4 — Inspect the outputs

Each run writes to `--output-dir` under a versioned folder:

```
results[_forgetting]/SpliceAI_<project>_<flank>_<exp>_rs<seed>/<exp>/
├── models/
│   ├── model_<epoch>.pt        # one checkpoint per epoch
│   └── model_best.pt           # best validation loss
└── LOG/
    ├── TRAIN/  VAL/  TEST/      # loss, AUPRC, top-k, precision/recall/F1 per epoch
    └── GENOMIC/                 # only with --genomic-eval-dataset: the forgetting curve
        ├── donor_auprc.txt      # genomic donor AUPRC per epoch  (retention)
        └── acceptor_auprc.txt
```

To choose a checkpoint, plot the **GENOMIC** AUPRC (how much canonical-site accuracy is retained)
against the **TEST** AUPRC (in-distribution gain) across epochs, and pick the `model_<epoch>.pt` on the
best gain-vs-retention trade-off. Sweep `--distill-weight` / `--rehearsal-shards` to move along that
curve.

## Tips

- **Progressive unfreezing.** Rather than `--unfreeze-all`, start at `--unfreeze 2` and increase
  gradually across successive transfer rounds — rerun with `--unfreeze 4`, then `8`, each round
  resuming from the previous round's `model_best.pt` via `--pretrained-model` — stopping as soon as the
  forgetting curve dips.
- **Same flanking size everywhere.** The genomic HDF5s for rehearsal / distillation / eval must be
  built at the same `--flanking-size` as your training data.
- **Stand up the gauge first.** Add `--genomic-eval-dataset` before anything else so every later
  experiment is measurable.

## See also

- [`transfer` documentation](../../docs/source/content/openspliceai_transfer.rst) — full options and the
  "Mitigating catastrophic forgetting" section.
- [`create-data` documentation](../../docs/source/content/openspliceai_create-data.rst).
