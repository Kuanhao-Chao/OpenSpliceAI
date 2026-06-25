# OpenSpliceAI `transfer` examples

The `transfer` subcommand **fine-tunes a pretrained OpenSpliceAI model on new data** instead of
training from scratch — it loads an existing checkpoint and continues training on your dataset, which
converges far faster and works even when you have relatively little data. Two common scenarios:

- **A new species** — adapt the human `OSAI_MANE` model to mouse, zebrafish, honey bee, etc.
- **A narrow / specialized assay** — fine-tune on data that covers only a few genes (e.g. an MPSA
  minigene library), where you also have to actively guard against *catastrophic forgetting* (below).

This folder walks through the whole path:

```
create-data  →  transfer  →  inspect                              (baseline)
create-data  →  transfer (+ forgetting controls)  →  inspect      (narrow-data, forgetting-aware)
```

Everything here uses the **80 nt** model so the demo runs in minutes (the 80 nt checkpoints are
~400 KB). For production, swap in `--flanking-size 10000` and the `10000nt/` models — the workflow is
identical.

---

## At a glance

| Script | Step | What it does |
|---|---|---|
| [`create-data_bee_example.sh`](./create-data_bee_example.sh) | 1 | Build the **target** datasets (`dataset_{train,validation,test}.h5`) from a genome + annotation (honey bee sample). |
| [`create-data_genomic_eval.sh`](./create-data_genomic_eval.sh) | 1b | *(optional, recommended)* Build a **human held-out genomic** dataset (chr22) — the distribution you don't want to forget. |
| [`transfer_example.sh`](./transfer_example.sh) | 2 | **Baseline** fine-tune of the pretrained MANE model — no forgetting mitigation. |
| [`transfer_forgetting_cmd.sh`](./transfer_forgetting_cmd.sh) | 3 | Fine-tune **with the catastrophic-forgetting controls** turned on. |

Run them in order; Step 4 is reading the outputs.

---

## Prerequisites

- **OpenSpliceAI installed** — `pip install -e .` from the repo root (or the devel build). Verify with
  `openspliceai --help` (you should see the six subcommands and the version banner).
- **A pretrained checkpoint** — the repo ships `models/openspliceai-mane/80nt/model_80nt_rs10.pt`
  (and `rs10–rs14` for the five-seed ensemble).
- **Example data** — `examples/data/honeybee/HAv3.1_genomic_10_sample.{fna,gff}` (target species) and
  `examples/data/chr22.{fa,gff}` (human genomic set for Step 1b).

---

## Step 1 — Build the target datasets

```bash
bash examples/transfer/create-data_bee_example.sh
```

`create-data` converts a genome FASTA + annotation GFF into one-hot-encoded, windowed HDF5 datasets.
This writes `examples/transfer/results/dataset_{train,validation,test}.h5` at `--flanking-size 80`.
Steps 2 and 3 read these. (The `--flanking-size` here **must** match the model you fine-tune.)

### Step 1b — Build a human held-out genomic set *(recommended for Step 3)*

```bash
bash examples/transfer/create-data_genomic_eval.sh
```

Because we're adapting a **human** model to **honey bee**, "forgetting" means losing the model's grip on
**human** splice sites. So the right yardstick — and the right material to rehearse / distill on — is a
held-out *human* genomic set, not the honey-bee test set. This builds one from chr22 into
`examples/transfer/results_genomic/` (use `dataset_test.h5` as the gauge, `dataset_train.h5` to
rehearse / distill on). It's optional: `transfer_forgetting_cmd.sh` falls back to reusing the target
test set so it runs out of the box, but the genomic set makes the forgetting numbers *mean* what you
think.

---

## Step 2 — Baseline transfer

```bash
bash examples/transfer/transfer_example.sh
```

A plain fine-tune of `model_80nt_rs10.pt` on the target data, with **no** forgetting mitigation. It
writes checkpoints + logs under `examples/transfer/results/` (see
[Reading the results](#step-4--reading-the-results)). Keep this as your reference point: it shows both
the in-distribution gain *and*, if you also build the genomic set, how much the unmitigated run forgets.

---

## Step 3 — Transfer with catastrophic-forgetting mitigation

```bash
bash examples/transfer/transfer_forgetting_cmd.sh
```

When you fine-tune on a **narrow** dataset (a specialized assay, a handful of genes), a fully unfrozen
model can **catastrophically forget** canonical splice sites across the rest of the genome:
donor/acceptor scores at well-annotated junctions collapse toward zero even though those sites haven't
changed. `transfer` gives you four optional, default-off controls (added in v0.0.8.dev0) to *measure*
and *counteract* this. The next section explains each; the script turns them all on at once.

---

## Understanding catastrophic forgetting & the four levers

**Why it happens, briefly.** With every layer unfrozen, AdamW's weight decay slowly pulls weights
toward zero, and your fine-tuning loss only ever reinforces the new task — so the pretrained,
genome-wide splice detectors aren't reinforced and erode. (Training a softmax toward soft / partial
labels can also lower confidence across the shared output head, dragging scores down everywhere.) The
fixes either **slow the drift** or **re-inject genomic signal** so the old ability is preserved.

### 1. Stop decaying toward zero — `--weight-decay`, `--l2sp`

`--weight-decay 0` turns off AdamW's default `0.01` decay (which shrinks every weight toward zero each
step). `--l2sp <μ>` goes further: instead of pulling weights toward zero, it penalizes drift away from
the **pretrained** weights — "stay near where you started." (`--l2sp` is only active alongside a
distillation teacher, whose weights are the reference.)

### 2. Reduce plasticity — progressive unfreezing (`--unfreeze`)

Rather than `--unfreeze-all`, train only the last few residual units so the general motif detectors are
preserved. Best practice on narrow data is **progressive unfreezing**: start at `--unfreeze 2`, then
open up gradually — rerun with `--unfreeze 4`, then `8`, **each round resuming from the previous round's
`model_best.pt` via `--pretrained-model`** — and stop as soon as the forgetting curve (below) starts to
dip.

### 3. Rehearsal / experience replay — `--rehearsal-dataset`, `--rehearsal-shards`

Interleave **real, labeled genomic shards** into the training stream. Each epoch then draws some
batches from your target data and some from genomic data (with their true labels, through the same
loss), so a fraction of every epoch's gradient is "keep predicting genomic splice sites correctly."
`--rehearsal-shards N` sets the mix ratio (`-1` = all available shards).

### 4. Knowledge distillation / Learning without Forgetting — `--distill-weight`, `--distill-shards`, `--distill-teacher`

Keep a **frozen copy of the pretrained model** as a *teacher*. Each step, on top of the target loss,
run both the teacher and your model on a batch of genomic windows and add
`λ · cross_entropy(teacher_softmax.detach(), student_softmax)`, pulling your model's genomic
predictions toward the teacher's. The key property: the teacher's outputs *are* the targets, so
**`--distill-shards` needs no labels** — just genomic sequence. `--distill-weight λ` is the strength
(`0` disables); `--distill-teacher` defaults to `--pretrained-model`.

### Rehearsal vs. distillation — which to use

Both re-inject genomic signal; they differ in what they need and what they cost:

|  | Rehearsal (`--rehearsal-dataset`) | Distillation (`--distill-weight`) |
|---|---|---|
| Re-injects signal via | real **labeled** genomic examples in the batch | matching a frozen **teacher's** outputs |
| Needs genomic **labels**? | **Yes** | **No** (teacher provides the targets) |
| Needs | a labeled genomic `dataset_*.h5` | only the pretrained checkpoint + genomic sequence |
| Cost | one model | two models per step (teacher + student) → slower, more memory |
| Ceiling | can *exceed* the original (real labels) | capped at the teacher's genomic quality |

They are **not** mutually exclusive — you can run both. If you only have continuous / soft labels for
your assay (so reconciling them with the original binary genomic labels is awkward), **distillation is
the lower-friction first try** because it sidesteps labels entirely.

### The gauge — `--genomic-eval-dataset`

This one **changes nothing about training** — it's pure measurement. After every epoch the model is run
(in eval mode, no gradient) over a held-out genomic set, and donor/acceptor AUPRC + top-k are appended
to `LOG/GENOMIC/`. That gives you a **forgetting curve**: one number per epoch showing how much
canonical-site accuracy is retained. Stand this up *first* — it turns "is it forgetting?" into something
you can watch, and it's what lets you pick the right checkpoint. It's distinct from `--test-dataset`,
which is still your (in-distribution) target data.

---

## Step 4 — Reading the results

Each run writes to `--output-dir` under a versioned folder:

```
results[_forgetting]/SpliceAI_<project>_<flank>_<exp>_rs<seed>/<exp>/
├── models/
│   ├── model_<epoch>.pt        # one checkpoint per epoch
│   └── model_best.pt           # best validation loss
└── LOG/
    ├── TRAIN/  VAL/  TEST/      # loss, AUPRC, top-k, precision/recall/F1 per epoch (one line/epoch)
    └── GENOMIC/                 # only with --genomic-eval-dataset: the forgetting curve
        ├── donor_auprc.txt
        └── acceptor_auprc.txt
```

The TEST logs are your **gain** (in-distribution performance); the GENOMIC logs are your **retention**
(how much of the genome-wide ability survives). Read them together. A worked example — donor AUPRC per
epoch from a forgetting-aware run:

```
epoch   TEST (gain)   GENOMIC (retention)
  1        0.55            0.95
  2        0.71            0.93
  3        0.82            0.88     <-- best trade-off: big gain, little forgetting
  4        0.86            0.74
  5        0.88            0.49     <-- forgetting is now severe
```

Here you'd take `models/model_3.pt`: epochs 4–5 buy a little more in-distribution gain at a steep cost
in retention. Sweeping `--distill-weight` / `--rehearsal-shards` shifts the whole curve up or down on
the retention axis, so re-plot and re-pick after each sweep.

---

## Tuning

- **`--distill-weight λ`** — start with `0.3`; sweep `{0.1, 0.3, 1.0}`. Higher λ = more retention but
  more drag on learning the new task. Watch both curves and keep the knee.
- **`--rehearsal-shards N`** — the mix ratio. More genomic shards = stronger retention, fewer
  target-task gradients per epoch. Start at a handful and increase if the genomic curve still dips.
- **`--unfreeze`** — follow the progressive schedule (2 → 4 → 8), resuming from the prior round's
  `model_best.pt`. The more you unfreeze, the more you can fit the new task and the more you can forget —
  the genomic curve tells you when to stop.

---

## Troubleshooting / FAQ

- **"shapes don't line up" / context-clipping error.** Every genomic HDF5 used for
  `--genomic-eval-dataset` / `--rehearsal-dataset` / `--distill-shards` **must be built at the same
  `--flanking-size`** as your training data (they share the `SL + CL_max` input width). Rebuild the
  offending set with the matching flanking size.
- **Out of memory with distillation.** Distillation runs a second (teacher) model each step. Lower
  `--distill-batch-size` (it defaults to the training batch size) — it only affects the genomic anchor
  pass, not your target batches.
- **"I don't have labeled genomic data."** Use **distillation** — `--distill-shards` needs only genomic
  *sequence*; the frozen teacher supplies the targets.
- **The forgetting curve looks flat or noisy.** The per-epoch genomic eval subsamples 1000 "expressed"
  windows, seeded by `--random-seed`, so it's reproducible across epochs but a touch coarse — give it a
  large enough genomic set, and compare *trends* across epochs rather than single points.
- **Where do the best results come from?** Almost always: stand up the gauge, do a conservative pass
  (`--weight-decay 0`, `--unfreeze 2`), read the curve, then add rehearsal or distillation and unfreeze
  further only as the curve allows.

---

## Full flag reference (forgetting controls)

| Flag | Default | Effect |
|---|---|---|
| `--weight-decay` | `0.01` | AdamW weight decay; set `0` to stop decaying weights toward zero. |
| `--l2sp` | `0.0` | Penalize drift from the **pretrained** weights (needs a distillation teacher). |
| `--unfreeze N` | `1` | Train only the last `N` residual units (use progressively: 2 → 4 → 8). |
| `--unfreeze-all` | off | Train every layer (not recommended on narrow data). |
| `--genomic-eval-dataset` | none | Per-epoch genomic **forgetting curve** → `LOG/GENOMIC/` (measurement only). |
| `--rehearsal-dataset` | none | Mix **labeled** genomic shards into training (experience replay). |
| `--rehearsal-shards N` | `-1` | How many genomic shards to mix (`-1` = all). |
| `--distill-weight λ` | `0.0` | Distillation / LwF strength (`0` disables). |
| `--distill-teacher` | = `--pretrained-model` | Frozen teacher checkpoint. |
| `--distill-shards` | none | Genomic anchors the teacher scores (**no labels needed**). |
| `--distill-batch-size` | = train batch size | Batch size for the anchor pass (lower it if OOM). |

---

## See also

- [`transfer` documentation](../../docs/source/content/openspliceai_transfer.rst) — full option list and
  the "Mitigating catastrophic forgetting" section.
- [`create-data` documentation](../../docs/source/content/openspliceai_create-data.rst).
