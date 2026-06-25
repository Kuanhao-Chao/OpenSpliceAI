# Mitigating catastrophic forgetting during `transfer`

When you fine-tune OpenSpliceAI on a **narrow** dataset (a single locus, an assay covering a few
genes, etc.), a fully unfrozen model can **catastrophically forget** canonical splice sites elsewhere
in the genome — donor/acceptor scores at well-annotated junctions collapse toward zero even though
those sites are unchanged. The `transfer` subcommand has four optional, default-off controls (added in
v0.0.8.dev0) to detect and counteract this. This folder's
[`transfer_forgetting_cmd.sh`](./transfer_forgetting_cmd.sh) demonstrates them together.

## The flags

| Flag | What it does |
|---|---|
| `--weight-decay 0` | AdamW's default `0.01` decays every weight toward **zero** each step, eroding pretrained splice features. Set `0` to disable. |
| `--l2sp 0.1` | Instead of decaying toward zero, penalize drift away from the **pretrained** weights (active only with a distillation teacher). |
| `--unfreeze 2` | Freeze most layers (train only the last 2 residual units) to preserve general motif detectors — prefer this over `--unfreeze-all` on narrow data. |
| `--rehearsal-dataset` / `--rehearsal-shards` | Interleave real genomic shards (with their true labels) into training — experience replay. |
| `--distill-weight` / `--distill-shards` / `--distill-teacher` | Knowledge distillation (Learning without Forgetting): add `λ · cross_entropy(teacher_soft_targets, student)` on genomic anchor windows scored by a frozen copy of the pretrained model. **No genomic labels needed.** |
| `--genomic-eval-dataset` | Log a per-epoch **forgetting curve** (donor/acceptor AUPRC + top-k) under `LOG/GENOMIC/`. |

## Run it

```bash
bash examples/transfer/transfer_forgetting_cmd.sh
```

The demo uses the 80nt MANE model for speed and reuses the test set as the genomic source. In a real
run, point `--rehearsal-dataset` / `--distill-shards` / `--genomic-eval-dataset` at **held-out genomic
shards** (e.g. MANE test chromosomes) created with `openspliceai create-data` at the **same flanking
size** as your fine-tuning data (the per-batch context clipping assumes a fixed `SL + CL_max` width).

## Reading the results

Each epoch writes a value to the genomic forgetting curve:

```
results_forgetting/SpliceAI_MANE_transfer_forgetting_80_0_rs42/0/LOG/
├── GENOMIC/donor_auprc.txt      # genomic donor AUPRC per epoch  (retention)
├── GENOMIC/acceptor_auprc.txt
├── TEST/donor_auprc.txt         # in-distribution AUPRC per epoch (gain)
└── ...
models/model_<epoch>.pt          # one checkpoint per epoch
```

Plot the **GENOMIC** AUPRC (how much canonical-site accuracy is retained) against the **TEST** AUPRC
(in-distribution gain) across epochs, and pick the `model_<epoch>.pt` on the best gain-vs-retention
trade-off. Sweep `--distill-weight` / `--rehearsal-shards` to move along that curve.

See the full discussion in the
[`transfer` documentation](../../docs/source/content/openspliceai_transfer.rst)
("Mitigating catastrophic forgetting").
