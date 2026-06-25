#!/bin/bash
# Step 3: fine-tune (transfer) a pretrained OpenSpliceAI model on a NARROW dataset while
# mitigating CATASTROPHIC FORGETTING of canonical genomic splice sites.
#
# Builds on transfer_example.sh by turning on the forgetting-mitigation flags (v0.0.8.dev0).
# Each flag is explained inline below; see examples/transfer/README.md (Step 3 + "the four
# levers") and the "Mitigating catastrophic forgetting" section of the transfer docs.
#
# Prerequisites:
#   - Step 1  (create-data_bee_example.sh)    -> the target fine-tuning datasets
#   - Step 1b (create-data_genomic_eval.sh)   -> a human held-out genomic set (recommended)

# Resolve the repository root (this script lives in <repo>/examples/transfer/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

SPECIES="MANE"
FLANKING_SIZE="80"   # 80nt keeps the demo fast; use 10000 for the full model

# Output directory
OUTPUT_DIR="$PDIR/examples/transfer/results_forgetting/"
mkdir -p "$OUTPUT_DIR"

# --- Datasets ---------------------------------------------------------------
# Your (narrow) fine-tuning data, produced by Step 1 (create-data_bee_example.sh):
TRAIN_DATASET="$PDIR/examples/transfer/results/dataset_train.h5"
TEST_DATASET="$PDIR/examples/transfer/results/dataset_test.h5"

# The GENOMIC set is the distribution we don't want to forget. Since we're adapting a HUMAN model to
# honey bee, that's human genomic sequence. Build it with Step 1b (create-data_genomic_eval.sh) ->
# examples/transfer/results_genomic/. It feeds three flags below: the forgetting gauge
# (--genomic-eval-dataset), rehearsal (--rehearsal-dataset), and distillation anchors (--distill-shards).
# All must be at the SAME flanking size (80) as the training data.
GENOMIC_EVAL="$PDIR/examples/transfer/results_genomic/dataset_test.h5"    # held-out gauge
GENOMIC_SRC="$PDIR/examples/transfer/results_genomic/dataset_train.h5"    # rehearse / distill on
# Fallback so the script still runs if you skipped Step 1b: reuse the target test set (demonstrates the
# mechanics, but the "forgetting curve" then reflects the target test set, not human genomic loss).
if [ ! -f "$GENOMIC_EVAL" ]; then GENOMIC_EVAL="$TEST_DATASET"; GENOMIC_SRC="$TEST_DATASET"; fi

# Pretrained checkpoint to fine-tune (also the default distillation teacher).
PRETRAINED_MODEL="$PDIR/models/openspliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs10.pt"

# --- Run --------------------------------------------------------------------
openspliceai transfer \
    --flanking-size "$FLANKING_SIZE" \
    --train-dataset "$TRAIN_DATASET" \
    --test-dataset "$TEST_DATASET" \
    --pretrained-model "$PRETRAINED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --project-name "${SPECIES}_transfer_forgetting" \
    --loss cross_entropy_loss \
    --epochs 10 \
    `# (1) reduce plasticity: train only the last 2 residual units (progressive: raise to 4, 8 over rounds)` \
    --unfreeze 2 \
    `# (2) stop weight decay from shrinking weights to zero; --l2sp pulls them toward the pretrained start` \
    --weight-decay 0 \
    --l2sp 0.1 \
    `# (3) rehearsal: interleave real LABELED genomic shards into training (replay)` \
    --rehearsal-dataset "$GENOMIC_SRC" \
    --rehearsal-shards 2 \
    `# (4) distillation (LwF): match a frozen teacher on genomic windows -- needs NO labels` \
    --distill-weight 0.5 \
    --distill-shards "$GENOMIC_SRC" \
    `# the gauge: log a per-epoch forgetting curve (measurement only) -> $OUTPUT_DIR/.../LOG/GENOMIC/` \
    --genomic-eval-dataset "$GENOMIC_EVAL"

# After training, read the genomic forgetting curve (one value per epoch) against the in-distribution
# TEST metrics and pick the checkpoint on the gain-vs-retention front (see README Step 4):
#   retention: .../LOG/GENOMIC/donor_auprc.txt , acceptor_auprc.txt
#   gain:      .../LOG/TEST/donor_auprc.txt    , acceptor_auprc.txt
#   checkpoints: .../models/model_<epoch>.pt
