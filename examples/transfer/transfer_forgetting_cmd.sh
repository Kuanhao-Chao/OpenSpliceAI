#!/bin/bash
# Fine-tune (transfer) a pretrained OpenSpliceAI model on a NARROW dataset while
# mitigating CATASTROPHIC FORGETTING of canonical genomic splice sites.
#
# This builds on transfer_example.sh by adding the forgetting-mitigation flags
# introduced in v0.0.8.dev0. See examples/transfer/README.md (Step 3) and the
# "Mitigating catastrophic forgetting" section of the transfer docs.

# Resolve the repository root (this script lives in <repo>/examples/transfer/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

SPECIES="MANE"
FLANKING_SIZE="80"   # 80nt keeps the demo fast; use 10000 for the full model

# Output directory
OUTPUT_DIR="$PDIR/examples/transfer/results_forgetting/"
mkdir -p "$OUTPUT_DIR"

# --- Datasets ---------------------------------------------------------------
# Your (narrow) fine-tuning data, produced by `openspliceai create-data`:
TRAIN_DATASET="$PDIR/examples/transfer/results/dataset_train.h5"
TEST_DATASET="$PDIR/examples/transfer/results/dataset_test.h5"

# A held-out GENOMIC dataset (e.g. MANE test chromosomes) created at the SAME
# flanking size. It plays three roles below: forgetting curve, rehearsal source,
# and distillation anchors. For this demo we reuse the test set; in practice
# point these at real held-out genomic shards.
GENOMIC_DATASET="$TEST_DATASET"

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
    `# (1) reduce plasticity: freeze most layers instead of --unfreeze-all` \
    --unfreeze 2 \
    `# (2) stop decaying weights toward zero; pull them toward the pretrained start instead` \
    --weight-decay 0 \
    --l2sp 0.1 \
    `# (3) rehearsal: interleave genomic shards (true labels) into training` \
    --rehearsal-dataset "$GENOMIC_DATASET" \
    --rehearsal-shards 2 \
    `# (4) distillation (LwF): keep the student close to a frozen teacher on genomic anchors` \
    --distill-weight 0.5 \
    --distill-shards "$GENOMIC_DATASET" \
    `# track forgetting each epoch -> $OUTPUT_DIR/.../LOG/GENOMIC/` \
    --genomic-eval-dataset "$GENOMIC_DATASET"

# After training, inspect the genomic forgetting curve (one value per epoch):
#   donor:    .../LOG/GENOMIC/donor_auprc.txt
#   acceptor: .../LOG/GENOMIC/acceptor_auprc.txt
# and compare against the in-distribution metrics in LOG/TEST/ to pick the
# checkpoint (models/model_<epoch>.pt) on the gain-vs-retention front.
