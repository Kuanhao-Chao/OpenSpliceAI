#!/bin/bash
# Step 2 of the transfer workflow: BASELINE fine-tuning of a pretrained OpenSpliceAI
# model on the datasets built in Step 1 (no catastrophic-forgetting mitigation). See
# README.md. For the forgetting-aware version, use transfer_forgetting_cmd.sh.
#
# Prerequisite: run create-data_bee_example.sh first to produce
#   examples/transfer/results/dataset_{train,validation,test}.h5

# Resolve the repository root (this script lives in <repo>/examples/transfer/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

SPECIES="MANE"
FLANKING_SIZE="80"   # must match the create-data flanking size and the pretrained model

# Output + dataset paths (Step 1 wrote the datasets here)
OUTPUT_DIR="$PDIR/examples/transfer/results/"
mkdir -p "$OUTPUT_DIR"
TRAIN_DATASET="$OUTPUT_DIR/dataset_train.h5"
TEST_DATASET="$OUTPUT_DIR/dataset_test.h5"
PRETRAINED_MODEL="$PDIR/models/openspliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs10.pt"

OUTPUT_FILE="$OUTPUT_DIR/transfer_output.log"
ERROR_FILE="$OUTPUT_DIR/transfer_error.log"

# Run the OpenSpliceAI fine-tuning (transfer) command
openspliceai transfer --flanking-size "$FLANKING_SIZE" \
    --train-dataset "$TRAIN_DATASET" \
    --test-dataset "$TEST_DATASET" \
    --pretrained-model "$PRETRAINED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --project-name "${SPECIES}_transfer" \
    --scheduler CosineAnnealingWarmRestarts \
    --loss cross_entropy_loss > "$OUTPUT_FILE" 2> "$ERROR_FILE"
