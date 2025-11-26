#!/bin/bash
# A script to run the openspliceai train command with the required arguments

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

SPECIES="MANE"
FLANKING_SIZE="10000"

# Set up output directory
OUTPUT_DIR="$PDIR/examples/transfer/results/"
mkdir -p $OUTPUT_DIR

# Set up dataset paths (adjust these as needed for each species)
TRAIN_DATASET="$PDIR/examples/transfer/results/dataset_train.h5"
TEST_DATASET="$PDIR/examples/transfer/results/dataset_test.h5"
PRETRAINED_MODEL="$PDIR/models/openspliceai-mane/10000nt/model_10000nt_rs10.pt"

OUTPUT_FILE="$OUTPUT_DIR/transfer_output.log"
ERROR_FILE="$OUTPUT_DIR/transfer_error.log"

# Run the OpenSpliceAI fine-tuning command
openspliceai transfer --flanking-size $FLANKING_SIZE \
    --train-dataset $TRAIN_DATASET \
    --test-dataset $TEST_DATASET \
    --pretrained-model $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --project-name ${SPECIES}_train \
    --scheduler CosineAnnealingWarmRestarts \
    --loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE