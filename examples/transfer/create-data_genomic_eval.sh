#!/bin/bash
# Step 1b (optional but recommended): build a HUMAN held-out GENOMIC dataset — the distribution we do
# NOT want the model to forget while it is being adapted to a new species. This is the conceptually
# correct source for the forgetting flags in transfer_forgetting_cmd.sh:
#     --genomic-eval-dataset   (the gauge / forgetting curve)
#     --rehearsal-dataset      (replay)
#     --distill-shards         (distillation anchors)
#
# Here we use human chr22 (ships in examples/data/). It MUST be built at the same --flanking-size (80)
# as the fine-tuning data and the pretrained model, or the per-batch context clipping won't line up.

# Resolve the repository root (this script lives in <repo>/examples/transfer/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

GENOME="$PDIR/examples/data/chr22.fa"
ANNOTATION="$PDIR/examples/data/chr22.gff"
OUTPUT_DIR="$PDIR/examples/transfer/results_genomic/"
FLANKING_SIZE=80   # must match the transfer flanking size and the pretrained model

mkdir -p "${OUTPUT_DIR}"

# Produces dataset_{train,validation,test}.h5 of human genomic windows in $OUTPUT_DIR:
#   - dataset_test.h5   -> use for --genomic-eval-dataset (held-out gauge)
#   - dataset_train.h5  -> use for --rehearsal-dataset / --distill-shards
openspliceai create-data \
    --genome-fasta "$GENOME" \
    --annotation-gff "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" \
    --flanking-size "$FLANKING_SIZE" \
    --parse-type canonical \
    --write-fasta > "$OUTPUT_DIR/create-data.log" 2> "$OUTPUT_DIR/create-data.err.log"
