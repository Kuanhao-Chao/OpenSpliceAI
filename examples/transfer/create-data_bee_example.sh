#!/bin/bash
# Step 1 of the transfer workflow: build train/validation/test HDF5 datasets from a
# genome FASTA + annotation GFF, ready for `openspliceai transfer`. See README.md.
#
# Here we use a small honey-bee sample so the demo is fast. The output datasets feed
# transfer_example.sh and transfer_forgetting_cmd.sh (same examples/transfer/results/).

# Resolve the repository root (this script lives in <repo>/examples/transfer/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

GENOME="$PDIR/examples/data/honeybee/HAv3.1_genomic_10_sample.fna"
ANNOTATION="$PDIR/examples/data/honeybee/HAv3.1_genomic_10_sample.gff"
OUTPUT_DIR="$PDIR/examples/transfer/results/"
FLANKING_SIZE=80   # must match the --flanking-size used by the transfer scripts and the model

mkdir -p "${OUTPUT_DIR}"

# Produces dataset_{train,validation,test}.h5 in $OUTPUT_DIR.
openspliceai create-data \
    --genome-fasta "$GENOME" \
    --annotation-gff "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" \
    --flanking-size "$FLANKING_SIZE" \
    --parse-type canonical \
    --remove-paralogs --min-identity 0.8 --min-coverage 0.8 \
    --write-fasta > "$OUTPUT_DIR/create-data.log" 2> "$OUTPUT_DIR/create-data.err.log"
