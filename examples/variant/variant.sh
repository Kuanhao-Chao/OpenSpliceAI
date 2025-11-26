#!/bin/bash
# A script to run the openspliceai variant command with the required arguments

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# # Ensure the openspliceai package is installed
# pip install $PDIR

# Define required arguments
REF_GENOME_PATH="$PDIR/data/hg19.fa"
MODEL_PATH="$PDIR/models/openspliceai-mane/400nt/model_400nt_rs14.pt"
INPUT_PATH="$PDIR/examples/data/input.vcf"
OUTPUT_PATH="$PDIR/examples/variant/output.vcf"
FLANKING_SIZE=400
MODEL_TYPE="pytorch"
ANNOTATION_PATH="$PDIR/data/grch37.txt" # NOTE: this is a custom annotation file

# Build the variant command
CMD="openspliceai variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -I "$INPUT_PATH" -O "$OUTPUT_PATH""

# Run the command
echo $CMD 
$CMD 