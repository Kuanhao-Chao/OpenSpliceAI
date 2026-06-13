#!/bin/bash
# A script to score Indel variants with OpenSpliceAI

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Define required arguments
REF_GENOME_PATH="/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
MODEL_PATH="$PDIR/models/openspliceai-mane/10000nt/"
INPUT_PATH="./test_input.vcf"
OUTPUT_PATH="./test_output.vcf"
FLANKING_SIZE=10000
MODEL_TYPE="pytorch"
ANNOTATION_PATH="$PDIR/data/grch38.txt"

# Build the variant command
CMD="openspliceai variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -I "$INPUT_PATH" -O "$OUTPUT_PATH" --precision 8 -D 3000"

# Run the command
echo $CMD
$CMD
