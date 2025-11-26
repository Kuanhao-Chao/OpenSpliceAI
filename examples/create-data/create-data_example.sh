#!/bin/bash
# A script to run the openspliceai create-data command with the required arguments

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# GENOME="$PDIR/examples/data/human/GCF_000001405.40_GRCh38.p14_genomic.fna"
# ANNOTATION="$PDIR/examples/data/human/MANE.GRCh38.v1.3.refseq_genomic.gff"

GENOME="$PDIR/examples/data/human/GCF_000001405.40_GRCh38.p14_genomic_10_sample.fna"
ANNOTATION="$PDIR/examples/data/human/MANE.GRCh38.v1.3.refseq_genomic_10_sample.gff"
OUTPUT_DIR="$PDIR/examples/create-data/results/"
EXTRA_PARAMS="--split-method human --canonical-only"

mkdir -p ${OUTPUT_DIR}

# Run the command
openspliceai create-data \
    --remove-paralogs \
    --min-identity 0.8 \
    --min-coverage 0.8 \
    --genome-fasta "$GENOME" \
    --annotation-gff "$ANNOTATION" \
    --output-dir "$OUTPUT_DIR" \
    --parse-type canonical \
    --write-fasta \
    $EXTRA_PARAMS > "$OUTPUT_DIR/output.log" 2> "$OUTPUT_DIR/error.log"
