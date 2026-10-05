#!/bin/bash
# Score a reference-anchored custom sequence (a multi-variant window) as a single MNV record.
# See custom_sequence_mnv.vcf: encode the whole edited window as one REF->ALT record and let
# `openspliceai variant` add the flanking context. (For sequences NOT in a reference genome,
# use score_custom_sequence.py instead.)

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Define required arguments (point REF_GENOME_PATH at the reference the VCF's REF alleles match)
REF_GENOME_PATH="/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
MODEL_PATH="$PDIR/models/openspliceai-mane/10000nt/"
INPUT_PATH="$SCRIPT_DIR/custom_sequence_mnv.vcf"
OUTPUT_PATH="$SCRIPT_DIR/custom_sequence_mnv.scored.vcf"
FLANKING_SIZE=10000
MODEL_TYPE="pytorch"
ANNOTATION_PATH="$PDIR/data/grch38.txt"

# -D 60 leaves headroom for windows up to ~60 bp (default 50 allows up to 51 bp REF)
CMD="openspliceai variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -D 60 -I "$INPUT_PATH" -O "$OUTPUT_PATH" --precision 4"

# Run the command
echo $CMD
$CMD
