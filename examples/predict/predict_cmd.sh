#!/bin/bash
# A script to run the openspliceai predict command with the required arguments

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# # Ensure the openspliceai package is installed
# pip install $PDIR

# Define required arguments
DATA_PATH="$PDIR/examples/data/chr22.fa"
MODEL_PATH="$PDIR/models/openspliceai-mane/10000nt/model_10000nt_rs14.pt"
FLANKING_SIZE=10000
OUTPUT_PATH="$PDIR/examples/predict/results"

# Define optional arguments (set to empty string to disable)
ANNOTATION_PATH="$PDIR/examples/data/chr22.gff"
THRESHOLD=0.9
DEBUG_MODE=""
TURBO_MODE=T

# Build the openspliceai predict command
CMD="openspliceai predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH""

# Add optional flags
if [ -n "$ANNOTATION_PATH" ]; then
  CMD="$CMD -a "$ANNOTATION_PATH""
fi

if [ -n "$THRESHOLD" ]; then
  CMD="$CMD -t $THRESHOLD"
fi

if [ -n "$DEBUG_MODE" ]; then
  CMD="$CMD -D"-
fi

if [ -z "$TURBO_MODE" ]; then
  CMD="$CMD -p"
fi

# Run the command
echo $CMD
$CMD