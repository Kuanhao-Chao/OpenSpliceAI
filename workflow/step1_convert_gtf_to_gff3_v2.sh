#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Convert filtered GTF to GFF3"
echo "=========================================="

INPUT="${1:-/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step0.gtf}"
OUTPUT="${2:-/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step1.gff3}"

if [ ! -f "$INPUT" ]; then
    echo "❌ Error: Input not found: $INPUT"
    echo "Please run step0_filter_by_isoform_list.py first!"
    exit 1
fi

if [ -f "$OUTPUT" ]; then
    echo "⚠️  Output already exists: $OUTPUT"
    read -p "Overwrite? (y/n): " answer
    if [ "$answer" != "y" ]; then
        echo "Skipping conversion."
        exit 0
    fi
fi

echo "Converting $INPUT to GFF3..."
agat_convert_sp_gxf2gxf.pl \
  --gff "$INPUT" \
  --output "$OUTPUT" \
  2>&1 | tee agat_conversion_filtered.log

echo -e "\n✅ Conversion completed!"
echo "Output: $OUTPUT"

echo -e "\nStatistics:"
echo "  Total lines: $(wc -l < $OUTPUT)"
echo "  Gene count: $(grep -c $'\tgene\t' $OUTPUT)"
echo "  Transcript count: $(grep -c $'\ttranscript\t' $OUTPUT)"
echo "  Exon count: $(grep -c $'\texon\t' $OUTPUT)"
