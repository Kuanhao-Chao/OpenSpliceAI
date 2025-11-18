#!/bin/bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TISSUE="${1:-neuron}"

ISOFORM_LIST="/home1/xyf/data/openspliceai_tissue_data/isoform/${TISSUE}.txt"
ORIGINAL_GTF="/home1/xyf/data/openspliceai_data/gtf/origin_gtf/20251014_OUT.transcript_models.gtf"
TISSUE_DIR="/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/${TISSUE}"
STEP0_GTF="${TISSUE_DIR}/${TISSUE}_step0.gtf"
STEP1_GFF="${TISSUE_DIR}/${TISSUE}_step1.gff3"
STEP2_GFF="${TISSUE_DIR}/${TISSUE}_step2.gff3"
STEP3_GFF="${TISSUE_DIR}/${TISSUE}_step3.gff3"
REFERENCE_GTF="/home1/xyf/data/openspliceai_data/gtf/reference_gtf/genes.gtf"

mkdir -p "$TISSUE_DIR"

echo "=========================================="
echo "Running ${TISSUE} GTF → GFF3 preprocessing pipeline"
echo "=========================================="

echo ""
echo "[Step 0] Filtering original GTF by isoform list"
echo "    Isoform list : $ISOFORM_LIST"
echo "    Input  GTF   : $ORIGINAL_GTF"
echo "    Output GTF   : $STEP0_GTF"
python "$WORKFLOW_DIR/step0_filter_by_isoform_list.py" \
  --isoform-list "$ISOFORM_LIST" \
  --input-gtf "$ORIGINAL_GTF" \
  --output-gtf "$STEP0_GTF"

echo ""
echo "[Step 1] Converting filtered GTF to GFF3 (AGAT)"
echo "    Input  : $STEP0_GTF"
echo "    Output : $STEP1_GFF"
rm -f "$STEP1_GFF"
bash "$WORKFLOW_DIR/step1_convert_gtf_to_gff3_v2.sh" "$STEP0_GTF" "$STEP1_GFF"

echo ""
echo "[Step 2] Renaming transcript features to mRNA"
echo "    Input  : $STEP1_GFF"
echo "    Output : $STEP2_GFF"
python "$WORKFLOW_DIR/step2_fix_transcript_to_mRNA_v2.py" \
  --input "$STEP1_GFF" \
  --output "$STEP2_GFF"

echo ""
echo "[Step 3] Adding gene_biotype annotations from reference GTF"
echo "    Reference : $REFERENCE_GTF"
echo "    Input     : $STEP2_GFF"
echo "    Output    : $STEP3_GFF"
python "$WORKFLOW_DIR/step3_add_biotype_from_reference.py" \
  --reference "$REFERENCE_GTF" \
  --input "$STEP2_GFF" \
  --output "$STEP3_GFF"

echo ""
echo "✅ Pipeline completed. Final GFF3: $STEP3_GFF"
