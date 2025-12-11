# Annotation Converter (GFF to TSV)

This script converts a GFF3 annotation file into the TSV format required by OpenSpliceAI (similar to `data/grch38_chr.txt`).

## Description

The script parses a GFF3 file, filters for `protein_coding` genes, selects the longest (canonical) transcript for each gene, and outputs the coordinates of the gene, transcript, and exons.

## Usage

```bash
python annotation_converter.py <gff_file> <output_file>
```

### Arguments

- `gff_file`: Path to the input GFF3 file.
- `output_file`: Path to the output TSV file.

## Output Format

The output is a tab-separated values (TSV) file with the following columns:

1. `#NAME`: Gene name (from `Name` attribute or ID).
2. `CHROM`: Chromosome name (e.g., `chr1`).
3. `STRAND`: Strand (`+` or `-`).
4. `TX_START`: Transcript start position (0-based).
5. `TX_END`: Transcript end position (1-based).
6. `EXON_START`: Comma-separated list of exon start positions (0-based).
7. `EXON_END`: Comma-separated list of exon end positions (1-based).
