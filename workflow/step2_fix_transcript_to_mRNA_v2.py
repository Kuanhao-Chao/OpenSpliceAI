#!/usr/bin/env python3
"""
将过滤后的 GFF3 中的 transcript 改为 mRNA
"""

import argparse

DEFAULT_INPUT="/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step1.gff3"
DEFAULT_OUTPUT="/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step2.gff3"

parser = argparse.ArgumentParser(description="Rename transcript features to mRNA (Step 2).")
parser.add_argument("--input", default=DEFAULT_INPUT, help="Input GFF3 path.")
parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output GFF3 path.")
args = parser.parse_args()

INPUT=args.input
OUTPUT=args.output 

print("="*60) 
print("Step 2: Converting transcript to mRNA")
print("="*60)

print(f"\nInput:  {INPUT}")
print(f"Output: {OUTPUT}")

count = 0
total_lines = 0

with open(INPUT, 'r') as f_in, open(OUTPUT, 'w') as f_out:
    for line in f_in:
        total_lines += 1
        
        if line.startswith('#'):
            f_out.write(line)
            continue
        
        fields = line.strip().split('\t')
        if len(fields) >= 9:
            if fields[2] == 'transcript':
                fields[2] = 'mRNA'
                count += 1
        
        f_out.write('\t'.join(fields) + '\n')

print(f"\n✅ Processed {total_lines:,} lines")
print(f"✅ Converted {count:,} transcripts to mRNA")

# 验证
import subprocess

print("\n" + "="*60)
print("Verification:")
print("="*60)

result = subprocess.run(['grep', '-c', '\tmRNA\t', OUTPUT], 
                       capture_output=True, text=True)
mrna_count = int(result.stdout.strip()) if result.returncode == 0 else 0

result = subprocess.run(['grep', '-c', '\ttranscript\t', OUTPUT], 
                       capture_output=True, text=True)
transcript_count = int(result.stdout.strip()) if result.returncode == 0 else 0

print(f"mRNA count: {mrna_count:,}")
print(f"Transcript count: {transcript_count:,}")

if mrna_count > 0 and transcript_count == 0:
    print("\n✅ Conversion successful!")
else:
    print("\n⚠️  Warning: Check the output file")

print(f"\nOutput saved to: {OUTPUT}")
