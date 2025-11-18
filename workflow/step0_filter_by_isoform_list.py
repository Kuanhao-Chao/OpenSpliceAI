#!/usr/bin/env python3
"""
根据 all_isoform.txt 过滤原始 GTF
只保留列表中转录本及其相关的 gene 和 exon
"""
import sys
from collections import defaultdict
import argparse

DEFAULT_ISOFORM_LIST = '/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/isoform/neuron.txt'
DEFAULT_INPUT_GTF = '/home1/xyf/data/openspliceai_data/gtf/origin_gtf/20251014_OUT.transcript_models.gtf'
DEFAULT_OUTPUT_GTF = '/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step0.gtf'

parser = argparse.ArgumentParser(description="Filter a GTF using an isoform list (Step 0).")
parser.add_argument("--isoform-list", default=DEFAULT_ISOFORM_LIST, help="Path to isoform list file.")
parser.add_argument("--input-gtf", default=DEFAULT_INPUT_GTF, help="Original GTF to filter.")
parser.add_argument("--output-gtf", default=DEFAULT_OUTPUT_GTF, help="Filtered GTF output path.")
args = parser.parse_args()

ISOFORM_LIST = args.isoform_list
INPUT_GTF = args.input_gtf
OUTPUT_GTF = args.output_gtf

print("="*70)
print("Step 0: Filter GTF by isoform list")
print("="*70)

# ============================================
# Phase 1: 读取 all_isoform.txt
# ============================================
print(f"\nPhase 1: Reading isoform list...")
print(f"File: {ISOFORM_LIST}")

transcript_ids_to_keep = set()

try:
    with open(ISOFORM_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 假设每行是一个转录本 ID
                transcript_ids_to_keep.add(line)
except FileNotFoundError:
    print(f"\n❌ Error: File not found: {ISOFORM_LIST}")
    sys.exit(1)

print(f"✅ Loaded {len(transcript_ids_to_keep):,} transcript IDs")

if len(transcript_ids_to_keep) == 0:
    print("\n❌ Error: No transcript IDs found in the list!")
    sys.exit(1)

# 显示前几个
print(f"\nSample transcript IDs:")
for i, tid in enumerate(list(transcript_ids_to_keep)[:5]):
    print(f"  {i+1}. {tid}")

# ============================================
# Phase 2: 第一遍扫描 - 收集相关的 gene_id
# ============================================
print(f"\n" + "="*70)
print("Phase 2: First pass - identifying genes...")
print("="*70)
print(f"Input: {INPUT_GTF}")

gene_ids_to_keep = set()

with open(INPUT_GTF, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        feature_type = fields[2]
        attributes = fields[8]
        
        # 提取 transcript_id 和 gene_id
        transcript_id = None
        gene_id = None
        
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('transcript_id'):
                transcript_id = attr.split('"')[1]
            elif attr.startswith('gene_id'):
                gene_id = attr.split('"')[1]
        
        # 如果这个转录本在保留列表中，记录其 gene_id
        if transcript_id in transcript_ids_to_keep:
            if gene_id:
                gene_ids_to_keep.add(gene_id)

print(f"✅ Found {len(gene_ids_to_keep):,} genes with target transcripts")

if len(gene_ids_to_keep) == 0:
    print("\n⚠️  Warning: No genes found matching the transcript list!")
    print("This might mean:")
    print("  1. Transcript IDs in the list don't match GTF")
    print("  2. Different ID formats (e.g., with/without version numbers)")
    print("\nShowing first few lines from GTF for comparison:")
    with open(INPUT_GTF, 'r') as f:
        count = 0
        for line in f:
            if not line.startswith('#') and '\ttranscript\t' in line:
                print(f"  {line.strip()}")
                count += 1
                if count >= 3:
                    break
    sys.exit(1)

# ============================================
# Phase 3: 第二遍扫描 - 过滤并输出
# ============================================
print(f"\n" + "="*70)
print("Phase 3: Second pass - filtering...")
print("="*70)
print(f"Output: {OUTPUT_GTF}")

stats = {
    'gene': 0,
    'transcript': 0,
    'exon': 0,
    'other': 0,
    'skipped': 0
}

with open(INPUT_GTF, 'r') as f_in, open(OUTPUT_GTF, 'w') as f_out:
    for line in f_in:
        # 保留注释行
        if line.startswith('#'):
            f_out.write(line)
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        feature_type = fields[2]
        attributes = fields[8]
        
        # 提取 gene_id 和 transcript_id
        gene_id = None
        transcript_id = None
        
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_id'):
                gene_id = attr.split('"')[1]
            elif attr.startswith('transcript_id'):
                transcript_id = attr.split('"')[1]
        
        # 判断是否保留
        keep = False
        
        if feature_type == 'gene':
            # 保留在列表中的基因
            if gene_id in gene_ids_to_keep:
                keep = True
                stats['gene'] += 1
        
        elif feature_type == 'transcript':
            # 保留在列表中的转录本
            if transcript_id in transcript_ids_to_keep:
                keep = True
                stats['transcript'] += 1
        
        else:
            # 其他特征（如 exon）：检查其所属转录本
            if transcript_id in transcript_ids_to_keep:
                keep = True
                if feature_type == 'exon':
                    stats['exon'] += 1
                else:
                    stats['other'] += 1
        
        if keep:
            f_out.write(line)
        else:
            stats['skipped'] += 1

# ============================================
# 总结
# ============================================
print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Features kept:")
print(f"  Gene:       {stats['gene']:>8,}")
print(f"  Transcript: {stats['transcript']:>8,}")
print(f"  Exon:       {stats['exon']:>8,}")
print(f"  Other:      {stats['other']:>8,}")
print(f"  Total kept: {sum([stats[k] for k in ['gene', 'transcript', 'exon', 'other']]):>8,}")
print(f"\nFeatures skipped: {stats['skipped']:>8,}")

print(f"\n✅ Filtering completed!")
print(f"✅ Output saved to: {OUTPUT_GTF}")

# 验证输出
print("\n" + "="*70)
print("Verification:")
print("="*70)

import subprocess

result = subprocess.run(['wc', '-l', OUTPUT_GTF], 
                       capture_output=True, text=True)
if result.returncode == 0:
    line_count = result.stdout.strip().split()[0]
    print(f"Output file lines: {line_count}")

result = subprocess.run(['grep', '-c', '\tgene\t', OUTPUT_GTF], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"Gene count: {result.stdout.strip()}")

result = subprocess.run(['grep', '-c', '\ttranscript\t', OUTPUT_GTF], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"Transcript count: {result.stdout.strip()}")

result = subprocess.run(['grep', '-c', '\texon\t', OUTPUT_GTF], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"Exon count: {result.stdout.strip()}")

print("\n" + "="*70)
