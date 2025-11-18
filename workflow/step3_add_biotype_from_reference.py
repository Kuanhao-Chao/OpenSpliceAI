#!/usr/bin/env python3
"""
基于 GENCODE reference GTF 添加 gene_biotype
1. 从 reference GTF 读取 gene_id → gene_type 映射
2. 为 filtered GFF3 中的每个基因匹配 biotype
3. 匹配不上的默认为 protein_coding
"""
import re
import argparse

DEFAULT_REFERENCE_GTF = '/home1/xyf/data/openspliceai_data/gtf/reference_gtf/genes.gtf'
DEFAULT_INPUT_GFF3 = "/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step2.gff3"
DEFAULT_OUTPUT_GFF3 = "/home1/xyf/data/openspliceai_tissue_data/tissue_gtf/neuron/neuron_step3.gff3"

parser = argparse.ArgumentParser(description="Add gene_biotype to GFF3 (Step 3).")
parser.add_argument("--reference", default=DEFAULT_REFERENCE_GTF, help="Reference GTF with gene_biotype.")
parser.add_argument("--input", default=DEFAULT_INPUT_GFF3, help="Input GFF3 path.")
parser.add_argument("--output", default=DEFAULT_OUTPUT_GFF3, help="Output GFF3 path.")
args = parser.parse_args()

REFERENCE_GTF = args.reference
INPUT_GFF3 = args.input
OUTPUT_GFF3 = args.output

print("="*70)
print("Step 3: Add biotype based on GENCODE reference")
print("="*70)

# ============================================
# Phase 1: 从 reference GTF 读取 gene_type
# ============================================
print(f"\nPhase 1: Reading reference GTF...")
print(f"File: {REFERENCE_GTF}")

gene_type_map = {}  # gene_id -> gene_type

try:
    with open(REFERENCE_GTF, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            # 只处理 gene 行
            if fields[2] != 'gene':
                continue
            
            attributes = fields[8]
            
            # 提取 gene_id 和 gene_type
            gene_id = None
            gene_type = None
            
            # GENCODE GTF 格式：gene_id "ENSG00000243485"
            gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            if gene_id_match:
                gene_id = gene_id_match.group(1)
            
            # 可能是 gene_type 或 gene_biotype
            gene_type_match = re.search(r'gene_type "([^"]+)"', attributes)
            if gene_type_match:
                gene_type = gene_type_match.group(1)
            else:
                # 尝试查找 gene_biotype
                gene_type_match = re.search(r'gene_biotype "([^"]+)"', attributes)
                if gene_type_match:
                    gene_type = gene_type_match.group(1)
            
            if gene_id and gene_type:
                gene_type_map[gene_id] = gene_type

except FileNotFoundError:
    print(f"\n❌ Error: Reference file not found: {REFERENCE_GTF}")
    print("Please check the file path.")
    exit(1)

print(f"✅ Loaded {len(gene_type_map):,} gene type mappings")

# 统计 gene_type 分布
gene_type_stats = {}
for gene_type in gene_type_map.values():
    gene_type_stats[gene_type] = gene_type_stats.get(gene_type, 0) + 1

print(f"\nGene type distribution in reference:")
for gene_type, count in sorted(gene_type_stats.items(), key=lambda x: -x[1])[:10]:
    print(f"  {gene_type}: {count:,}")

# ============================================
# Phase 2: 处理 GFF3，添加 biotype
# ============================================
print(f"\n" + "="*70)
print("Phase 2: Adding biotype to GFF3...")
print("="*70)
print(f"Input:  {INPUT_GFF3}")
print(f"Output: {OUTPUT_GFF3}")

stats = {
    'gene': 0,
    'mRNA': 0,
    'exon': 0,
    'other': 0,
    'matched': 0,
    'unmatched': 0,
    'biotype_stats': {}
}

# 先收集所有基因的 gene_id 和对应的 biotype
gene_biotype_for_gff = {}  # gene_id -> biotype

with open(INPUT_GFF3, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        if fields[2] != 'gene':
            continue
        
        attributes = fields[8]
        
        # 提取 gene_id（GFF3 格式：ID=ENSG00000184895）
        id_match = re.search(r'ID=([^;]+)', attributes)
        gene_id_match = re.search(r'gene_id=([^;]+)', attributes)
        
        # 优先使用 ID，然后是 gene_id
        gene_id = None
        if id_match:
            gene_id = id_match.group(1)
        elif gene_id_match:
            gene_id = gene_id_match.group(1)
        
        if not gene_id:
            continue
        
        # 查找对应的 gene_type
        if gene_id in gene_type_map:
            biotype = gene_type_map[gene_id]
            stats['matched'] += 1
        else:
            # 默认为 protein_coding
            biotype = 'protein_coding'
            stats['unmatched'] += 1
        
        gene_biotype_for_gff[gene_id] = biotype
        
        # 统计 biotype
        stats['biotype_stats'][biotype] = stats['biotype_stats'].get(biotype, 0) + 1

print(f"\nMatching results:")
print(f"  Matched genes: {stats['matched']:,}")
print(f"  Unmatched genes (defaulting to protein_coding): {stats['unmatched']:,}")

print(f"\nBiotype distribution for output:")
for biotype, count in sorted(stats['biotype_stats'].items(), key=lambda x: -x[1]):
    print(f"  {biotype}: {count:,}")

# 现在写入输出文件，为每个特征添加 gene_biotype
print(f"\nWriting output...")

feature_stats = {'gene': 0, 'mRNA': 0, 'exon': 0, 'other': 0}

with open(INPUT_GFF3, 'r') as f_in, open(OUTPUT_GFF3, 'w') as f_out:
    for line in f_in:
        if line.startswith('#'):
            f_out.write(line)
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        attributes = fields[8]
        
        # 提取 gene_id（可能在 ID, Parent, 或 gene_id 属性中）
        gene_id = None
        
        # 对于 gene 特征，gene_id 在 ID 中
        if fields[2] == 'gene':
            id_match = re.search(r'ID=([^;]+)', attributes)
            if id_match:
                gene_id = id_match.group(1)
        
        # 对于 mRNA，gene_id 在 Parent 或 gene_id 属性中
        elif fields[2] == 'mRNA':
            parent_match = re.search(r'Parent=([^;]+)', attributes)
            if parent_match:
                gene_id = parent_match.group(1)
        
        # 对于 exon 等，先找 gene_id 属性
        if not gene_id:
            gene_id_match = re.search(r'gene_id=([^;]+)', attributes)
            if gene_id_match:
                gene_id = gene_id_match.group(1)
        
        # 查找对应的 biotype
        biotype = gene_biotype_for_gff.get(gene_id, 'protein_coding')
        
        # 移除已有的 gene_biotype
        attributes = re.sub(r';?gene_biotype=[^;]*', '', attributes)
        
        # 添加 gene_biotype
        if not attributes.endswith(';'):
            attributes += ';'
        attributes += f'gene_biotype={biotype}'
        
        # 统计
        if fields[2] in feature_stats:
            feature_stats[fields[2]] += 1
        else:
            feature_stats['other'] += 1
        
        fields[8] = attributes
        f_out.write('\t'.join(fields) + '\n')

# ============================================
# 总结
# ============================================
print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"Features processed:")
for ftype, count in sorted(feature_stats.items()):
    print(f"  {ftype}: {count:,}")

print(f"\n✅ Completed!")
print(f"✅ Output saved to: {OUTPUT_GFF3}")

# 验证
print("\n" + "="*70)
print("Verification:")
print("="*70)

import subprocess

result = subprocess.run(['grep', '-c', 'gene_biotype=', OUTPUT_GFF3], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"Lines with gene_biotype: {result.stdout.strip()}")

# 显示一些示例
print(f"\nSample output lines:")
result = subprocess.run(['grep', '-m', '3', '\tgene\t', OUTPUT_GFF3], 
                       capture_output=True, text=True)
if result.returncode == 0:
    for line in result.stdout.strip().split('\n'):
        print(f"  {line[:150]}...")

print("\n" + "="*70)
