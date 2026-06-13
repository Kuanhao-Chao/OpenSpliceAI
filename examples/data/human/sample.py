import argparse
import re
from collections import defaultdict, OrderedDict

def parse_attributes(attr_str):
    attrs = {}
    parts = attr_str.split(';')
    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            attrs[key.strip()] = value.strip()
    return attrs

def process_gff(input_gff, output_gff, chrom_pattern):
    selected_genes = defaultdict(list)  # chrom: list of (gene_id, end)
    max_end_per_chrom = {}

    # First pass to collect gene information
    with open(input_gff, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            chrom = fields[0]
            if not chrom_pattern.match(chrom):
                continue  # Skip non-main chromosomes
            feature_type = fields[2]
            if feature_type.lower() == 'gene':
                attrs = parse_attributes(fields[8])
                gene_id = attrs.get('ID')
                if not gene_id:
                    continue
                try:
                    end = int(fields[4])
                except ValueError:
                    continue  # Skip invalid entries
                if len(selected_genes[chrom]) < 10:
                    selected_genes[chrom].append((gene_id, end))
                    current_max = max_end_per_chrom.get(chrom, 0)
                    if end > current_max:
                        max_end_per_chrom[chrom] = end

    # Prepare selected gene IDs for quick lookup
    selected_gene_ids = defaultdict(set)
    for chrom in selected_genes:
        gene_ids = [gene[0] for gene in selected_genes[chrom]]
        selected_gene_ids[chrom] = set(gene_ids)

    # Second pass to write the selected entries
    with open(input_gff, 'r') as infile, open(output_gff, 'w') as outfile:
        for line in infile:
            if line.startswith('#'):
                outfile.write(line)
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                outfile.write(line)
                continue
            chrom = fields[0]
            if not chrom_pattern.match(chrom):
                continue  # Skip non-main chromosomes
            feature_type = fields[2]
            attrs = parse_attributes(fields[8])
            if feature_type.lower() == 'gene':
                gene_id = attrs.get('ID')
                if gene_id in selected_gene_ids.get(chrom, set()):
                    outfile.write(line)
            else:
                parents = attrs.get('Parent', '')
                parents = [p.strip() for p in parents.split(',')]
                for parent in parents:
                    if parent in selected_gene_ids.get(chrom, set()):
                        outfile.write(line)
                        break

    return max_end_per_chrom

def read_fasta(filename):
    sequences = OrderedDict()
    current_id = None
    current_seq = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    return sequences

def write_fasta(sequences, filename):
    with open(filename, 'w') as f:
        for chrom, seq in sequences.items():
            f.write(f'>{chrom}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

def process_fasta(input_fasta, output_fasta, max_end_per_chrom, chrom_pattern):
    fasta_seqs = read_fasta(input_fasta)
    output_seqs = OrderedDict()
    for chrom, seq in fasta_seqs.items():
        if not chrom_pattern.match(chrom):
            continue  # Skip non-main chromosomes
        if chrom in max_end_per_chrom:
            max_end = max_end_per_chrom[chrom]
            truncate_length = max_end + 100
            truncate_length = min(truncate_length, len(seq))
            truncated_seq = seq[:truncate_length]
        else:
            truncated_seq = seq
        output_seqs[chrom] = truncated_seq
    write_fasta(output_seqs, output_fasta)

def main():
    parser = argparse.ArgumentParser(description='Process GFF and FASTA files.')
    parser.add_argument('--gff', required=True, help='Input GFF file')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out-gff', required=True, help='Output GFF file')
    parser.add_argument('--out-fasta', required=True, help='Output FASTA file')
    parser.add_argument('--chrom-pattern', 
                        default=r'^(chr)?(\d+|X|Y|M|MT)$',
                        help='Regex pattern to match main chromosomes (default: %(default)s)')
    args = parser.parse_args()
    
    chrom_pattern = re.compile(args.chrom_pattern, re.IGNORECASE)

    max_end_per_chrom = process_gff(args.gff, args.out_gff, chrom_pattern)
    process_fasta(args.fasta, args.out_fasta, max_end_per_chrom, chrom_pattern)

if __name__ == '__main__':
    main()