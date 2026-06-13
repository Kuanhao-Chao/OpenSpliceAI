#!/usr/bin/env python
"""Build a small region FASTA + matching GFF for the predict coordinate-recovery test.

We extract chr1:1..END from the reference (named 'chr1' so GFF absolute coords resolve), and write a
GFF with `gene` + `exon` features for a handful of real genes (both strands) taken from the SpliceAI
grch38 annotation. predict's `process_gff` extracts each gene region (reverse-complementing the minus
strand), so the resulting BED lets us check that annotated donor/acceptor sites are recovered at the
correct genome coordinates on BOTH strands (the key minus-strand coordinate-mapping check).

Also emits sites.tsv: gene, strand, type(donor/acceptor), genomic_pos(1-based) for every internal
splice boundary — the ground truth for the recovery check.

Usage: build_predict_region.py --ref <genome.fa> --ann <grch38_chr.txt> --genes SAMD11,KLHL17,... \
           --end 970000 --outdir <dir>
"""
import argparse
import os
from pyfaidx import Fasta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--genes", required=True, help="comma-separated gene names")
    ap.add_argument("--chrom", default="chr1")
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.genes.split(","))

    # collect chosen gene rows
    rows = {}
    with open(a.ann) as fh:
        next(fh)
        for line in fh:
            name, chrom, strand, tx_s, tx_e, ex_s, ex_e = line.rstrip("\n").split("\t")
            if name in want and chrom == a.chrom:
                estarts = [int(x) for x in ex_s.split(",") if x]
                eends = [int(x) for x in ex_e.split(",") if x]
                rows[name] = (chrom, strand, int(tx_s), int(tx_e), estarts, eends)

    # region FASTA: chr1:1..end, named 'chr1'
    fa = Fasta(a.ref, sequence_always_upper=True, rebuild=False)
    seq = str(fa[a.chrom][0:a.end])
    fapath = os.path.join(a.outdir, "region.fa")
    with open(fapath, "w") as out:
        out.write(f">{a.chrom}\n")
        for i in range(0, len(seq), 80):
            out.write(seq[i:i + 80] + "\n")

    # GFF (1-based, inclusive). Emit gene + exon features.
    gffpath = os.path.join(a.outdir, "region.gff")
    sites = []
    with open(gffpath, "w") as out:
        out.write("##gff-version 3\n")
        for name, (chrom, strand, tx_s, tx_e, estarts, eends) in rows.items():
            g_start, g_end = tx_s + 1, tx_e  # annotation TX_START is 0-based -> +1
            out.write(f"{chrom}\tval\tgene\t{g_start}\t{g_end}\t.\t{strand}\t.\tID={name}\n")
            for es, ee in zip(estarts, eends):
                out.write(f"{chrom}\tval\texon\t{es + 1}\t{ee}\t.\t{strand}\t.\tParent={name}\n")
            # internal splice boundaries (genomic, 1-based): donor at exon end, acceptor at exon start
            boundaries = sorted(set(estarts[1:] + eends[:-1]))
            for es in estarts[1:]:
                sites.append((name, strand, "acceptor", es + 1))
            for ee in eends[:-1]:
                sites.append((name, strand, "donor", ee))
            _ = boundaries

    with open(os.path.join(a.outdir, "sites.tsv"), "w") as out:
        out.write("gene\tstrand\ttype\tpos\n")
        for name, strand, typ, pos in sorted(sites, key=lambda x: x[3]):
            out.write(f"{name}\t{strand}\t{typ}\t{pos}\n")

    print(f"genes: {list(rows)}")
    print(f"region {a.chrom}:1-{a.end} ({len(seq)} bp) -> {fapath}")
    print(f"gff -> {gffpath}; {len(sites)} annotated internal splice sites -> sites.tsv")


if __name__ == "__main__":
    main()
