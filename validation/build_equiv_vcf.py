#!/usr/bin/env python
"""Build a VCF for the OpenSpliceAI-keras vs original-SpliceAI numerical-equivalence test.

We pick real multi-exon genes (both strands) from the SpliceAI grch38 annotation, place variants
AT and NEAR their annotated internal exon boundaries (so the model produces strong, non-trivial
splice signal), plus deep-intronic (far) variants and a set of structural edge cases:
deletion, insertion, multiallelic, MNV (-> '.|...' sentinel), and a deliberate REF mismatch.

REF bases are read straight from the reference FASTA so every REF matches the genome (the only
exception is the intentional ref-mismatch record). The SAME VCF + SAME annotation + SAME reference
are fed to BOTH tools, so any score difference localises to the algorithm, not the inputs.

Usage:
    python build_equiv_vcf.py --ref <genome.fa> --ann <grch38_chr.txt> --out equiv_test.vcf
"""
import argparse
from pyfaidx import Fasta


def complement(b):
    return {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}.get(b.upper(), "N")


def pick_genes(ann_path, n_per_strand=3, min_exons=4, chroms=("chr1", "chr2", "chr3")):
    """Return up to n_per_strand multi-exon genes per strand from the chosen chromosomes."""
    out = {"+": [], "-": []}
    with open(ann_path) as fh:
        next(fh)  # header
        for line in fh:
            name, chrom, strand, tx_s, tx_e, ex_s, ex_e = line.rstrip("\n").split("\t")
            if chrom not in chroms:
                continue
            estarts = [int(x) for x in ex_s.split(",") if x]
            eends = [int(x) for x in ex_e.split(",") if x]
            if len(estarts) < min_exons:
                continue
            if len(out[strand]) < n_per_strand:
                out[strand].append((name, chrom, strand, estarts, eends))
            if len(out["+"]) >= n_per_strand and len(out["-"]) >= n_per_strand:
                break
    return out["+"] + out["-"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fa = Fasta(a.ref, sequence_always_upper=True, rebuild=False)
    genes = pick_genes(a.ann)
    print("selected genes:", [(g[0], g[2], len(g[3])) for g in genes])

    rows = []  # (chrom, pos, ref, alt, note)

    def ref_at(chrom, pos1):  # 1-based -> base
        return str(fa[chrom][pos1 - 1:pos1])

    for name, chrom, strand, estarts, eends in genes:
        # internal exon boundaries (skip the very first start / very last end)
        # annotation values are 0-based; +1 -> 1-based genomic coordinate of the boundary base.
        donor = eends[0] + 1          # boundary at end of first exon (1-based)
        acceptor = estarts[1] + 1     # boundary at start of second exon (1-based)
        for site, kind in ((donor, "donor_bdry"), (acceptor, "acceptor_bdry")):
            for off in (0, 2):        # at the site and 2bp into it
                p = site + off
                r = ref_at(chrom, p)
                if r in ("A", "C", "G", "T"):
                    alt = {"A": "G", "C": "T", "G": "A", "T": "C"}[r]
                    rows.append((chrom, p, r, alt, f"{name}:{kind}+{off}({strand})"))
        # deep-intronic far variant (midway into the first intron)
        far = (eends[0] + estarts[1]) // 2
        r = ref_at(chrom, far)
        if r in ("A", "C", "G", "T"):
            alt = {"A": "G", "C": "T", "G": "A", "T": "C"}[r]
            rows.append((chrom, far, r, alt, f"{name}:far_intron({strand})"))

    # structural edge cases anchored on the first gene's donor boundary
    g0 = genes[0]
    chrom0 = g0[1]
    d0 = g0[4][0] + 1
    # deletion (2bp): REF = 2 genome bases, ALT = first base
    r2 = ref_at(chrom0, d0) + ref_at(chrom0, d0 + 1)
    rows.append((chrom0, d0, r2, r2[0], "deletion_2bp"))
    # insertion (REF=1 base, ALT=base+2 inserted)
    r1 = ref_at(chrom0, d0 + 5)
    rows.append((chrom0, d0 + 5, r1, r1 + "AC", "insertion_2bp"))
    # multiallelic SNV (two ALTs)
    rm = ref_at(chrom0, d0 + 8)
    alts = ",".join(b for b in "ACGT" if b != rm)[:3]  # up to 3 alt alleles
    rows.append((chrom0, d0 + 8, rm, alts, "multiallelic"))
    # MNV (len(ref)>1 and len(alt)>1) -> sentinel '.|...'
    rmn = ref_at(chrom0, d0 + 10) + ref_at(chrom0, d0 + 11)
    rows.append((chrom0, d0 + 10, rmn, "AC", "mnv_sentinel"))
    # deliberate REF mismatch (REF != genome) -> both tools skip
    rmm = ref_at(chrom0, d0 + 14)
    wrong = "A" if rmm != "A" else "C"
    rows.append((chrom0, d0 + 14, wrong, "G", "ref_mismatch_skip"))

    # sort by (chrom, pos), dedup identical (chrom,pos,ref,alt)
    seen, uniq = set(), []
    for row in sorted(rows, key=lambda x: (x[0], x[1])):
        k = (row[0], row[1], row[2], row[3])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(row)

    with open(a.out, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write('##INFO=<ID=NOTE,Number=1,Type=String,Description="equivalence-test annotation">\n')
        for chrom in sorted({r[0] for r in uniq}):
            out.write(f"##contig=<ID={chrom}>\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, ref, alt, note in uniq:
            out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\tNOTE={note}\n")

    print(f"wrote {len(uniq)} variant records -> {a.out}")
    for r in uniq:
        print("  ", r)


if __name__ == "__main__":
    main()
