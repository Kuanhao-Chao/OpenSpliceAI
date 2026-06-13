#!/usr/bin/env python
"""Check that predict's BED splice-site calls land on annotated splice sites (both strands).

For each annotated internal splice site (sites.tsv: gene,strand,type,pos 1-based), find the nearest
predict BED call of the matching type/strand and record the signed offset. A 1-bp BED interval
[start,end) represents 1-based position `end`. Reports per-strand recovery within +/-3bp and the
offset distribution — the end-to-end coordinate-mapping check, especially for the minus strand.

Usage: compare_predict_recovery.py <sites.tsv> <acceptor.bed> <donor.bed> [min_score]
"""
import sys
from collections import defaultdict


def load_bed(path, min_score):
    """Return list of (pos_1based, strand, score)."""
    out = []
    with open(path) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            end, score, strand = int(c[2]), float(c[4]), c[5]
            if score >= min_score:
                out.append((end, strand, score))
    return out


def main():
    sites_path, acc_path, don_path = sys.argv[1:4]
    min_score = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    beds = {"acceptor": load_bed(acc_path, min_score), "donor": load_bed(don_path, min_score)}

    sites = []
    with open(sites_path) as fh:
        next(fh)
        for line in fh:
            gene, strand, typ, pos = line.rstrip("\n").split("\t")
            sites.append((gene, strand, typ, int(pos)))

    per_strand = defaultdict(lambda: {"n": 0, "rec": 0, "offsets": []})
    misses = []
    for gene, strand, typ, pos in sites:
        cands = [b for b in beds[typ] if b[1] == strand]
        if not cands:
            per_strand[strand]["n"] += 1
            misses.append((gene, strand, typ, pos, None))
            continue
        nearest = min(cands, key=lambda b: abs(b[0] - pos))
        off = nearest[0] - pos
        s = per_strand[strand]
        s["n"] += 1
        if abs(off) <= 3:
            s["rec"] += 1
            s["offsets"].append(off)
        else:
            misses.append((gene, strand, typ, pos, off))

    print(f"min_score={min_score}; annotated sites={len(sites)}; "
          f"BED acc={len(beds['acceptor'])} don={len(beds['donor'])}")
    for strand in sorted(per_strand):
        s = per_strand[strand]
        offs = s["offsets"]
        hist = defaultdict(int)
        for o in offs:
            hist[o] += 1
        print(f"strand {strand}: recovered {s['rec']}/{s['n']} within +/-3bp "
              f"({100*s['rec']/max(s['n'],1):.1f}%); offset histogram {dict(sorted(hist.items()))}")
    if misses:
        print(f"non-recovered ({len(misses)}):")
        for m in misses[:20]:
            print("   ", m)


if __name__ == "__main__":
    main()
