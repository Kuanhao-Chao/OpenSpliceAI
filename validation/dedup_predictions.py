#!/usr/bin/env python
"""Deduplicate an OpenSpliceAI `predict` BED file.

When `predict` splits a sequence longer than `--split-threshold` (default 1.5 Mb), adjacent segments
overlap by `flanking_size // 2` bp so predictions stay seamless across the boundary. Positions inside
those overlap zones are therefore emitted TWICE — once per segment — with identical coordinates and
(near-)identical scores. The coordinates are correct; the rows are merely redundant.

This helper collapses duplicate intervals, keeping the maximum score per
(chrom, start, end, strand, label) key. Run it on `donor_predictions.bed` / `acceptor_predictions.bed`
when you split large sequences (e.g. whole chromosomes) and need a unique-per-position BED.

Usage:
    python dedup_predictions.py in.bed [-o out.bed]   # default: in.dedup.bed
"""
import argparse


def dedup(in_path, out_path):
    best = {}          # (chrom,start,end,strand,label) -> (score, full_cols)
    order = []
    total = 0
    with open(in_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            c = line.split("\t")
            if len(c) < 6:
                continue
            total += 1
            chrom, start, end, name, score, strand = c[0], c[1], c[2], c[3], c[4], c[5]
            label = name.rsplit("_", 1)[-1]   # ..._Donor / ..._Acceptor
            key = (chrom, int(start), int(end), strand, label)
            sc = float(score)
            if key not in best:
                order.append(key)
                best[key] = (sc, c)
            elif sc > best[key][0]:
                best[key] = (sc, c)
    order.sort(key=lambda k: (k[0], k[1], k[2]))
    with open(out_path, "w") as out:
        for key in order:
            out.write("\t".join(best[key][1]) + "\n")
    return total, len(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bed")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    out = a.out or a.bed.replace(".bed", ".dedup.bed")
    total, kept = dedup(a.bed, out)
    print(f"{a.bed}: {total} rows -> {kept} unique ({total - kept} duplicates removed) -> {out}")


if __name__ == "__main__":
    main()
