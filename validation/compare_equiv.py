#!/usr/bin/env python
"""Field-level comparison of original-SpliceAI vs OpenSpliceAI-keras VCF annotations.

Both INFO fields share the format ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL.
We key each '|'-annotation by (CHROM,POS,REF,ALT,ALLELE,SYMBOL) and compare all 10 fields exactly
(DS as the rounded string, DP as int / '.'). Prints a summary and every mismatch.

Usage: compare_equiv.py orig.vcf os.vcf
Exit code 0 iff every annotation matches on all 10 fields and the key sets are identical.
"""
import sys

FIELDS = ["DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL"]


def parse(path, key):
    """Return {(chrom,pos,ref,alt,allele,symbol): [10 fields]} for INFO tag `key`."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt, info = c[0], c[1], c[3], c[4], c[7]
            val = None
            for kv in info.split(";"):
                if kv.startswith(key + "="):
                    val = kv[len(key) + 1:]
                    break
            if val is None:
                continue
            for anno in val.split(","):
                f = anno.split("|")
                if len(f) != 10:
                    continue
                allele, symbol = f[0], f[1]
                out[(chrom, pos, ref, alt, allele, symbol)] = f[2:]
    return out


def main():
    orig = parse(sys.argv[1], "SpliceAI")
    os_ = parse(sys.argv[2], "OpenSpliceAI")
    ko, ks = set(orig), set(os_)

    only_orig = ko - ks
    only_os = ks - ko
    common = ko & ks

    ds_mismatch, dp_mismatch, off_by_001 = [], [], 0
    for k in sorted(common):
        a, b = orig[k], os_[k]
        for i, name in enumerate(FIELDS):
            if a[i] != b[i]:
                if name.startswith("DS"):
                    ds_mismatch.append((k, name, a[i], b[i]))
                    try:
                        if abs(float(a[i]) - float(b[i])) <= 0.01 + 1e-9:
                            off_by_001 += 1
                    except ValueError:
                        pass
                else:
                    dp_mismatch.append((k, name, a[i], b[i]))

    n_anno = len(common)
    n_ds_fields = n_anno * 4
    n_dp_fields = n_anno * 4
    print(f"annotations: orig={len(orig)} os={len(os_)} common={n_anno}")
    print(f"key-set: only_orig={len(only_orig)} only_os={len(only_os)}")
    print(f"DS fields: {n_ds_fields - len(ds_mismatch)}/{n_ds_fields} exact "
          f"({100*(n_ds_fields-len(ds_mismatch))/max(n_ds_fields,1):.2f}%); "
          f"off-by-0.01: {off_by_001}")
    print(f"DP fields: {n_dp_fields - len(dp_mismatch)}/{n_dp_fields} exact "
          f"({100*(n_dp_fields-len(dp_mismatch))/max(n_dp_fields,1):.2f}%)")

    if only_orig:
        print("ONLY IN ORIG:", list(only_orig)[:10])
    if only_os:
        print("ONLY IN OS:", list(only_os)[:10])
    for k, name, va, vb in (ds_mismatch + dp_mismatch)[:40]:
        print(f"  MISMATCH {k} {name}: orig={va} os={vb}")

    ok = not (only_orig or only_os or ds_mismatch or dp_mismatch)
    print("VERDICT:", "EXACT MATCH" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
