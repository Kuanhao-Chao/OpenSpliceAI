"""Build a small, explicitly labelled review dataset from real audited VCFs.

Only immutable source-content hashes from the audit manifests are accepted.
The score subset has OR4F5, minus-strand OR4F16, SAMD11 and the real alternate-
contig duplicate/REF-mismatch cases. Its reference access is intentionally
limited, so it cannot be mistaken for a completed human data publication.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

from .build import atomic, file_sha, finalize, prepare
from .format import canonical_json
from .reference import Reference


def make_config(root):
    root = Path(root)
    return {"id": "grch38-scored-review-20261010", "label": "GRCh38 real scored review subset",
            "scope": "review-subset", "shard_size": 2, "chunks": [1, 2, 17208, 17209],
            "r10_manifest": str(root / "results/full_snv_concordance/cpu_completion_20260913/manifests/rs10_complete.tsv"),
            "r13_manifest": str(root / "results/full_snv_concordance/hybrid_finish_20261009/manifests/audit_initial_000/accepted.tsv"),
            "reference": "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna",
            "reference_sha256": "c5e1eb75150eac186eec9fa50af1613b991a9466247321dc03f9e155b6ed59d4",
            "annotation": "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/data/grch38_chr.txt",
            "annotation_sha256": "e1aad08cf529cbf1310eac0f20344c23a4af7f54b0608fdbe94f2ce1af42d3b4",
            "regions": [{"chrom": "chr1", "start": 68990, "end": 70110},
                        {"chrom": "chr1", "start": 685500, "end": 686900},
                        {"chrom": "chr1", "start": 925400, "end": 930500},
                        {"chrom": "chr2_KI270773v1_alt", "start": 18500, "end": 19400}],
            "default_view": {"chrom": "chr1", "start": 925850, "end": 926130},
            "r13_evidence": "Frozen accepted audit from 2026-10-09; legacy annotations are labelled by historical audit evidence, not newly generated receipts.",
            "models": {
                "r10": {"label": "OpenSpliceAI MANE r10 · single model", "precision": 5, "seed": 10,
                        "sha256": "7b0d6af6239e27392226bbada310ee9e2ce0ced149fb6fdff8996c12cba3cc85", "annotation": "MANE scoring table"},
                "r13": {"label": "OpenSpliceAI MANE r13 · preview · single model", "precision": 5, "seed": 13,
                        "sha256": "bae2756b452f31d45b74604816561da79284d791be37f467491a68342c819b20", "annotation": "MANE scoring table"},
                "baseline": {"label": "Original masked SpliceAI v1.3", "precision": 2, "annotation": "Original VCF source annotation"}},
            "reference_review_regions": [{"chrom": "chr1", "start": 65536, "end": 81920},
                                         {"chrom": "chr1", "start": 671744, "end": 704512},
                                         {"chrom": "chr1", "start": 917504, "end": 966656},
                                         {"chrom": "chr2_KI270773v1_alt", "start": 16384, "end": 32768}]}


def review_reference(config, output):
    ref = Reference(config["reference"])
    regions = config["reference_review_regions"]
    manifest = json.loads((output / "manifest.json").read_text())
    for chrom in sorted({r["chrom"] for r in regions}):
        ranges = [[r["start"], r["end"]] for r in regions if r["chrom"] == chrom]
        length = ref.contigs[chrom][0]
        name, index_name = f"reference-{chrom}.pack", f"reference-{chrom}.idx"
        directory = bytearray((length + 16383) // 16384 * 44)
        with (output / name).open("wb") as pack:
            for a, b in ranges:
                for start in range(a, min(b, length), 16384):
                    data = gzip.compress(ref.read(chrom, start, min(start + 16384, length)).encode(), mtime=0)
                    entry = struct.pack("<QI", pack.tell(), len(data)) + hashlib.sha256(data).digest()
                    directory[start // 16384 * 44:(start // 16384 + 1) * 44] = entry
                    pack.write(data)
        atomic(output / index_name, directory)
        contig = next(c for c in manifest["contigs"] if c["name"] == chrom)
        contig["reference"] = {"path": name, "index": index_name, "blockBp": 16384, "length": length, "ranges": ranges}
        for file in (name, index_name):
            manifest["files"].append({"path": file, "bytes": (output / file).stat().st_size, "sha256": file_sha(output / file)})
    ref.close()
    atomic(output / "manifest.json", canonical_json(manifest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="docs/genome-browser/public/data/review")
    parser.add_argument("--config-output", default="results/genome_browser/review/config.json")
    args = parser.parse_args()
    config = make_config(args.root)
    atomic(args.config_output, canonical_json(config))
    prepare(args.config_output, args.output)
    finalize(args.config_output, args.output)
    review_reference(config, Path(args.output))
    from .search_index import attach_reference
    attach_reference(args.output)
    # Preparation internals remain in results; public artifacts contain no private paths.
    output = Path(args.output)
    for name in ("preparation.sqlite", "preparation.sqlite-wal", "preparation.sqlite-shm", "configuration.sha256"):
        path = output / name
        if path.exists():
            destination = Path(args.config_output).parent / name
            if destination.exists():
                destination.unlink()
            path.replace(destination)


if __name__ == "__main__":
    main()
