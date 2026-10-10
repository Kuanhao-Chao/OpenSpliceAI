"""Paged FM index for static human sequence search.

Native suffix-array construction is optional preparation-only work. Queries
need no server computation. Each gzip page has absolute rank checkpoints and
SA samples every 32 reference bases; LF locate therefore takes at most 31 steps.
The fixed-width page directory permits 44-byte HTTP range reads.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
from pathlib import Path

from .build import atomic, file_sha
from .format import canonical_json, sha256
from .reference import Reference

PAGE_BP = 4096
SAMPLE_RATE = 32
DIRECTORY_ENTRY = 44


def fm_pages(sequence: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import pydivsufsort

    if "$" in sequence:
        raise ValueError("reference contains FM sentinel")
    text = (sequence.upper() + "$").encode("ascii")
    alphabet = "".join(sorted(set(text.decode("ascii"))))
    sa = pydivsufsort.divsufsort(text)
    codes = np.frombuffer(text, dtype=np.uint8)
    bwt = codes[(sa.astype(np.int64) - 1) % len(codes)]
    counts = {c: int(np.count_nonzero(codes == ord(c))) for c in alphabet}
    cumulative, running = {}, 0
    for char in alphabet:
        cumulative[char] = running
        running += counts[char]
    metadata = {"length": len(sequence), "rows": len(text), "alphabet": alphabet,
                "cumulative": cumulative, "counts": counts, "pageBp": PAGE_BP,
                "sampleRate": SAMPLE_RATE, "directoryEntryBytes": DIRECTORY_ENTRY}
    yield metadata
    rank = [0] * len(alphabet)
    for start in range(0, len(bwt), PAGE_BP):
        fragment = bwt[start:start + PAGE_BP]
        suffixes = sa[start:start + PAGE_BP]
        samples = np.where(suffixes % SAMPLE_RATE == 0, suffixes, 0xffffffff).astype("<u4")
        raw = struct.pack("<I" + "I" * len(rank), len(fragment), *rank) + fragment.tobytes() + samples.tobytes()
        for i, char in enumerate(alphabet):
            rank[i] += int(np.count_nonzero(fragment == ord(char)))
        yield gzip.compress(raw, compresslevel=6, mtime=0)


def build_reference(config_path, output, *, with_search=False, contig_names=None):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_identity = file_sha(config["reference"])
    if config.get("reference_sha256") and source_identity != config["reference_sha256"]:
        raise ValueError("reference SHA-256 does not agree with frozen configuration")
    ref = Reference(config["reference"])
    result = {}
    try:
        for chrom, (length, *_) in ref.contigs.items():
            if contig_names is not None and chrom not in contig_names:
                continue
            receipt_path = output / f"reference-{chrom}.receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                valid = receipt["sourceIdentity"] == source_identity and all(
                    (output / f["path"]).is_file() and file_sha(output / f["path"]) == f["sha256"] for f in receipt["files"]
                )
                if not valid:
                    raise ValueError(f"reference resume content mismatch: {chrom}")
                if not with_search or "search" in receipt:
                    result[chrom] = receipt
                    continue
            # Native SA work needs one contig in memory, never the entire genome.
            sequence = ref.read(chrom, 0, length)
            step = 16384
            name = f"reference-{chrom}.pack"
            directory_name = f"reference-{chrom}.idx"
            files = []
            with (output / (name + ".partial")).open("wb") as pack, (output / (directory_name + ".partial")).open("wb") as index:
                for start in range(0, length, step):
                    data = gzip.compress(sequence[start:start + step].encode(), compresslevel=6, mtime=0)
                    index.write(struct.pack("<QI", pack.tell(), len(data)) + hashlib.sha256(data).digest())
                    pack.write(data)
            for path in (name, directory_name):
                (output / (path + ".partial")).replace(output / path)
                files.append({"path": path, "bytes": (output / path).stat().st_size, "sha256": file_sha(output / path)})
            receipt = {"sourceIdentity": source_identity,
                       "reference": {"path": name, "index": directory_name, "blockBp": step, "length": length},
                       "files": files}
            if with_search:
                iterator = fm_pages(sequence)
                metadata = next(iterator)
                search_name, search_index = f"search-{chrom}.pack", f"search-{chrom}.idx"
                with (output / (search_name + ".partial")).open("wb") as pack, (output / (search_index + ".partial")).open("wb") as index:
                    for data in iterator:
                        index.write(struct.pack("<QI", pack.tell(), len(data)) + hashlib.sha256(data).digest())
                        pack.write(data)
                for path in (search_name, search_index):
                    (output / (path + ".partial")).replace(output / path)
                    receipt["files"].append({"path": path, "bytes": (output / path).stat().st_size, "sha256": file_sha(output / path)})
                metadata.update({"path": search_name, "index": search_index})
                receipt["search"] = metadata
            atomic(receipt_path, canonical_json(receipt))
            result[chrom] = receipt
            print(json.dumps({"reference": chrom, "length": length, "search": with_search,
                              "bytes": sum(f["bytes"] for f in receipt["files"])}), flush=True)
    finally:
        ref.close()
    return result


def attach_reference(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for contig in manifest["contigs"]:
        path = output / f"reference-{contig['name']}.receipt.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        if receipt["sourceIdentity"] != manifest["referenceSha256"]:
            raise ValueError("reference provenance differs from score snapshot")
        contig["reference"] = receipt["reference"]
        if "search" in receipt:
            contig["search"] = receipt["search"]
        for descriptor in receipt["files"]:
            if not any(f["path"] == descriptor["path"] for f in manifest["files"]):
                manifest["files"].append(descriptor)
    atomic(output / "manifest.json", canonical_json(manifest))
