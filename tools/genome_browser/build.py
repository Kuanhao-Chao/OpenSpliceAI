"""Read-only VCF conversion into immutable packed browser snapshots.

Each shard is an independently resumable manifest slice. Chunk-boundary tile
fragments are indexed together and never silently overwrite one another.
SQLite holds the preparation index, avoiding hundreds of thousands of inodes.
"""
from __future__ import annotations

import csv
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .format import (BLOCK_BP, INDEX_BP, SCALE, canonical_json, encode_block,
                     merge_summaries, parse_vcf, sha256, summarize)
from .reference import Reference, genes_from_tsv


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def compressed_json(path, value):
    raw = canonical_json(value)
    data = gzip.compress(raw, compresslevel=6, mtime=0)
    atomic(path, data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data),
            "decodedBytes": len(raw), "decodedSha256": sha256(raw)}


def load_manifest(path):
    with Path(path).open() as handle:
        return {int(row["chunk_id"]): row for row in csv.DictReader(handle, delimiter="\t")}


def accepted(row):
    return row is not None and row.get("state") in ("valid", "accepted") and not row.get("error")


def init_db(path):
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS shards (id INTEGER PRIMARY KEY, identity TEXT, pack TEXT, sha TEXT, bytes INTEGER, records INTEGER, r13_records INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS blocks (shard INTEGER, chrom TEXT, start INTEGER, offset INTEGER, length INTEGER, raw_bytes INTEGER, sha TEXT, summary TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS block_region ON blocks(chrom,start)")
    db.commit()
    return db


def vcf_rows(path, chunk, expected_sha):
    digest = hashlib.sha256()
    ordinal = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.startswith(b"#"):
                continue
            ordinal += 1
            yield parse_vcf(line.decode("utf-8"), chunk, ordinal)
    if not expected_sha or digest.hexdigest() != expected_sha:
        raise ValueError(f"VCF content hash differs from audit manifest: chunk {chunk}")


def prepare(config_path, output, *, shard_size=None, first_shard=0, last_shard=None, shard_step=1, shard_offset=0):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config.get("reference_sha256") and file_sha(config["reference"]) != config["reference_sha256"]:
        raise ValueError("reference SHA-256 does not agree with frozen configuration")
    if config.get("annotation_sha256") and file_sha(config["annotation"]) != config["annotation_sha256"]:
        raise ValueError("annotation SHA-256 does not agree with frozen configuration")
    shard_size = config.get("shard_size", 1000) if shard_size is None else shard_size
    if shard_size != config.get("shard_size", 1000) or shard_size < 1:
        raise ValueError("shard size must match the frozen configuration")
    output.mkdir(parents=True, exist_ok=True)
    config_identity = sha256(canonical_json(config))
    lock = output / "configuration.sha256"
    if lock.exists() and lock.read_text().strip() != config_identity:
        raise ValueError("snapshot configuration changed; use a new output directory")
    if not lock.exists():
        atomic(lock, (config_identity + "\n").encode())
    r10 = load_manifest(config["r10_manifest"])
    r13 = load_manifest(config["r13_manifest"]) if config.get("r13_manifest") else {}
    ids = config.get("chunks") or sorted(r10)
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("chunk IDs must be unique and in source order")
    if any(not accepted(r10.get(i)) for i in ids):
        raise ValueError("r10 conversion requires accepted source-content audit evidence for every chunk")
    ref = Reference(config["reference"])
    db = init_db(output / "preparation.sqlite")
    regions = config.get("regions", [])
    try:
        for shard, offset in enumerate(range(0, len(ids), shard_size)):
            if shard < first_shard or (last_shard is not None and shard > last_shard):
                continue
            if shard % shard_step != shard_offset:
                continue
            chunk_ids = ids[offset:offset + shard_size]
            identity = sha256(canonical_json({"config": config_identity, "chunks": chunk_ids,
                                              "r10": [r10[i]["output_file_sha256"] for i in chunk_ids],
                                              "r13": [r13[i].get("output_file_sha256") if accepted(r13.get(i)) else None for i in chunk_ids]}))
            prior = db.execute("SELECT identity,pack,sha,bytes FROM shards WHERE id=?", (shard,)).fetchone()
            if prior:
                path = output / prior[1]
                if prior[0] != identity or not path.exists() or path.stat().st_size != prior[3] or file_sha(path) != prior[2]:
                    raise ValueError(f"resume verification failed for shard {shard}")
                print(json.dumps({"shard": shard, "state": "verified_resume"}), flush=True)
                continue
            pack_name = f"scores-{shard:04d}.pack"
            pack_path = output / (pack_name + ".partial")
            descriptors = []
            rows, current = [], None
            records, r13_records = 0, 0
            closed, last_pos = set(), {}
            with pack_path.open("wb") as pack:
                def flush():
                    nonlocal rows
                    if not rows:
                        return
                    chrom, start = current
                    raw = encode_block(chrom, start, rows)
                    data = gzip.compress(raw, compresslevel=6, mtime=0)
                    byte_offset = pack.tell()
                    pack.write(data)
                    descriptors.append((shard, chrom, start, byte_offset, len(data), len(raw),
                                        sha256(data), canonical_json(summarize(rows, start)).decode()))
                    rows = []

                for chunk in chunk_ids:
                    m10 = r10[chunk]
                    m13 = r13.get(chunk)
                    has_r13 = accepted(m13)
                    primary = vcf_rows(m10["output_path"], chunk, m10["output_file_sha256"])
                    secondary = vcf_rows(m13["output_path"], chunk, m13["output_file_sha256"]) if has_r13 else None
                    sentinel = object()
                    pairs = itertools.zip_longest(primary, secondary, fillvalue=sentinel) if secondary else ((r, None) for r in primary)
                    seen = 0
                    for a, b in pairs:
                        if a is sentinel or b is sentinel:
                            raise ValueError(f"r10/r13 row counts differ for chunk {chunk}")
                        if b and any(a[field] != b[field] for field in ("chrom", "pos", "ref", "alts")):
                            raise ValueError(f"r10/r13 source keys differ at {chunk}:{a['ordinal']}")
                        seen += 1
                        chrom, pos = a["chrom"], a["pos"]
                        if chrom in last_pos and pos < last_pos[chrom]:
                            raise ValueError("source positions descend within a shard")
                        last_pos[chrom] = pos
                        if regions and not any(r["chrom"] == chrom and r["start"] <= pos < r["end"] for r in regions):
                            continue
                        if chrom not in ref.contigs or pos >= ref.contigs[chrom][0]:
                            raise ValueError(f"source lies outside pinned reference: {chrom}:{pos + 1}")
                        key = chrom, pos // BLOCK_BP * BLOCK_BP
                        if key != current:
                            flush()
                            if current and current[0] != chrom:
                                closed.add(current[0])
                                if chrom in closed:
                                    raise ValueError("source contig reopens within a shard")
                            current = key
                            sequence = ref.read(chrom, key[1], min(key[1] + BLOCK_BP, ref.contigs[chrom][0]))
                        flags = int(sequence[pos - key[1]] != a["ref"]) | (2 if has_r13 else 0)
                        for alt in a["alts"]:
                            models = {
                                "r10": [e for e in a["annotations"].get("OpenSpliceAI", []) if e["alt"] == alt],
                                "baseline": [e for e in a["annotations"].get("SpliceAI", []) if e["alt"] == alt],
                                "r13": [e for e in (b["annotations"].get("OpenSpliceAI", []) if b else []) if e["alt"] == alt],
                            }
                            rows.append({"pos": pos, "ref": a["ref"], "alt": alt, "chunk": chunk,
                                         "ordinal": a["ordinal"], "flags": flags, "models": models})
                            records += 1
                            r13_records += has_r13
                    if seen != int(m10["output_records"]):
                        raise ValueError(f"r10 record count differs from audit for chunk {chunk}")
                    if has_r13 and seen != int(m13["output_records"]):
                        raise ValueError(f"r13 record count differs from audit for chunk {chunk}")
                    if len(rows) > 500000:
                        raise ValueError("block exceeds memory safety bound; choose a smaller block schema")
                flush()
                pack.flush()
                os.fsync(pack.fileno())
            final_pack = output / pack_name
            pack_path.replace(final_pack)
            digest, size = file_sha(final_pack), final_pack.stat().st_size
            with db:
                db.execute("DELETE FROM blocks WHERE shard=?", (shard,))
                db.executemany("INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?)", descriptors)
                db.execute("INSERT INTO shards VALUES (?,?,?,?,?,?,?)", (shard, identity, pack_name, digest, size, records, r13_records))
            print(json.dumps({"shard": shard, "chunks": len(chunk_ids), "sourceOccurrences": records,
                              "acceptedR13Occurrences": r13_records, "bytes": size, "sha256": digest}), flush=True)
    finally:
        db.close()
        ref.close()


def prepare_parallel(config_path, output, *, workers=2, first_shard=0, last_shard=None):
    if not 1 <= workers <= 8:
        raise ValueError("preparation requires 1–8 explicitly allocated CPU workers")
    if workers == 1:
        return prepare(config_path, output, first_shard=first_shard, last_shard=last_shard)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = sha256(canonical_json(json.loads(Path(config_path).read_text())))
    lock = output / "configuration.sha256"
    if lock.exists() and lock.read_text().strip() != identity:
        raise ValueError("snapshot configuration changed")
    if not lock.exists(): atomic(lock, (identity + "\n").encode())
    db = init_db(output / "preparation.sqlite"); db.close()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare, config_path, output, first_shard=first_shard,
                               last_shard=last_shard, shard_step=workers, shard_offset=i) for i in range(workers)]
        for future in futures: future.result()


def finalize(config_path, output, *, base_url="", allow_subset=False):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    db = init_db(output / "preparation.sqlite")
    r10 = load_manifest(config["r10_manifest"])
    ids = config.get("chunks") or sorted(r10)
    shard_size = config.get("shard_size", 1000)
    expected = (len(ids) + shard_size - 1) // shard_size
    shards = db.execute("SELECT id,pack,sha,bytes,records,r13_records FROM shards ORDER BY id").fetchall()
    if not allow_subset and [s[0] for s in shards] != list(range(expected)):
        raise ValueError("not all expected shards have passed conversion; manifest was not published")
    ref = Reference(config["reference"])
    genes = genes_from_tsv(config["annotation"])
    files = []

    def write_json(name, obj):
        descriptor = compressed_json(output / name, obj)
        descriptor["path"] = name
        files.append(descriptor)
        return descriptor

    # Search metadata is small; exon geometry is fetched only for the current contig.
    gene_descriptor = write_json("genes.json.gz", [{**g, "exons": []} for g in genes])
    contigs = []
    for chrom, (length, *_) in ref.contigs.items():
        windows = {}
        totals = []
        for row in db.execute("SELECT start,shard,offset,length,raw_bytes,sha,summary FROM blocks WHERE chrom=? ORDER BY start,shard,offset", (chrom,)):
            start, shard, offset, compressed, raw, digest, summary = row
            item = windows.setdefault(start // INDEX_BP, {"blocks": [], "summaries": []})
            item["blocks"].append({"start": start, "end": min(start + BLOCK_BP, length),
                                   "path": f"scores-{shard:04d}.pack", "offset": offset,
                                   "bytes": compressed, "rawBytes": raw, "sha256": digest})
            item["summaries"].append(json.loads(summary))
        indexes = {}
        for window, value in sorted(windows.items()):
            summaries = merge_summaries(value["summaries"])
            value["summaries"] = summaries
            indexes[str(window)] = write_json(f"index-{chrom}-{window:03d}.json.gz", value)
            coarse = []
            # Overview bins are 1 Mb, with explicit source-occurrence denominators.
            for s in summaries:
                c = dict(s)
                c["start"] = s["start"] // 1048576 * 1048576
                c["end"] = min(c["start"] + 1048576, length)
                coarse.append(c)
            totals.append(coarse)
        contig = {"name": chrom, "length": length, "indexes": indexes,
                  "overview": merge_summaries(totals)}
        features = [g for g in genes if g["chrom"] == chrom]
        if features:
            contig["genes"] = write_json(f"genes-{chrom}.json.gz", features)
        contigs.append(contig)
    ref.close()
    for _, name, digest, size, _, _ in shards:
        files.append({"path": name, "sha256": digest, "bytes": size})
    reference_digest = file_sha(config["reference"])
    annotation_digest = file_sha(config["annotation"])
    for field, actual in (("reference_sha256", reference_digest), ("annotation_sha256", annotation_digest)):
        if config.get(field) and config[field] != actual:
            raise ValueError(f"pinned {field} does not agree")
    overview_descriptor = write_json("overview.json.gz", {c["name"]: c["overview"] for c in contigs if c["overview"]})
    for contig in contigs:
        contig["overview"] = []
    manifest = {"format": "OSGB1", "id": config["id"], "label": config["label"],
                "created": datetime.now(timezone.utc).isoformat(), "assembly": "GRCh38.p14",
                "baseUrl": base_url, "scope": config.get("scope", "source-collection"),
                "defaultModel": "r10", "r13Final": bool(config.get("r13_final", False)),
                "r13Evidence": config.get("r13_evidence", "frozen accepted audit manifest"),
                "referenceSha256": reference_digest, "annotationSha256": annotation_digest,
                "models": config["models"], "scoreScale": SCALE, "blockBp": BLOCK_BP,
                "indexBp": INDEX_BP, "summaryBp": 1024, "genes": gene_descriptor,
                "contigs": contigs, "files": files, "overview": overview_descriptor,
                "preparedShards": len(shards), "expectedShards": expected,
                "sourceOccurrences": sum(s[4] for s in shards),
                "acceptedR13Occurrences": sum(s[5] for s in shards),
                "configSha256": sha256(canonical_json(config)),
                "r10ManifestSha256": file_sha(config["r10_manifest"]),
                "r13ManifestSha256": file_sha(config["r13_manifest"]) if config.get("r13_manifest") else None,
                "defaultView": config.get("default_view", {"chrom": "chr1", "start": 69040, "end": 70060}),
                "sourceCensus": config.get("source_census"),
                "methods": {"distance": 50, "flank": 10000, "mask": 1,
                            "scores": "single-model variant delta, no additional inference",
                            "coordinates": "internal 0-based half-open; displayed POS and affected sites 1-based",
                            "duplicates": "preserved source occurrences; summaries count annotations, not unique SNVs",
                            "missing": "absent predictions and unaccepted r13 chunks are never numeric zeros"}}
    if config.get("r13_final"):
        evidence = config.get("r13_final_audit", {})
        if not evidence.get("path") or not evidence.get("sha256") or file_sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("r13 promotion requires authenticated final audit evidence")
        audit = json.loads(Path(evidence["path"]).read_text())
        m13 = load_manifest(config.get("r13_manifest", ""))
        census = config.get("r13_distinct_scored_census", {})
        if (not audit.get("full_content") or not audit.get("completion_gate_passed") or
                not all(accepted(m13.get(i)) for i in ids) or len(ids) != 100000 or
                not census.get("path") or file_sha(census["path"]) != census.get("sha256") or
                not json.loads(Path(census["path"]).read_text()).get("passed")):
            raise ValueError("r13 promotion requires the full-domain content/provenance audit and passed distinct-scored census")
    if len(shards) != expected:
        manifest["scope"] = "preparation-subset"
    atomic(output / "manifest.json", canonical_json(manifest))
    db.close()
    return manifest
