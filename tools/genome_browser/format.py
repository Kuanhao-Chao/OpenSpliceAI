"""OSGB1 columnar blocks. Coordinates are zero-based, half-open throughout.

Every source row survives. Repeated annotations remain distinct; the UI groups
identical values while retaining row identities. Five-decimal DS are integers,
so even very small deltas survive browser rendering and export.
"""
from __future__ import annotations

import array
import gzip
import hashlib
import json
import math
import struct
import sys
from decimal import Decimal, InvalidOperation

MAGIC = b"OSGB1\0\0\0"
BASES = "ACGTNRYWSKMBDHV."
EVENTS = ("AG", "AL", "DG", "DL")
SCALE = 100000
BLOCK_BP = 16384
INDEX_BP = 16777216
SUMMARY_BP = 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, sort_keys=True).encode()


def ds_integer(value: str) -> int:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid DS: {value!r}") from exc
    if not number.is_finite() or not 0 <= number <= 1 or number * SCALE != (number * SCALE).to_integral_value():
        raise ValueError(f"DS outside [0,1] or exceeds five decimals: {value!r}")
    return int(number * SCALE)


def parse_vcf(line: str, chunk: int, ordinal: int) -> dict:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 8:
        raise ValueError(f"VCF row has {len(fields)} columns at {chunk}:{ordinal}")
    chrom, pos, _, ref, alt, _, _, info = fields[:8]
    pos0 = int(pos) - 1
    if pos0 < 0 or len(ref) != 1 or ref not in BASES:
        raise ValueError(f"invalid SNV position/REF at {chunk}:{ordinal}")
    alts = alt.split(",")
    if any(len(a) != 1 or a not in BASES or a == "." for a in alts):
        raise ValueError(f"non-SNV ALT at {chunk}:{ordinal}")
    annotations = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key not in ("SpliceAI", "OpenSpliceAI"):
            continue
        entries = []
        if value not in (".", ""):
            for entry in value.split(","):
                cells = entry.split("|")
                if len(cells) != 10 or cells[0] not in alts or not cells[1]:
                    raise ValueError(f"invalid {key} entry at {chunk}:{ordinal}: {entry!r}")
                # A wholly missing annotation is absent, never an artificial zero.
                if all(c == "." for c in cells[2:]):
                    continue
                ds = [ds_integer(c) for c in cells[2:6]]
                dp = [int(c) for c in cells[6:10]]
                if any(not -32768 <= n <= 32767 for n in dp):
                    raise ValueError("DP exceeds signed int16; change schema rather than truncate")
                entries.append({"alt": cells[0], "gene": cells[1], "ds": ds, "dp": dp})
        annotations.setdefault(key, []).extend(entries)
    return {"chrom": chrom, "pos": pos0, "ref": ref, "alts": alts,
            "chunk": chunk, "ordinal": ordinal, "annotations": annotations}


def _column(kind: str, values: list[int]) -> bytes:
    result = array.array(kind, values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def encode_block(chrom: str, start: int, rows: list[dict]) -> bytes:
    genes = sorted({e["gene"] for row in rows for entries in row["models"].values() for e in entries})
    gene_ids = {g: i for i, g in enumerate(genes)}
    columns = []
    payload = bytearray()

    def add(name, kind, values):
        data = _column(kind, values)
        columns.append({"name": name, "type": kind, "count": len(values),
                        "offset": len(payload), "bytes": len(data)})
        payload.extend(data)

    for name, kind, values in (
        ("pos", "I", [r["pos"] for r in rows]),
        ("ref", "B", [BASES.index(r["ref"]) for r in rows]),
        ("alt", "B", [BASES.index(r["alt"]) for r in rows]),
        ("chunk", "I", [r["chunk"] for r in rows]),
        ("ordinal", "I", [r["ordinal"] for r in rows]),
        ("flags", "B", [r["flags"] for r in rows]),
    ):
        add(name, kind, values)
    for model in ("r10", "r13", "baseline"):
        entries = [(i, e) for i, row in enumerate(rows) for e in row["models"].get(model, [])]
        add(model + ".row", "I", [i for i, _ in entries])
        add(model + ".gene", "I", [gene_ids[e["gene"]] for _, e in entries])
        for event, channel in enumerate(EVENTS):
            add(model + ".DS_" + channel, "I", [e["ds"][event] for _, e in entries])
            add(model + ".DP_" + channel, "h", [e["dp"][event] for _, e in entries])
    header = canonical_json({"chrom": chrom, "start": start, "end": start + BLOCK_BP,
                             "rows": len(rows), "genes": genes, "columns": columns, "scale": SCALE})
    return MAGIC + struct.pack("<I", len(header)) + header + payload


def decode_block(data: bytes) -> tuple[dict, dict]:
    if data[:8] != MAGIC:
        raise ValueError("unsupported genome-browser block")
    length = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[12:12 + length])
    payload = memoryview(data)[12 + length:]
    columns = {}
    for col in header["columns"]:
        width = struct.calcsize("<" + col["type"])
        if col["bytes"] != width * col["count"] or col["offset"] + col["bytes"] > len(payload):
            raise ValueError("truncated column")
        columns[col["name"]] = list(struct.unpack_from("<" + col["type"] * col["count"], payload, col["offset"]))
    return header, columns


def summarize(rows: list[dict], start: int) -> list[dict]:
    """Counts are source occurrences/annotations, never falsely called unique SNVs."""
    bins = {}
    for row in rows:
        bin_start = (row["pos"] // SUMMARY_BP) * SUMMARY_BP
        item = bins.setdefault(bin_start, {"start": bin_start, "end": bin_start + SUMMARY_BP,
                                          "rows": 0, "acceptedR13": 0, "refMismatch": 0, "models": {}})
        item["rows"] += 1
        item["acceptedR13"] += bool(row["flags"] & 2)
        item["refMismatch"] += bool(row["flags"] & 1)
        for model, entries in row["models"].items():
            if not entries:
                continue
            stats = item["models"].setdefault(model, {"annotations": 0, "max": [0] * 4,
                                                       "sum": [0] * 4, "zero": 0})
            for entry in entries:
                stats["annotations"] += 1
                stats["zero"] += not any(entry["ds"])
                for i, value in enumerate(entry["ds"]):
                    stats["max"][i] = max(stats["max"][i], value)
                    stats["sum"][i] += value
    return [bins[k] for k in sorted(bins)]


def merge_summaries(summaries: list[list[dict]]) -> list[dict]:
    bins = {}
    for fragment in summaries:
        for source in fragment:
            item = bins.setdefault(source["start"], {"start": source["start"], "end": source["end"],
                                                    "rows": 0, "acceptedR13": 0, "refMismatch": 0, "models": {}})
            for field in ("rows", "acceptedR13", "refMismatch"):
                item[field] += source[field]
            for model, stats in source["models"].items():
                target = item["models"].setdefault(model, {"annotations": 0, "max": [0] * 4,
                                                          "sum": [0] * 4, "zero": 0})
                target["annotations"] += stats["annotations"]
                target["zero"] += stats["zero"]
                for i in range(4):
                    target["max"][i] = max(target["max"][i], stats["max"][i])
                    target["sum"][i] += stats["sum"][i]
    return [bins[k] for k in sorted(bins)]
