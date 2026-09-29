"""Auditing and map/reduce workflows used by the command-line interface."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Optional, Sequence

from .aggregate import Aggregate, AnalysisConfig
from .sites import SiteIndex
from .vcf import (
    VariantGroup,
    canonical_info,
    has_final_newline,
    inspect_header,
    iter_variant_groups,
    iter_vcf_records,
    merge_group_streams,
    parse_annotations,
    stream_with_sided_edges,
)


@dataclass(frozen=True)
class PairRow:
    chunk_id: str
    values: Mapping[str, str]


PROVENANCE_CONTRACT = "audited-vcf-snapshot-v1"
FINALITY_CONTRACT = "explicit-reducer-finality-v1"

_AUDIT_INTEGER_FIELDS = (
    "source_bytes",
    "source_mtime_ns",
    "source_rows",
    "prediction_bytes",
    "prediction_mtime_ns",
    "prediction_rows",
)
_AUDIT_DIGEST_FIELDS = (
    "source_digest",
    "prediction_source_digest",
    "input_file_sha256",
    "output_file_sha256",
)
_AUDIT_SNAPSHOT_FIELDS = _AUDIT_INTEGER_FIELDS + _AUDIT_DIGEST_FIELDS
_OUTPUT_PROVENANCE_CLASSES = {
    "receipt_bound",
    "receipt_invalid",
    "legacy_log_supported",
    "legacy_unprovenanced",
}

# The scoring campaign's deep auditor and this analysis package use different
# historical names for the same immutable snapshot fields.  Keep the emitted
# pair schema canonical (source_*/prediction_*) while accepting both forms at
# the integration boundary.  A field may have more than one alias so old
# campaign manifests remain readable without weakening the required snapshot
# checks.
_AUDIT_FIELD_ALIASES = {
    "source_bytes": ("source_bytes", "input_size"),
    "source_mtime_ns": ("source_mtime_ns", "input_mtime_ns"),
    "source_rows": ("source_rows", "input_records"),
    "prediction_bytes": ("prediction_bytes", "output_size"),
    "prediction_mtime_ns": ("prediction_mtime_ns", "output_mtime_ns"),
    "prediction_rows": ("prediction_rows", "output_records"),
    "source_digest": ("source_digest", "input_source_digest"),
    "prediction_source_digest": (
        "prediction_source_digest",
        "output_source_digest",
    ),
    "input_file_sha256": ("input_file_sha256",),
    "output_file_sha256": ("output_file_sha256",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_tsv(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_pairs_file(
    path: str | Path,
    required_columns: Sequence[str],
    task_index: Optional[int] = None,
    chunks_per_task: int = 250,
) -> list[PairRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = set(required_columns) - fields
        if missing:
            raise ValueError(f"{path}: missing TSV columns: {', '.join(sorted(missing))}")
        rows = [
            PairRow(str(row.get("chunk_id", index + 1)), {k: str(v) for k, v in row.items()})
            for index, row in enumerate(reader)
            if row
        ]
    if task_index is None:
        return rows
    if task_index < 0 or chunks_per_task < 1:
        raise ValueError("task_index must be nonnegative and chunks_per_task positive")
    start = task_index * chunks_per_task
    return rows[start : start + chunks_per_task]


def _manifest_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else payload.get("chunks")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: JSON manifest must contain a chunks list")
        return [{str(k): v for k, v in row.items()} for row in rows]
    with path.open("r", encoding="utf-8", newline="") as handle:
        delimiter = "\t" if path.suffix.lower() != ".csv" else ","
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _first_value(row: Mapping, names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _serialized_manifest_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _output_provenance_fields(row: Mapping) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_name, value in row.items():
        name = str(raw_name)
        lowered = name.lower()
        if value in (None, ""):
            continue
        if (
            lowered == "output_provenance_class"
            or lowered == "provenance_state"
            or "receipt" in lowered
            or ("log" in lowered and "evidence" in lowered)
        ):
            result[name] = _serialized_manifest_value(value)
    # The scoring auditor names this classification ``provenance_state``;
    # accept an explicit analysis-facing alias too, but never let the alias
    # contradict the audited value.
    audited_state = result.get("provenance_state", "").lower()
    explicit_class = result.get("output_provenance_class", "").lower()
    if audited_state and explicit_class and audited_state != explicit_class:
        raise ValueError(
            "output_provenance_class contradicts audited provenance_state: "
            f"{explicit_class!r} != {audited_state!r}"
        )
    provenance_class = explicit_class or audited_state or "legacy_unprovenanced"
    if provenance_class not in _OUTPUT_PROVENANCE_CLASSES:
        raise ValueError(
            f"unsupported output_provenance_class: {provenance_class!r}"
        )
    result["output_provenance_class"] = provenance_class
    return result


def _chunk_sort_key(value: str) -> tuple:
    import re

    match = re.search(r"(\d+)$", str(value))
    return (0, int(match.group(1))) if match else (1, str(value))


def _normalized_audit_snapshot(
    row: Mapping, path: Path, chunk_id: str
) -> dict[str, str | int]:
    snapshot: dict[str, str | int] = {}
    for field in _AUDIT_INTEGER_FIELDS:
        value = _first_value(row, _AUDIT_FIELD_ALIASES[field])
        if value in (None, ""):
            raise ValueError(
                f"{path}: accepted chunk {chunk_id} lacks audited field {field}"
            )
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: accepted chunk {chunk_id} has invalid {field}: {value!r}"
            ) from exc
        if normalized < 0:
            raise ValueError(
                f"{path}: accepted chunk {chunk_id} has negative {field}"
            )
        snapshot[field] = normalized
    for field in _AUDIT_DIGEST_FIELDS:
        value = _first_value(row, _AUDIT_FIELD_ALIASES[field]).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(
                f"{path}: accepted chunk {chunk_id} has invalid {field}"
            )
        snapshot[field] = value
    if snapshot["source_rows"] != snapshot["prediction_rows"]:
        raise ValueError(
            f"{path}: accepted chunk {chunk_id} does not preserve the audited record count"
        )
    if snapshot["source_digest"] != snapshot["prediction_source_digest"]:
        raise ValueError(
            f"{path}: accepted chunk {chunk_id} does not preserve the audited source digest"
        )
    return snapshot


def _valid_manifest_index(
    path: str | Path, accepted_states: Sequence[str], manifest_sha256: str
) -> dict[str, dict]:
    path = Path(path)
    accepted = {state.lower() for state in accepted_states}
    index: dict[str, dict] = {}
    for row in _manifest_rows(path):
        state = _first_value(row, ("state", "status", "validation_state")).lower()
        if state not in accepted:
            continue
        chunk_id = _first_value(row, ("chunk_id", "chunk", "id", "file_id"))
        source = _first_value(
            row, ("source_vcf", "input_vcf", "source_path", "input_path")
        )
        prediction = _first_value(
            row,
            ("prediction_vcf", "output_vcf", "prediction_path", "output_path", "path"),
        )
        if not chunk_id or not source or not prediction:
            raise ValueError(
                f"{path}: accepted manifest row lacks chunk ID, source path, or prediction path: {row}"
            )
        if chunk_id in index:
            raise ValueError(f"{path}: duplicate valid chunk ID {chunk_id}")
        index[chunk_id] = {
            "chunk_id": chunk_id,
            "source_vcf": source,
            "prediction_vcf": prediction,
            "seed": _first_value(row, ("seed", "model_seed")),
            "manifest_sha256": manifest_sha256,
            **_normalized_audit_snapshot(row, path, chunk_id),
            **_output_provenance_fields(row),
        }
    return index


def _load_valid_manifest(
    path: str | Path, accepted_states: Sequence[str]
) -> tuple[dict[str, dict], str]:
    path = Path(path)
    before = path.stat()
    manifest_sha256 = sha256_file(path)
    index = _valid_manifest_index(path, accepted_states, manifest_sha256)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: audit manifest changed while pairs were being built")
    if sha256_file(path) != manifest_sha256:
        raise ValueError(f"{path}: audit manifest digest changed while pairs were being built")
    return index, manifest_sha256


def build_pairs_files(
    left_manifest: str | Path,
    concordance_output: str | Path,
    right_manifest: Optional[str | Path] = None,
    seeds_output: Optional[str | Path] = None,
    accepted_states: Sequence[str] = ("valid",),
    require_left_count: Optional[int] = None,
    require_overlap_count: Optional[int] = None,
) -> dict:
    """Build deterministic mapper inputs only from audit-approved chunks."""

    left, left_manifest_sha256 = _load_valid_manifest(left_manifest, accepted_states)
    if require_left_count is not None and len(left) != require_left_count:
        raise ValueError(
            f"left valid count is {len(left)}, expected exactly {require_left_count}"
        )
    concordance_output = Path(concordance_output)
    concordance_base_fields = (
        "chunk_id",
        "source_vcf",
        "prediction_vcf",
        "seed",
        *_AUDIT_SNAPSHOT_FIELDS,
        "manifest_sha256",
    )
    concordance_optional_fields = sorted(
        {
            field
            for audit_row in left.values()
            for field in audit_row
            if field not in concordance_base_fields
        }
    )
    concordance_fields = (*concordance_base_fields, *concordance_optional_fields)
    _atomic_write_tsv(
        concordance_output,
        concordance_fields,
        (left[chunk_id] for chunk_id in sorted(left, key=_chunk_sort_key)),
    )
    result = {
        "schema_version": 2,
        "kind": "pairs-build",
        "provenance_contract": PROVENANCE_CONTRACT,
        "created_at": utc_now(),
        "left_manifest": str(Path(left_manifest).resolve()),
        "left_manifest_sha256": left_manifest_sha256,
        "accepted_states": list(accepted_states),
        "concordance_pairs": str(concordance_output.resolve()),
        "concordance_pairs_sha256": sha256_file(concordance_output),
        "concordance_count": len(left),
    }
    if right_manifest is not None:
        if seeds_output is None:
            raise ValueError("seeds_output is required with right_manifest")
        right, right_manifest_sha256 = _load_valid_manifest(
            right_manifest, accepted_states
        )
        overlap = left.keys() & right.keys()
        if require_overlap_count is not None and len(overlap) != require_overlap_count:
            raise ValueError(
                f"seed overlap count is {len(overlap)}, expected exactly {require_overlap_count}"
            )
        seeds_output = Path(seeds_output)
        seed_rows = []
        for chunk_id in sorted(overlap, key=_chunk_sort_key):
            left_row = left[chunk_id]
            right_row = right[chunk_id]
            if (
                left_row["source_rows"] != right_row["source_rows"]
                or left_row["source_digest"] != right_row["source_digest"]
            ):
                raise ValueError(
                    f"source VCF identity mismatch for overlapping chunk {chunk_id}"
                )
            seed_row: dict[str, str | int] = {
                "chunk_id": chunk_id,
                "left_source_vcf": left_row["source_vcf"],
                "left_vcf": left_row["prediction_vcf"],
                "right_source_vcf": right_row["source_vcf"],
                "right_vcf": right_row["prediction_vcf"],
            }
            for side, audit_row in (("left", left_row), ("right", right_row)):
                for field in _AUDIT_SNAPSHOT_FIELDS:
                    seed_row[f"{side}_{field}"] = audit_row[field]
                seed_row[f"{side}_manifest_sha256"] = audit_row["manifest_sha256"]
                for field, value in audit_row.items():
                    if field not in {
                        "chunk_id",
                        "source_vcf",
                        "prediction_vcf",
                        "seed",
                        "manifest_sha256",
                        *_AUDIT_SNAPSHOT_FIELDS,
                    }:
                        seed_row[f"{side}_{field}"] = value
            seed_rows.append(seed_row)
        seed_base_fields = (
            "chunk_id",
            "left_source_vcf",
            "left_vcf",
            "right_source_vcf",
            "right_vcf",
            *(f"left_{field}" for field in _AUDIT_SNAPSHOT_FIELDS),
            "left_manifest_sha256",
            *(f"right_{field}" for field in _AUDIT_SNAPSHOT_FIELDS),
            "right_manifest_sha256",
        )
        seed_optional_fields = sorted(
            {
                field
                for seed_row in seed_rows
                for field in seed_row
                if field not in seed_base_fields
            }
        )
        seed_fields = (*seed_base_fields, *seed_optional_fields)
        _atomic_write_tsv(seeds_output, seed_fields, seed_rows)
        result.update(
            {
                "right_manifest": str(Path(right_manifest).resolve()),
                "right_manifest_sha256": right_manifest_sha256,
                "right_valid_count": len(right),
                "seed_overlap_pairs": str(seeds_output.resolve()),
                "seed_overlap_pairs_sha256": sha256_file(seeds_output),
                "seed_overlap_count": len(overlap),
            }
        )
    atomic_write_json(
        concordance_output.with_suffix(concordance_output.suffix + ".metadata.json"), result
    )
    if right_manifest is not None and seeds_output is not None:
        atomic_write_json(
            seeds_output.with_suffix(seeds_output.suffix + ".metadata.json"), result
        )
    return result


def _canonical_record(record, drop: Sequence[str] = ()) -> bytes:
    fields = list(record.columns[:7])
    fields.append(canonical_info(record.info, drop=drop))
    return ("\t".join(fields) + "\n").encode("utf-8")


def audit_pair(row: PairRow, dp_distance: int = 50) -> dict:
    source = Path(row.values["source_vcf"])
    prediction = Path(row.values["prediction_vcf"])
    source_exists = source.exists()
    prediction_exists = prediction.exists()
    source_stat = source.stat() if source_exists else None
    prediction_stat = prediction.stat() if prediction_exists else None
    result = {
        "chunk_id": row.chunk_id,
        "source_vcf": str(source),
        "prediction_vcf": str(prediction),
        "seed": row.values.get("seed", ""),
        "source_exists": source_exists,
        "prediction_exists": prediction_exists,
        "source_bytes": source_stat.st_size if source_stat is not None else None,
        "source_mtime_ns": source_stat.st_mtime_ns if source_stat is not None else None,
        "prediction_bytes": (
            prediction_stat.st_size if prediction_stat is not None else None
        ),
        "prediction_mtime_ns": (
            prediction_stat.st_mtime_ns if prediction_stat is not None else None
        ),
        "source_rows": 0,
        "prediction_rows": 0,
        "preserved_rows": 0,
        "unsupported_contig_rows": 0,
        "invalid_spliceai_annotations": 0,
        "invalid_openspliceai_annotations": 0,
        "out_of_range_spliceai_dp": 0,
        "out_of_range_openspliceai_dp": 0,
        "spliceai_info_header": False,
        "openspliceai_info_header": False,
        "dp_distance": dp_distance,
        "source_digest": None,
        "prediction_source_digest": None,
        "input_file_sha256": None,
        "output_file_sha256": None,
        "final_newline": False,
        "error": None,
        "state": "unknown",
    }
    result.update(_output_provenance_fields(row.values))
    if not source_exists:
        result["state"] = "missing_source"
        return result
    if not prediction_exists:
        result["state"] = "missing"
        return result
    if prediction_stat is not None and prediction_stat.st_size == 0:
        result["state"] = "empty"
        return result
    try:
        result["final_newline"] = has_final_newline(prediction)
        source_header = inspect_header(source)
        prediction_header = inspect_header(prediction)
        source_contigs = source_header["contigs"]
        result["spliceai_info_header"] = "SpliceAI" in source_header["info_ids"]
        result["openspliceai_info_header"] = (
            "OpenSpliceAI" in prediction_header["info_ids"]
        )
        source_digest = hashlib.sha256()
        prediction_digest = hashlib.sha256()
        source_identity: dict[str, object] = {}
        prediction_identity: dict[str, object] = {}
        mismatch = False
        sentinel = object()
        source_records = iter_vcf_records(
            source, verification_result=source_identity
        )
        prediction_records = iter_vcf_records(
            prediction,
            verification_result=prediction_identity,
            canonical_drop=("OpenSpliceAI",),
        )
        for source_record, prediction_record in zip_longest(
            source_records, prediction_records, fillvalue=sentinel
        ):
            if source_record is not sentinel:
                result["source_rows"] += 1
                source_digest.update(_canonical_record(source_record))
                if source_record.key.chrom not in source_contigs:
                    result["unsupported_contig_rows"] += 1
                annotations, invalid = parse_annotations(source_record.info.get("SpliceAI"))
                result["invalid_spliceai_annotations"] += invalid
                result["out_of_range_spliceai_dp"] += sum(
                    abs(dp) > dp_distance for ann in annotations for dp in ann.dps
                )
            if prediction_record is not sentinel:
                result["prediction_rows"] += 1
                prediction_digest.update(
                    _canonical_record(prediction_record, drop=("OpenSpliceAI",))
                )
                annotations, invalid = parse_annotations(
                    prediction_record.info.get("OpenSpliceAI")
                )
                result["invalid_openspliceai_annotations"] += invalid
                result["out_of_range_openspliceai_dp"] += sum(
                    abs(dp) > dp_distance for ann in annotations for dp in ann.dps
                )
            if source_record is sentinel or prediction_record is sentinel:
                mismatch = True
                continue
            if _canonical_record(source_record) == _canonical_record(
                prediction_record, drop=("OpenSpliceAI",)
            ):
                result["preserved_rows"] += 1
            else:
                mismatch = True
        result["source_digest"] = source_digest.hexdigest()
        result["prediction_source_digest"] = prediction_digest.hexdigest()
        result["input_file_sha256"] = source_identity["file_sha256"]
        result["output_file_sha256"] = prediction_identity["file_sha256"]
        if (
            source_identity["records"] != result["source_rows"]
            or source_identity["canonical_digest"] != result["source_digest"]
            or prediction_identity["records"] != result["prediction_rows"]
            or prediction_identity["canonical_digest"]
            != result["prediction_source_digest"]
        ):
            raise ValueError("internal VCF identity accounting mismatch")
        if not result["final_newline"]:
            result["state"] = "truncated"
        elif result["source_rows"] != result["prediction_rows"]:
            result["state"] = "row_count_mismatch"
        elif mismatch or result["preserved_rows"] != result["source_rows"]:
            result["state"] = "source_mismatch"
        elif not result["openspliceai_info_header"]:
            result["state"] = "missing_openspliceai_info_header"
        elif (
            result["invalid_spliceai_annotations"]
            or result["invalid_openspliceai_annotations"]
            or result["out_of_range_spliceai_dp"]
            or result["out_of_range_openspliceai_dp"]
        ):
            result["state"] = "annotation_error"
        else:
            result["state"] = "valid"
        current_source = source.stat()
        current_prediction = prediction.stat()
        if (
            source_stat is None
            or prediction_stat is None
            or (current_source.st_size, current_source.st_mtime_ns)
            != (source_stat.st_size, source_stat.st_mtime_ns)
            or (current_prediction.st_size, current_prediction.st_mtime_ns)
            != (prediction_stat.st_size, prediction_stat.st_mtime_ns)
        ):
            result["state"] = "mutated_during_audit"
            result["error"] = "VCF size or mtime_ns changed during the audit snapshot"
    except (EOFError, OSError, UnicodeError, ValueError) as exc:
        result["state"] = "parse_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_audit(
    pairs_file: str | Path,
    output: str | Path,
    parquet: Optional[str | Path] = None,
    task_index: Optional[int] = None,
    chunks_per_task: int = 250,
    dp_distance: int = 50,
) -> dict:
    rows = read_pairs_file(
        pairs_file,
        ("chunk_id", "source_vcf", "prediction_vcf"),
        task_index,
        chunks_per_task,
    )
    if dp_distance < 0:
        raise ValueError("dp_distance must be nonnegative")
    records = [audit_pair(row, dp_distance=dp_distance) for row in rows]
    counts: dict[str, int] = {}
    for record in records:
        state = str(record["state"])
        counts[state] = counts.get(state, 0) + 1
    payload = {
        "schema_version": 1,
        "kind": "audit",
        "created_at": utc_now(),
        "pairs_file": str(Path(pairs_file).resolve()),
        "pairs_sha256": sha256_file(pairs_file),
        "task_index": task_index,
        "chunks_per_task": chunks_per_task,
        "state_counts": counts,
        "chunks": records,
    }
    atomic_write_json(output, payload)
    if parquet:
        write_parquet(records, parquet)
    return payload


def _required_pair_value(row: PairRow, field: str) -> str:
    value = row.values.get(field, "")
    if value == "":
        raise ValueError(
            f"chunk {row.chunk_id}: missing audited provenance field {field}"
        )
    return value


def _pair_nonnegative_int(row: PairRow, field: str) -> int:
    value = _required_pair_value(row, field)
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"chunk {row.chunk_id}: invalid audited provenance field {field}={value!r}"
        ) from exc
    if result < 0:
        raise ValueError(
            f"chunk {row.chunk_id}: negative audited provenance field {field}"
        )
    return result


def _pair_sha256(row: PairRow, field: str) -> str:
    value = _required_pair_value(row, field).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(
            f"chunk {row.chunk_id}: invalid audited provenance digest {field}"
        )
    return value


def _pair_output_provenance_class(row: PairRow, field: str) -> str:
    value = _required_pair_value(row, field).lower()
    if value not in _OUTPUT_PROVENANCE_CLASSES:
        raise ValueError(
            f"chunk {row.chunk_id}: unsupported {field}={value!r}"
        )
    return value


def _verify_vcf_snapshot(
    row: PairRow,
    path_field: str,
    field_prefix: str,
    identity_prefix: str,
    role: str,
    drop: Sequence[str] = (),
    verify_content: bool = True,
) -> dict:
    path = Path(_required_pair_value(row, path_field))
    expected_bytes = _pair_nonnegative_int(
        row, f"{field_prefix}{identity_prefix}bytes"
    )
    expected_mtime_ns = _pair_nonnegative_int(
        row, f"{field_prefix}{identity_prefix}mtime_ns"
    )
    expected_rows = _pair_nonnegative_int(
        row, f"{field_prefix}{identity_prefix}rows"
    )
    digest_field = (
        f"{field_prefix}prediction_source_digest"
        if identity_prefix == "prediction_"
        else f"{field_prefix}source_digest"
    )
    expected_digest = _pair_sha256(row, digest_field)
    file_digest_field = (
        f"{field_prefix}output_file_sha256"
        if identity_prefix == "prediction_"
        else f"{field_prefix}input_file_sha256"
    )
    expected_file_sha256 = _pair_sha256(row, file_digest_field)
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(
            f"chunk {row.chunk_id}: audited {role} VCF is unavailable: {path}"
        ) from exc
    if (before.st_size, before.st_mtime_ns) != (expected_bytes, expected_mtime_ns):
        raise ValueError(
            f"chunk {row.chunk_id}: audited {role} VCF stat mismatch for {path}; "
            f"observed bytes/mtime_ns={(before.st_size, before.st_mtime_ns)} "
            f"expected={(expected_bytes, expected_mtime_ns)}"
        )
    snapshot = {
        "role": role,
        "path": str(path.resolve()),
        "bytes": expected_bytes,
        "mtime_ns": expected_mtime_ns,
        "records": expected_rows,
        "canonical_digest": expected_digest,
        "file_sha256": expected_file_sha256,
        "canonical_drop": list(drop),
    }
    if verify_content:
        _consume_verified_vcf(snapshot)
    return snapshot


def _iterator_verification(snapshot: Mapping) -> dict:
    return {
        "file_sha256": str(snapshot["file_sha256"]),
        "records": int(snapshot["records"]),
        "canonical_digest": str(snapshot["canonical_digest"]),
        "canonical_drop": list(snapshot.get("canonical_drop", [])),
    }


def _consume_verified_vcf(snapshot: Mapping) -> None:
    path = Path(snapshot["path"])
    for _record in iter_vcf_records(
        path,
        verification=_iterator_verification(snapshot),
        canonical_drop=tuple(snapshot.get("canonical_drop", ())),
    ):
        pass
    observed = path.stat()
    expected = (int(snapshot["bytes"]), int(snapshot["mtime_ns"]))
    if (observed.st_size, observed.st_mtime_ns) != expected:
        raise ValueError(f"verified VCF changed during identity check: {path}")


def _verify_mapper_provenance(rows: Sequence[PairRow], kind: str) -> dict:
    chunks: list[dict] = []
    manifest_sha256s: set[str] = set()
    provenance_class_counts: Counter = Counter()
    for row in rows:
        files: list[dict] = []
        if kind == "concordance":
            manifest_sha256s.add(_pair_sha256(row, "manifest_sha256"))
            provenance_class_counts[
                _pair_output_provenance_class(row, "output_provenance_class")
            ] += 1
            files.append(
                _verify_vcf_snapshot(
                    row, "source_vcf", "", "source_", "source"
                )
            )
            files.append(
                _verify_vcf_snapshot(
                    row,
                    "prediction_vcf",
                    "",
                    "prediction_",
                    "prediction",
                    drop=("OpenSpliceAI",),
                    verify_content=False,
                )
            )
            if (
                files[0]["records"] != files[1]["records"]
                or files[0]["canonical_digest"] != files[1]["canonical_digest"]
            ):
                raise ValueError(
                    f"chunk {row.chunk_id}: prediction no longer preserves its audited source identity"
                )
        elif kind == "seeds":
            for side in ("left", "right"):
                manifest_sha256s.add(_pair_sha256(row, f"{side}_manifest_sha256"))
                provenance_class_counts[
                    _pair_output_provenance_class(
                        row, f"{side}_output_provenance_class"
                    )
                ] += 1
                source_snapshot = _verify_vcf_snapshot(
                    row,
                    f"{side}_source_vcf",
                    f"{side}_",
                    "source_",
                    f"{side} source",
                )
                prediction_snapshot = _verify_vcf_snapshot(
                    row,
                    f"{side}_vcf",
                    f"{side}_",
                    "prediction_",
                    f"{side} prediction",
                    drop=("OpenSpliceAI",),
                    verify_content=False,
                )
                if (
                    source_snapshot["records"] != prediction_snapshot["records"]
                    or source_snapshot["canonical_digest"]
                    != prediction_snapshot["canonical_digest"]
                ):
                    raise ValueError(
                        f"chunk {row.chunk_id}: {side} prediction no longer preserves its audited source identity"
                    )
                files.extend((source_snapshot, prediction_snapshot))
            if (
                files[0]["records"] != files[2]["records"]
                or files[0]["canonical_digest"] != files[2]["canonical_digest"]
            ):
                raise ValueError(
                    f"chunk {row.chunk_id}: seed outputs are not bound to the same source identity"
                )
        else:
            raise ValueError(f"unsupported mapper provenance kind: {kind}")
        chunks.append({"chunk_id": row.chunk_id, "vcfs": files})
    return {
        "status": "pending_output_stream_verification",
        "contract": PROVENANCE_CONTRACT,
        "audit_manifest_sha256s": sorted(manifest_sha256s),
        "output_provenance_class_counts": dict(sorted(provenance_class_counts.items())),
        "verified_chunk_count": len(chunks),
        "verified_vcf_count": sum(len(chunk["vcfs"]) for chunk in chunks),
        "chunks": chunks,
    }


def _assert_verified_vcfs_unchanged(provenance: Mapping) -> None:
    for chunk in provenance.get("chunks", []):
        chunk_id = str(chunk.get("chunk_id", ""))
        for snapshot in chunk.get("vcfs", []):
            path = Path(snapshot["path"])
            try:
                observed = path.stat()
            except OSError as exc:
                raise ValueError(
                    f"chunk {chunk_id}: verified VCF disappeared during mapping: {path}"
                ) from exc
            expected = (int(snapshot["bytes"]), int(snapshot["mtime_ns"]))
            if (observed.st_size, observed.st_mtime_ns) != expected:
                raise ValueError(
                    f"chunk {chunk_id}: verified VCF changed during mapping: {path}"
                )


def _read_pairs_snapshot(
    path: str | Path,
    required_columns: Sequence[str],
    task_index: Optional[int],
    chunks_per_task: int,
) -> tuple[list[PairRow], str]:
    path = Path(path)
    before = path.stat()
    pairs_sha256 = sha256_file(path)
    rows = read_pairs_file(
        path, required_columns, task_index=task_index, chunks_per_task=chunks_per_task
    )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: pairs file changed while mapper input was read")
    if sha256_file(path) != pairs_sha256:
        raise ValueError(f"{path}: pairs file digest changed while mapper input was read")
    return rows, pairs_sha256


def _chunk_vcf_snapshot(chunk: Mapping, role: str) -> Mapping:
    matches = [value for value in chunk.get("vcfs", []) if value.get("role") == role]
    if len(matches) != 1:
        raise ValueError(
            f"chunk {chunk.get('chunk_id')}: expected exactly one verified {role} snapshot"
        )
    return matches[0]


def _map_groups(
    groups: Iterable[VariantGroup],
    aggregate: Aggregate,
    chunk_id: str,
) -> list[dict]:
    edges: list[dict] = []
    for position, group in stream_with_sided_edges(groups):
        if position != "interior":
            edges.append(
                {"chunk_id": str(chunk_id), "edge": position, "group": group.to_dict()}
            )
        else:
            aggregate.add_group(group)
    return edges


def run_map_concordance(
    pairs_file: str | Path,
    output: str | Path,
    config: AnalysisConfig,
    task_index: Optional[int] = None,
    chunks_per_task: int = 250,
    run_label: str = "provisional",
    parquet_dir: Optional[str | Path] = None,
    sites: Optional["SiteIndex"] = None,
    aggregate_type=Aggregate,
) -> dict:
    rows, pairs_sha256 = _read_pairs_snapshot(
        pairs_file,
        ("chunk_id", "prediction_vcf"),
        task_index,
        chunks_per_task,
    )
    verified_provenance = _verify_mapper_provenance(rows, "concordance")
    verified_provenance.update(
        {
            "pairs_file": str(Path(pairs_file).resolve()),
            "pairs_sha256": pairs_sha256,
        }
    )
    aggregate = aggregate_type(config, sites=sites)
    edges: list[dict] = []
    for row, chunk_provenance in zip(rows, verified_provenance["chunks"]):
        prediction_snapshot = _chunk_vcf_snapshot(
            chunk_provenance, "prediction"
        )
        groups = iter_variant_groups(
            (row.values["prediction_vcf"],),
            {"left": "SpliceAI", "right": "OpenSpliceAI"},
            verification_by_path={
                str(prediction_snapshot["path"]): _iterator_verification(
                    prediction_snapshot
                )
            },
        )
        edges.extend(_map_groups(groups, aggregate, row.chunk_id))
    _assert_verified_vcfs_unchanged(verified_provenance)
    verified_provenance["status"] = "verified"
    payload = {
        "schema_version": 2,
        "kind": "concordance",
        "created_at": utc_now(),
        "descriptive_run_label": run_label,
        "left_label": "SpliceAI",
        "right_label": "OpenSpliceAI",
        "pairs_file": str(Path(pairs_file).resolve()),
        "pairs_sha256": pairs_sha256,
        "verified_provenance": verified_provenance,
        "task_index": task_index,
        "chunks_per_task": chunks_per_task,
        "chunk_ids": [row.chunk_id for row in rows],
        "aggregate": aggregate.to_dict(),
        "edge_groups": edges,
    }
    atomic_write_json(output, payload)
    if parquet_dir:
        write_compact_tables(aggregate, parquet_dir, _shard_stem(output))
    return payload


def run_map_seeds(
    pairs_file: str | Path,
    output: str | Path,
    config: AnalysisConfig,
    task_index: Optional[int] = None,
    chunks_per_task: int = 250,
    run_label: str = "provisional",
    parquet_dir: Optional[str | Path] = None,
    sites: Optional["SiteIndex"] = None,
    aggregate_type=Aggregate,
    shared_info: Optional[str] = None,
) -> dict:
    rows, pairs_sha256 = _read_pairs_snapshot(
        pairs_file,
        ("chunk_id", "left_vcf", "right_vcf"),
        task_index,
        chunks_per_task,
    )
    verified_provenance = _verify_mapper_provenance(rows, "seeds")
    verified_provenance.update(
        {
            "pairs_file": str(Path(pairs_file).resolve()),
            "pairs_sha256": pairs_sha256,
        }
    )
    aggregate = aggregate_type(config, sites=sites)
    edges: list[dict] = []
    for row, chunk_provenance in zip(rows, verified_provenance["chunks"]):
        left_snapshot = _chunk_vcf_snapshot(
            chunk_provenance, "left prediction"
        )
        right_snapshot = _chunk_vcf_snapshot(
            chunk_provenance, "right prediction"
        )
        groups = merge_group_streams(
            (row.values["left_vcf"],),
            (row.values["right_vcf"],),
            left_verification_by_path={
                str(left_snapshot["path"]): _iterator_verification(left_snapshot)
            },
            right_verification_by_path={
                str(right_snapshot["path"]): _iterator_verification(right_snapshot)
            },
            shared_info=shared_info,
        )
        edges.extend(_map_groups(groups, aggregate, row.chunk_id))
    _assert_verified_vcfs_unchanged(verified_provenance)
    verified_provenance["status"] = "verified"
    payload = {
        "schema_version": 2,
        "kind": "seeds",
        "created_at": utc_now(),
        "descriptive_run_label": run_label,
        "left_label": "OpenSpliceAI left seed",
        "right_label": "OpenSpliceAI right seed",
        "pairs_file": str(Path(pairs_file).resolve()),
        "pairs_sha256": pairs_sha256,
        "verified_provenance": verified_provenance,
        "task_index": task_index,
        "chunks_per_task": chunks_per_task,
        "chunk_ids": [row.chunk_id for row in rows],
        "aggregate": aggregate.to_dict(),
        "edge_groups": edges,
    }
    atomic_write_json(output, payload)
    if parquet_dir:
        write_compact_tables(aggregate, parquet_dir, _shard_stem(output))
    return payload


def _shard_stem(path: str | Path) -> str:
    return Path(path).name.rsplit(".", 1)[0]


def find_shards(
    inputs: Sequence[str | Path] = (), input_dir: Optional[str | Path] = None
) -> list[Path]:
    paths = [Path(path) for path in inputs]
    if input_dir:
        paths.extend(sorted(Path(input_dir).glob("map-*.json")))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no mapper JSON shards found")
    return unique


def run_reduce(
    kind: str,
    output: str | Path,
    inputs: Sequence[str | Path] = (),
    input_dir: Optional[str | Path] = None,
    parquet_dir: Optional[str | Path] = None,
    expected_task_count: Optional[int] = None,
    pairs_file: Optional[str | Path] = None,
    expected_total_chunks: int = 100_000,
    finality: str = "provisional",
    expected_overlap_count: Optional[int] = None,
    sites: Optional["SiteIndex"] = None,
    aggregate_type=Aggregate,
) -> dict:
    if kind not in {"concordance", "seeds"}:
        raise ValueError("kind must be either 'concordance' or 'seeds'")
    if expected_task_count is None or expected_task_count < 1:
        raise ValueError("expected_task_count is required and must be positive")
    if pairs_file is None:
        raise ValueError("pairs_file is required for fail-closed chunk verification")
    finality = str(finality).lower()
    if finality not in {"provisional", "final"}:
        raise ValueError("finality must be either 'provisional' or 'final'")
    if expected_total_chunks < 1:
        raise ValueError("expected_total_chunks must be positive")
    if expected_overlap_count is not None and expected_overlap_count < 0:
        raise ValueError("expected_overlap_count must be nonnegative")
    if kind == "concordance" and expected_overlap_count is not None:
        raise ValueError("expected_overlap_count applies only to seed reductions")
    paths = find_shards(inputs, input_dir)
    aggregate: Optional[Aggregate] = None
    edge_fragments: list[dict] = []
    task_indices: set[int] = set()
    pairs_hashes: set[str] = set()
    chunk_ids: list[str] = []
    mapper_run_labels: set[str] = set()
    chunks_per_task_values: set[int] = set()
    comparison_labels: set[tuple[str, str]] = set()
    verified_manifest_sha256s: set[str] = set()
    verified_chunk_count = 0
    verified_vcf_count = 0
    verified_provenance_class_counts: Counter = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if shard.get("kind") != kind:
            raise ValueError(f"{path}: expected kind {kind!r}, found {shard.get('kind')!r}")
        task_index = shard.get("task_index")
        if task_index is not None:
            task_index = int(task_index)
            if task_index in task_indices:
                raise ValueError(f"duplicate task_index {task_index}")
            task_indices.add(task_index)
        shard_pairs_sha256 = str(shard.get("pairs_sha256", ""))
        pairs_hashes.add(shard_pairs_sha256)
        mapper_run_labels.add(
            str(shard.get("descriptive_run_label", shard.get("run_label", "")))
        )
        chunks_per_task_values.add(int(shard.get("chunks_per_task", 0)))
        comparison_labels.add(
            (str(shard.get("left_label", "left")), str(shard.get("right_label", "right")))
        )
        shard_chunk_ids = [str(x) for x in shard.get("chunk_ids", [])]
        chunk_ids.extend(shard_chunk_ids)
        provenance = shard.get("verified_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{path}: mapper shard lacks verified provenance")
        if (
            provenance.get("status") != "verified"
            or provenance.get("contract") != PROVENANCE_CONTRACT
        ):
            raise ValueError(f"{path}: mapper provenance contract is not verified")
        if str(provenance.get("pairs_sha256", "")) != shard_pairs_sha256:
            raise ValueError(f"{path}: mapper provenance does not bind its pairs hash")
        provenance_chunk_ids = [
            str(chunk.get("chunk_id", ""))
            for chunk in provenance.get("chunks", [])
            if isinstance(chunk, Mapping)
        ]
        if provenance_chunk_ids != shard_chunk_ids:
            raise ValueError(f"{path}: mapper provenance chunk IDs do not match shard coverage")
        shard_verified_chunks = int(provenance.get("verified_chunk_count", -1))
        shard_verified_vcfs = int(provenance.get("verified_vcf_count", -1))
        expected_vcfs = len(shard_chunk_ids) * (2 if kind == "concordance" else 4)
        if (
            shard_verified_chunks != len(shard_chunk_ids)
            or shard_verified_vcfs != expected_vcfs
        ):
            raise ValueError(f"{path}: mapper provenance verification counts are incomplete")
        verified_chunk_count += shard_verified_chunks
        verified_vcf_count += shard_verified_vcfs
        verified_manifest_sha256s.update(
            str(value) for value in provenance.get("audit_manifest_sha256s", [])
        )
        shard_class_counts = provenance.get("output_provenance_class_counts", {})
        if not isinstance(shard_class_counts, Mapping):
            raise ValueError(f"{path}: invalid output provenance class counts")
        normalized_class_counts: Counter = Counter()
        for provenance_class, count in shard_class_counts.items():
            if str(provenance_class) not in _OUTPUT_PROVENANCE_CLASSES or int(count) < 0:
                raise ValueError(f"{path}: invalid output provenance class count")
            normalized_class_counts[str(provenance_class)] = int(count)
        expected_output_count = len(shard_chunk_ids) * (1 if kind == "concordance" else 2)
        if sum(normalized_class_counts.values()) != expected_output_count:
            raise ValueError(f"{path}: incomplete output provenance class counts")
        verified_provenance_class_counts.update(normalized_class_counts)
        incoming = aggregate_type.from_dict(shard["aggregate"])
        if aggregate is None:
            aggregate = incoming
        else:
            aggregate.merge(incoming)
        edge_fragments.extend(shard.get("edge_groups", []))
    assert aggregate is not None
    expected_indices = set(range(expected_task_count))
    if task_indices != expected_indices:
        missing = sorted(expected_indices - task_indices)
        unexpected = sorted(task_indices - expected_indices)
        raise ValueError(
            "incomplete mapper task set: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    if len(chunks_per_task_values) != 1 or 0 in chunks_per_task_values:
        raise ValueError(
            f"mapper shards disagree on chunks_per_task: {sorted(chunks_per_task_values)}"
        )
    if len(comparison_labels) != 1:
        raise ValueError(f"mapper shards disagree on comparison labels: {comparison_labels}")
    if len(mapper_run_labels) != 1:
        raise ValueError(
            f"mapper shards disagree on descriptive run label: {sorted(mapper_run_labels)}"
        )
    left_label, right_label = next(iter(comparison_labels))
    pairs_path = Path(pairs_file)
    pairs_before = pairs_path.stat()
    actual_pairs_hash = sha256_file(pairs_path)
    if pairs_hashes != {actual_pairs_hash}:
        raise ValueError(
            f"mapper pair-list hashes {sorted(pairs_hashes)} do not match {actual_pairs_hash}"
        )
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "chunk_id" not in (reader.fieldnames or ()):
            raise ValueError(f"{pairs_path}: missing chunk_id column")
        expected_pair_rows = [dict(row) for row in reader]
    pairs_after = pairs_path.stat()
    if (
        (pairs_before.st_size, pairs_before.st_mtime_ns)
        != (pairs_after.st_size, pairs_after.st_mtime_ns)
        or sha256_file(pairs_path) != actual_pairs_hash
    ):
        raise ValueError(f"{pairs_path}: pairs file changed during reduction")
    expected_chunk_ids = [str(row["chunk_id"]) for row in expected_pair_rows]
    expected_counts = Counter(expected_chunk_ids)
    if any(value != 1 for value in expected_counts.values()):
        duplicates = sorted(key for key, value in expected_counts.items() if value != 1)
        raise ValueError(f"pairs file contains duplicate chunk IDs: {duplicates[:20]}")
    observed_counts = Counter(chunk_ids)
    if observed_counts != expected_counts:
        missing = sorted((expected_counts - observed_counts).elements())
        duplicate_or_unexpected = sorted((observed_counts - expected_counts).elements())
        raise ValueError(
            "mapper chunk coverage mismatch: "
            f"missing={missing[:20]} duplicate_or_unexpected={duplicate_or_unexpected[:20]}"
        )
    numeric_chunk_ids: list[int] = []
    for value in expected_chunk_ids:
        try:
            numeric = int(value)
        except ValueError as exc:
            raise ValueError(
                "numeric chunk IDs are required for boundary-safe reduction"
            ) from exc
        if value != str(numeric) or numeric < 1:
            raise ValueError(
                "chunk IDs must be canonical positive integers for boundary-safe reduction"
            )
        numeric_chunk_ids.append(numeric)
    selected_chunks = set(numeric_chunk_ids)
    if expected_total_chunks < max(selected_chunks, default=0):
        raise ValueError("expected_total_chunks is smaller than an observed chunk ID")

    expected_manifest_sha256s: set[str] = set()
    expected_provenance_class_counts: Counter = Counter()
    receipt_evidence_output_count = 0
    log_evidence_output_count = 0
    for pair_row in expected_pair_rows:
        row = PairRow(str(pair_row["chunk_id"]), pair_row)
        if kind == "concordance":
            expected_manifest_sha256s.add(_pair_sha256(row, "manifest_sha256"))
            expected_provenance_class_counts[
                _pair_output_provenance_class(row, "output_provenance_class")
            ] += 1
            prefixes = ("",)
        else:
            expected_manifest_sha256s.add(
                _pair_sha256(row, "left_manifest_sha256")
            )
            expected_manifest_sha256s.add(
                _pair_sha256(row, "right_manifest_sha256")
            )
            for side in ("left", "right"):
                expected_provenance_class_counts[
                    _pair_output_provenance_class(
                        row, f"{side}_output_provenance_class"
                    )
                ] += 1
            prefixes = ("left_", "right_")
        for prefix in prefixes:
            evidence = {
                name[len(prefix) :] if prefix else name: value
                for name, value in pair_row.items()
                if name.startswith(prefix) and value not in (None, "")
            }
            if any("receipt" in name.lower() for name in evidence):
                receipt_evidence_output_count += 1
            if any(
                "log" in name.lower() and "evidence" in name.lower()
                for name in evidence
            ):
                log_evidence_output_count += 1
    if verified_manifest_sha256s != expected_manifest_sha256s:
        raise ValueError(
            "mapper provenance audit-manifest digests do not match the frozen pairs: "
            f"observed={sorted(verified_manifest_sha256s)} "
            f"expected={sorted(expected_manifest_sha256s)}"
        )
    if verified_provenance_class_counts != expected_provenance_class_counts:
        raise ValueError(
            "mapper output provenance classes do not match the frozen pairs"
        )
    if verified_chunk_count != len(expected_chunk_ids):
        raise ValueError("mapper provenance does not verify every frozen pair")

    expected_domain = set(range(1, expected_total_chunks + 1))
    missing_domain_ids = sorted(expected_domain - selected_chunks)
    unexpected_domain_ids = sorted(selected_chunks - expected_domain)
    domain_complete = (
        len(expected_chunk_ids) == expected_total_chunks
        and not missing_domain_ids
        and not unexpected_domain_ids
    )
    if finality == "final" and kind == "concordance" and not domain_complete:
        raise ValueError(
            "final concordance requires pair IDs exactly 1..expected_total_chunks: "
            f"observed_count={len(expected_chunk_ids)} "
            f"expected_count={expected_total_chunks} "
            f"missing={missing_domain_ids[:20]} unexpected={unexpected_domain_ids[:20]}"
        )
    if finality == "final" and kind == "seeds":
        if expected_overlap_count is None:
            raise ValueError(
                "final seed reduction requires an explicit expected_overlap_count policy"
            )
    if kind == "seeds" and expected_overlap_count is not None:
        if len(expected_chunk_ids) != expected_overlap_count:
            raise ValueError(
                "seed overlap count mismatch: "
                f"observed={len(expected_chunk_ids)} expected={expected_overlap_count}"
            )

    finality_payload = {
        "status": finality,
        "contract": FINALITY_CONTRACT,
        "policy": (
            "complete_chunk_domain"
            if kind == "concordance" and finality == "final"
            else "explicit_seed_overlap_count"
            if kind == "seeds" and finality == "final"
            else "partial_coverage_allowed"
        ),
        "observed_pair_count": len(expected_chunk_ids),
        "expected_total_chunks": expected_total_chunks,
        "expected_overlap_count": expected_overlap_count,
        "pair_id_domain_complete": domain_complete,
        "unselected_chunk_count": len(missing_domain_ids),
        "unselected_chunk_ids_preview": missing_domain_ids[:100],
        "unexpected_chunk_ids": unexpected_domain_ids[:100],
    }
    complete_edges: dict[str, VariantGroup] = {}
    incomplete_tokens: set[str] = set()
    edge_counts: dict[str, int] = {}
    for serialized in edge_fragments:
        chunk = int(serialized["chunk_id"])
        edge = str(serialized["edge"])
        left_complete = chunk == 1 or (chunk - 1) in selected_chunks
        right_complete = chunk == expected_total_chunks or (chunk + 1) in selected_chunks
        safe = (
            (edge == "first" and left_complete)
            or (edge == "last" and right_complete)
            or (edge == "both" and left_complete and right_complete)
        )
        group = VariantGroup.from_dict(serialized["group"])
        token = group.key.token()
        edge_counts[token] = edge_counts.get(token, 0) + 1
        if not safe:
            incomplete_tokens.add(token)
        if token in complete_edges:
            complete_edges[token].merge(group)
        else:
            complete_edges[token] = group
    # Completeness belongs to the joined variant, not an individual fragment.
    # A group filling a whole chunk may carry an incomplete edge into an
    # otherwise locally safe fragment in its selected neighbour.
    for token in incomplete_tokens:
        complete_edges.pop(token)
        aggregate.coverage["excluded_incomplete_edge_fragments"] += edge_counts[token]
    # The mappers deferred every chunk-boundary group; rejoining them below adds
    # real rows to the aggregate, so the reducer needs the same splice-site index the
    # mappers used. Requiring the digests to match is what stops boundary variants
    # being stratified against a different annotation than the interior ones.
    if aggregate is not None and aggregate.config.sites_digest:
        if sites is None:
            raise ValueError(
                "mapper shards used a site_distance stratum; --sites-file is required to reduce them"
            )
        if sites.digest != aggregate.config.sites_digest:
            raise ValueError(
                "sites file does not match the mappers: "
                f"{sites.digest} != {aggregate.config.sites_digest}"
            )
        aggregate.sites = sites
    elif sites is not None:
        raise ValueError("--sites-file was given but the mapper shards carry no site_distance stratum")

    for token in sorted(complete_edges):
        aggregate.add_group(complete_edges[token])
    finality_payload["excluded_incomplete_edge_fragments"] = int(
        aggregate.coverage["excluded_incomplete_edge_fragments"]
    )
    payload = {
        "schema_version": 2,
        "kind": f"reduced-{kind}",
        "created_at": utc_now(),
        "finality": finality_payload,
        "mapper_run_labels": sorted(mapper_run_labels),
        "pairs_sha256": actual_pairs_hash,
        "verified_pairs_file": str(pairs_path.resolve()),
        "verified_provenance": {
            "status": "verified",
            "contract": PROVENANCE_CONTRACT,
            "pairs_file": str(pairs_path.resolve()),
            "pairs_sha256": actual_pairs_hash,
            "audit_manifest_sha256s": sorted(expected_manifest_sha256s),
            "output_provenance_class_counts": dict(
                sorted(expected_provenance_class_counts.items())
            ),
            "receipt_evidence_output_count": receipt_evidence_output_count,
            "log_evidence_output_count": log_evidence_output_count,
            "verified_chunk_count": verified_chunk_count,
            "verified_vcf_count": verified_vcf_count,
            "mapper_shard_count": len(paths),
        },
        "expected_task_count": expected_task_count,
        "expected_total_chunks": expected_total_chunks,
        "left_label": left_label,
        "right_label": right_label,
        "input_shards": [str(path) for path in paths],
        "input_shard_sha256": {str(path): sha256_file(path) for path in paths},
        "task_indices": sorted(task_indices),
        "chunk_ids": chunk_ids,
        "raw": aggregate.to_dict(),
        "metrics": aggregate.derived(),
    }
    atomic_write_json(output, payload)
    if parquet_dir:
        write_compact_tables(aggregate, parquet_dir, "reduced")
    return payload


def write_parquet(rows: Sequence[Mapping], path: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path, compression="zstd")


def write_compact_tables(
    aggregate: Aggregate, directory: str | Path, prefix: str
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = aggregate.to_dict()
    write_parquet(raw["sample"], directory / f"{prefix}.sample.parquet")
    write_parquet(
        raw["top_discrepancies"], directory / f"{prefix}.top_discrepancies.parquet"
    )
