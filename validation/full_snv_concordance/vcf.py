"""Small, dependency-free streaming VCF reader and annotation normalizer.

Full-genome inputs are deliberately not loaded through pandas or pysam.  The
VCFs in this campaign are uncompressed, sample-free VCF 4.2 files, and a text
reader gives predictable memory use while still accepting gzip-compressed
fixtures and future shards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import io
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple


EVENTS: Tuple[str, ...] = ("AG", "AL", "DG", "DL")
INFO_KEYS = {"spliceai": "SpliceAI", "openspliceai": "OpenSpliceAI"}


@dataclass(frozen=True, order=True)
class VariantKey:
    chrom: str
    pos: int
    ref: str
    alt: str

    def token(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


@dataclass(frozen=True)
class Annotation:
    allele: str
    gene: str
    scores: Tuple[float, float, float, float]
    dps: Tuple[int, int, int, int]

    def value_key(self) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
        return self.scores, self.dps


@dataclass
class VariantGroup:
    """All annotations observed for one genomic allele.

    Values are sets because OpenSpliceAI emits the same multi-gene annotation
    on each source gene row.  Identical repetitions are benign duplicates;
    more than one distinct value for a method/gene is a conflict.
    """

    key: VariantKey
    sides: MutableMapping[str, MutableMapping[str, set[Annotation]]] = field(
        default_factory=dict
    )
    invalid_annotations: int = 0
    rows: int = 0
    annotation_observations: int = 0
    duplicate_annotations: int = 0

    def add(self, side: str, annotation: Annotation) -> None:
        self.annotation_observations += 1
        if annotation.allele != self.key.alt:
            self.invalid_annotations += 1
            return
        by_gene = self.sides.setdefault(side, {})
        values = by_gene.setdefault(annotation.gene, set())
        if annotation in values:
            self.duplicate_annotations += 1
        values.add(annotation)

    def merge(self, other: "VariantGroup") -> "VariantGroup":
        if self.key != other.key:
            raise ValueError(f"cannot merge different variants: {self.key} != {other.key}")
        self.invalid_annotations += other.invalid_annotations
        self.rows += other.rows
        self.annotation_observations += other.annotation_observations
        self.duplicate_annotations += other.duplicate_annotations
        for side, genes in other.sides.items():
            target = self.sides.setdefault(side, {})
            for gene, incoming_annotations in genes.items():
                values = target.setdefault(gene, set())
                self.duplicate_annotations += len(values.intersection(incoming_annotations))
                values.update(incoming_annotations)
        return self

    def to_dict(self) -> dict:
        sides = {}
        for side, genes in sorted(self.sides.items()):
            sides[side] = {
                gene: [
                    {
                        "allele": ann.allele,
                        "scores": list(ann.scores),
                        "dps": list(ann.dps),
                    }
                    for ann in sorted(
                        annotations, key=lambda a: (a.scores, a.dps, a.allele)
                    )
                ]
                for gene, annotations in sorted(genes.items())
            }
        return {
            "key": [self.key.chrom, self.key.pos, self.key.ref, self.key.alt],
            "sides": sides,
            "invalid_annotations": self.invalid_annotations,
            "rows": self.rows,
            "annotation_observations": self.annotation_observations,
            "duplicate_annotations": self.duplicate_annotations,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "VariantGroup":
        chrom, pos, ref, alt = data["key"]
        group = cls(
            VariantKey(str(chrom), int(pos), str(ref), str(alt)),
            invalid_annotations=int(data.get("invalid_annotations", 0)),
            rows=int(data.get("rows", 0)),
        )
        for side, genes in data.get("sides", {}).items():
            for gene, values in genes.items():
                for value in values:
                    group.add(
                        str(side),
                        Annotation(
                            allele=str(value.get("allele", alt)),
                            gene=str(gene),
                            scores=tuple(_normal_float(x) for x in value["scores"]),
                            dps=tuple(int(x) for x in value["dps"]),
                        ),
                    )
        # add() reconstructs observation/duplicate counts for the unique values;
        # restore the original counters after the annotation sets are rebuilt.
        group.annotation_observations = int(
            data.get("annotation_observations", group.annotation_observations)
        )
        group.duplicate_annotations = int(
            data.get("duplicate_annotations", group.duplicate_annotations)
        )
        return group


@dataclass(frozen=True)
class VCFRecord:
    key: VariantKey
    columns: Tuple[str, ...]
    info: Mapping[str, Optional[str]]


def _normal_float(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite score: {value!r}")
    # Canonicalize textual -0.00000 so hashes and equality are stable.
    return 0.0 if result == 0.0 else result


def parse_info(text: str) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    if text in ("", "."):
        return result
    for item in text.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = None
    return result


def canonical_info(info: Mapping[str, Optional[str]], drop: Sequence[str] = ()) -> str:
    ignored = set(drop)
    parts = []
    for key in sorted(info):
        if key in ignored:
            continue
        value = info[key]
        parts.append(key if value is None else f"{key}={value}")
    return ";".join(parts) if parts else "."


def _raw_spliceai_value(info: bytes) -> bytes:
    """Return the byte-exact SpliceAI value used by the scoring auditor.

    ``full_genome_scoring/audit_vcfs.py`` binds source identity with a
    length-prefixed digest of the seven fixed fields and the raw SpliceAI
    value.  The concordance reader historically used a newline-delimited
    canonical digest.  Computing both lets the reader consume either manifest
    without treating a verified campaign snapshot as changed.
    """

    if info in (b"", b"."):
        return b""
    for item in info.split(b";"):
        key, separator, value = item.partition(b"=")
        if key == b"SpliceAI":
            return value if separator else b""
    return b""


def _update_length_prefixed_digest(digest, fields: Sequence[bytes], info: bytes) -> None:
    for value in fields[:7]:
        digest.update(len(value).to_bytes(4, "big"))
        digest.update(value)
    digest.update(len(info).to_bytes(4, "big"))
    digest.update(info)


def parse_annotation(text: str) -> Annotation:
    fields = text.split("|")
    if len(fields) != 10:
        raise ValueError(f"expected 10 annotation fields, found {len(fields)}")
    allele, gene = fields[0].strip(), fields[1].strip()
    if not allele or not gene:
        raise ValueError("annotation allele and gene must be nonempty")
    scores = tuple(_normal_float(value) for value in fields[2:6])
    if any(value < 0.0 or value > 1.0 for value in scores):
        raise ValueError(f"score outside [0, 1]: {scores}")
    dps = tuple(int(value) for value in fields[6:10])
    return Annotation(allele=allele, gene=gene, scores=scores, dps=dps)


def parse_annotations(value: Optional[str]) -> Tuple[list[Annotation], int]:
    if value in (None, "", "."):
        return [], 0
    annotations: list[Annotation] = []
    invalid = 0
    for text in str(value).split(","):
        try:
            annotations.append(parse_annotation(text))
        except (TypeError, ValueError):
            invalid += 1
    return annotations, invalid


def _open_text(path: Path):
    if path.suffix in {".gz", ".bgz"}:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="", buffering=1024 * 1024)


class _HashingRawReader(io.RawIOBase):
    """Non-seekable raw reader that hashes bytes consumed by gzip."""

    def __init__(self, handle, digest) -> None:
        super().__init__()
        self._handle = handle
        self._digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self._handle.read(len(buffer))
        if not data:
            return 0
        buffer[: len(data)] = data
        self._digest.update(data)
        return len(data)

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


def iter_vcf_records(
    path: str | Path,
    verification: Optional[Mapping[str, object]] = None,
    verification_result: Optional[MutableMapping[str, object]] = None,
    canonical_drop: Sequence[str] = (),
) -> Iterator[VCFRecord]:
    """Yield records and optionally verify raw/canonical identity at EOF.

    For ordinary uncompressed campaign VCFs the raw file SHA-256, canonical
    record digest, and record count are accumulated in this same streaming
    pass. Compressed inputs need a raw-byte pre-pass because gzip exposes the
    decompressed stream to the parser.
    """

    path = Path(path)
    collect_identity = verification is not None or verification_result is not None
    raw_digest = hashlib.sha256() if collect_identity else None
    canonical_digest = hashlib.sha256() if collect_identity else None
    length_prefixed_digest = hashlib.sha256() if collect_identity else None
    record_count = 0
    compressed = path.suffix in {".gz", ".bgz"}
    hashing_buffer = None
    if compressed and collect_identity:
        assert raw_digest is not None
        hashing_buffer = io.BufferedReader(
            _HashingRawReader(path.open("rb"), raw_digest), buffer_size=1024 * 1024
        )
        handle = io.TextIOWrapper(
            gzip.GzipFile(fileobj=hashing_buffer, mode="rb"),
            encoding="utf-8",
            newline="",
        )
    else:
        handle = _open_text(path) if compressed else path.open("rb", buffering=1024 * 1024)
    try:
        with handle:
            saw_header = False
            for line_number, raw_value in enumerate(handle, 1):
                if isinstance(raw_value, bytes):
                    raw_bytes = raw_value
                    if raw_digest is not None:
                        raw_digest.update(raw_value)
                    raw = raw_value.decode("utf-8")
                else:
                    raw = raw_value
                    raw_bytes = raw_value.encode("utf-8")
                if raw.startswith("##"):
                    continue
                if raw.startswith("#CHROM"):
                    saw_header = True
                    continue
                if raw.startswith("#"):
                    continue
                if not saw_header:
                    raise ValueError(f"{path}:{line_number}: data before #CHROM header")
                line = raw.rstrip("\r\n")
                columns = line.split("\t")
                if len(columns) < 8:
                    raise ValueError(f"{path}:{line_number}: expected at least 8 VCF columns")
                try:
                    pos = int(columns[1])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid POS {columns[1]!r}") from exc
                if pos < 1:
                    raise ValueError(f"{path}:{line_number}: POS must be one-based")
                record = VCFRecord(
                    key=VariantKey(columns[0], pos, columns[3], columns[4]),
                    columns=tuple(columns),
                    info=parse_info(columns[7]),
                )
                if collect_identity:
                    record_count += 1
                    fields = list(record.columns[:7])
                    fields.append(canonical_info(record.info, drop=canonical_drop))
                    assert canonical_digest is not None
                    canonical_digest.update(("\t".join(fields) + "\n").encode("utf-8"))
                    assert length_prefixed_digest is not None
                    raw_fields = raw_bytes.rstrip(b"\r\n").split(b"\t")
                    if len(raw_fields) >= 8:
                        _update_length_prefixed_digest(
                            length_prefixed_digest,
                            raw_fields[:7],
                            _raw_spliceai_value(raw_fields[7]),
                        )
                yield record
            if not saw_header:
                raise ValueError(f"{path}: missing #CHROM header")
    finally:
        if hashing_buffer is not None:
            hashing_buffer.close()
    if collect_identity:
        assert canonical_digest is not None
        raw_sha256 = raw_digest.hexdigest() if raw_digest is not None else ""
        observed = {
            "file_sha256": raw_sha256,
            "records": record_count,
            "canonical_digest": canonical_digest.hexdigest(),
            "length_prefixed_digest": length_prefixed_digest.hexdigest(),
        }
        if verification_result is not None:
            verification_result.update(observed)
        if verification is not None:
            expected = {
                "file_sha256": str(verification["file_sha256"]),
                "records": int(verification["records"]),
                "canonical_digest": str(verification["canonical_digest"]),
            }
            if (
                observed["file_sha256"] != expected["file_sha256"]
                or observed["records"] != expected["records"]
                or (
                    expected["canonical_digest"] != observed["canonical_digest"]
                    and expected["canonical_digest"]
                    != observed["length_prefixed_digest"]
                )
            ):
                raise ValueError(
                    f"{path}: VCF snapshot identity mismatch; "
                    f"observed={observed} expected={expected}"
                )


def iter_variant_groups(
    paths: Iterable[str | Path],
    side_to_info: Mapping[str, str],
    verification_by_path: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Iterator[VariantGroup]:
    """Yield consecutive variant groups, merging groups split between files."""

    current: Optional[VariantGroup] = None
    for path in paths:
        resolved = str(Path(path).resolve())
        verification = (
            verification_by_path.get(resolved) if verification_by_path is not None else None
        )
        canonical_drop = tuple(verification.get("canonical_drop", ())) if verification else ()
        for record in iter_vcf_records(
            path, verification=verification, canonical_drop=canonical_drop
        ):
            if current is None:
                current = VariantGroup(record.key)
            elif current.key != record.key:
                yield current
                current = VariantGroup(record.key)
            current.rows += 1
            for side, info_key in side_to_info.items():
                annotations, invalid = parse_annotations(record.info.get(info_key))
                current.invalid_annotations += invalid
                for annotation in annotations:
                    current.add(side, annotation)
    if current is not None:
        yield current


def merge_group_streams(
    left_paths: Iterable[str | Path],
    right_paths: Iterable[str | Path],
    left_info: str = "OpenSpliceAI",
    right_info: str = "OpenSpliceAI",
    left_verification_by_path: Optional[
        Mapping[str, Mapping[str, object]]
    ] = None,
    right_verification_by_path: Optional[
        Mapping[str, Mapping[str, object]]
    ] = None,
    shared_info: Optional[str] = None,
) -> Iterator[VariantGroup]:
    """Zip two score streams whose source VCF records have identical order."""

    left_fields = {"left": left_info}
    if shared_info:
        left_fields["reference"] = shared_info
    left_iter = iter_variant_groups(
        left_paths, left_fields, verification_by_path=left_verification_by_path
    )
    right_iter = iter_variant_groups(
        right_paths, {"right": right_info}, verification_by_path=right_verification_by_path
    )
    sentinel = object()
    while True:
        left = next(left_iter, sentinel)
        right = next(right_iter, sentinel)
        if left is sentinel and right is sentinel:
            return
        if left is sentinel or right is sentinel:
            raise ValueError("seed VCF streams have different numbers of variant groups")
        assert isinstance(left, VariantGroup) and isinstance(right, VariantGroup)
        if left.key != right.key:
            raise ValueError(f"seed VCF variant order differs: {left.key} != {right.key}")
        group = VariantGroup(left.key, rows=max(left.rows, right.rows))
        group.invalid_annotations = left.invalid_annotations + right.invalid_annotations
        group.annotation_observations = (
            left.annotation_observations + right.annotation_observations
        )
        group.duplicate_annotations = left.duplicate_annotations + right.duplicate_annotations
        group.sides = {"left": left.sides.get("left", {}), "right": right.sides.get("right", {})}
        if shared_info:
            # Both audited outputs preserve the identical source. Retain its
            # comparator annotations once for a three-way exact-gene analysis.
            group.sides["reference"] = left.sides.get("reference", {})
        yield group


def split_edges(groups: Iterable[VariantGroup]) -> Tuple[list[VariantGroup], Iterator[VariantGroup]]:
    """Compatibility helper retained for callers that materialize tiny fixtures.

    Production mapping uses :func:`stream_with_edges`, which remains bounded.
    """

    materialized = list(groups)
    if len(materialized) <= 2:
        return materialized, iter(())
    return [materialized[0], materialized[-1]], iter(materialized[1:-1])


def stream_with_edges(groups: Iterable[VariantGroup]) -> Iterator[Tuple[str, VariantGroup]]:
    """Tag first/last groups as edges while keeping only two groups in memory.

    Edge groups are excluded from mapper aggregates and reconstructed by the
    reducer. This makes results invariant to a variant being split at a mapper
    or source-chunk boundary.
    """

    iterator = iter(groups)
    first = next(iterator, None)
    if first is None:
        return
    pending = next(iterator, None)
    if pending is None:
        yield "edge", first
        return
    yield "edge", first
    for group in iterator:
        yield "interior", pending
        pending = group
    yield "edge", pending


def stream_with_sided_edges(
    groups: Iterable[VariantGroup],
) -> Iterator[Tuple[str, VariantGroup]]:
    """Like :func:`stream_with_edges`, but distinguish first/last boundaries."""

    iterator = iter(groups)
    first = next(iterator, None)
    if first is None:
        return
    pending = next(iterator, None)
    if pending is None:
        yield "both", first
        return
    yield "first", first
    for group in iterator:
        yield "interior", pending
        pending = group
    yield "last", pending


def has_final_newline(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix in {".gz", ".bgz"}:
        # A full gzip read catches truncation and exposes the logical last byte.
        last = b""
        with gzip.open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                last = block[-1:]
        return last == b"\n"
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"\n"


def declared_contigs(path: str | Path) -> set[str]:
    result: set[str] = set()
    with _open_text(Path(path)) as handle:
        for line in handle:
            if line.startswith("##contig=<ID="):
                result.add(line.split("##contig=<ID=", 1)[1].split(",", 1)[0].split(">", 1)[0])
            elif line.startswith("#CHROM"):
                break
    return result


def inspect_header(path: str | Path) -> dict:
    """Return declared contig/INFO identifiers without materializing the VCF."""

    contigs: set[str] = set()
    info_ids: set[str] = set()
    saw_chrom = False
    with _open_text(Path(path)) as handle:
        for line in handle:
            if line.startswith("##contig=<ID="):
                contigs.add(
                    line.split("##contig=<ID=", 1)[1]
                    .split(",", 1)[0]
                    .split(">", 1)[0]
                )
            elif line.startswith("##INFO=<ID="):
                info_ids.add(
                    line.split("##INFO=<ID=", 1)[1]
                    .split(",", 1)[0]
                    .split(">", 1)[0]
                )
            elif line.startswith("#CHROM"):
                saw_chrom = True
                break
    return {"contigs": contigs, "info_ids": info_ids, "saw_chrom": saw_chrom}
