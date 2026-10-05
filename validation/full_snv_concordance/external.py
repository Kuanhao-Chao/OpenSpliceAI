"""Manifest-driven harmonization and scoring plans for functional datasets."""

from __future__ import annotations

import csv
import gzip
import io
from itertools import chain
import json
from pathlib import Path
import re
import shlex
import tarfile
from typing import Iterator, Mapping, Optional, Sequence
import xml.etree.ElementTree as ET
import zipfile

from .workflows import atomic_write_json, sha256_file, utc_now
from .vcf import iter_vcf_records


REQUIRED_DATASET_FIELDS = (
    "id",
    "title",
    "article_url",
    "coordinate_build",
    "columns",
)
REQUIRED_COLUMNS = ("chrom", "pos", "ref", "alt", "gene")


def load_source_manifest(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("datasets"), list):
        raise ValueError("external manifest must contain a datasets list")
    identifiers: set[str] = set()
    for dataset in manifest["datasets"]:
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in dataset]
        if missing:
            raise ValueError(f"dataset missing fields: {', '.join(missing)}")
        identifier = str(dataset["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate dataset id: {identifier}")
        identifiers.add(identifier)
        missing_columns = [
            name for name in REQUIRED_COLUMNS if name not in dataset.get("columns", {})
        ]
        if missing_columns:
            raise ValueError(
                f"dataset {identifier} missing column mappings: {', '.join(missing_columns)}"
            )
    return manifest


def _resolve_local_path(
    manifest_path: Path, dataset: Mapping, overrides: Mapping[str, str]
) -> Optional[Path]:
    value = overrides.get(str(dataset["id"]), dataset.get("local_path"))
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def _sha256_source(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = __import__("hashlib").sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_source_files(manifest: Mapping, dataset: Mapping, path: Path) -> dict[str, str]:
    provenance_key = dataset.get("provenance_key")
    if not provenance_key:
        return {}
    provenance = manifest.get("acquisition_provenance", {}).get(provenance_key, {})
    expected = provenance.get("files_sha256", {})
    verified: dict[str, str] = {}
    root = path if path.is_dir() else path.parent
    for name, expected_digest in expected.items():
        candidates = list(root.rglob(str(name))) if root.is_dir() else []
        if not candidates:
            raise ValueError(f"{dataset['id']}: required source file is missing: {name}")
        if len(candidates) != 1:
            raise ValueError(f"{dataset['id']}: source filename is ambiguous: {name}")
        observed = sha256_file(candidates[0])
        if observed != expected_digest:
            raise ValueError(
                f"{dataset['id']}: checksum mismatch for {name}: "
                f"{observed} != {expected_digest}"
            )
        verified[str(name)] = observed
    return verified


def _verified_liftover_chain(
    manifest: Mapping, manifest_path: Path, supplied: Optional[str | Path]
) -> tuple[Optional[Path], Optional[str]]:
    """Resolve a manifest-pinned chain and verify both its path and bytes."""

    if supplied is None:
        return None, None
    non_grch38 = [
        dataset
        for dataset in manifest["datasets"]
        if str(dataset["coordinate_build"]).upper() not in {"GRCH38", "HG38"}
    ]
    if any(not dataset.get("provenance_key") for dataset in non_grch38):
        raise ValueError("every non-GRCh38 dataset must name pinned acquisition provenance")
    provenance_keys = {str(dataset["provenance_key"]) for dataset in non_grch38}
    pinned = []
    for key in sorted(provenance_keys):
        provenance = manifest.get("acquisition_provenance", {}).get(key, {})
        if not provenance.get("liftover_chain") or not provenance.get(
            "liftover_chain_sha256"
        ):
            raise ValueError(f"acquisition provenance {key!r} lacks a pinned liftover chain")
        pinned.append(
            (str(provenance["liftover_chain"]), str(provenance["liftover_chain_sha256"]))
        )
    if len(set(pinned)) != 1:
        raise ValueError(
            "non-GRCh38 datasets must declare exactly one pinned liftover chain path and SHA-256"
        )
    expected_value, expected_sha256 = pinned[0]
    expected = Path(expected_value).expanduser()
    if not expected.is_absolute():
        expected = manifest_path.parent / expected
    expected = expected.resolve()
    observed = Path(supplied).expanduser().resolve()
    if observed != expected:
        raise ValueError(
            f"liftover chain path differs from manifest pin: {observed} != {expected}"
        )
    observed_sha256 = sha256_file(observed)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "liftover chain checksum mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    return observed, observed_sha256


def _dataset_completeness(
    manifest: Mapping, source_status: Sequence[Mapping], allow_incomplete_sensitivity: bool
) -> dict:
    expected = [str(dataset["id"]) for dataset in manifest["datasets"]]
    observed = [str(status.get("id", "")) for status in source_status]
    if observed != expected:
        raise ValueError(
            f"dataset status order/content differs from manifest: observed={observed} expected={expected}"
        )
    incomplete = []
    ready_states = {"ready", "ready_lifted_to_grch38"}
    for status in source_status:
        identifier = str(status["id"])
        rows = int(status.get("rows", 0))
        accepted = int(status.get("accepted", 0))
        rejected = int(status.get("rejected", 0))
        unprocessed = int(status.get("unprocessed", 0))
        reasons = []
        if status.get("state") not in ready_states:
            reasons.append(f"state={status.get('state')}")
        if rows <= 0:
            reasons.append("zero source rows")
        if accepted <= 0:
            reasons.append("zero accepted rows")
        if unprocessed:
            reasons.append(f"{unprocessed} unprocessed rows")
        if rows != accepted + rejected + unprocessed:
            reasons.append(
                f"row accounting mismatch {rows}!={accepted}+{rejected}+{unprocessed}"
            )
        if reasons:
            incomplete.append({"id": identifier, "reasons": reasons})
    complete = not incomplete
    mode = (
        "complete"
        if complete
        else "incomplete_sensitivity"
        if allow_incomplete_sensitivity
        else "incomplete_blocked"
    )
    return {
        "mode": mode,
        "configured_dataset_ids": expected,
        "complete": complete,
        "incomplete_datasets": incomplete,
        "requirements": (
            "Every configured dataset must be acquired, harmonized to GRCh38, contain source "
            "rows, have at least one accepted row, and balance "
            "rows=accepted+rejected+unprocessed."
        ),
    }


def _tabular_rows(dataset: Mapping, path: Path) -> Iterator[tuple[str, Mapping, Mapping]]:
    delimiter = str(dataset.get("delimiter", "\t"))
    if delimiter == "\\t":
        delimiter = "\t"
    columns = dataset["columns"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = set(columns.values()) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, 2):
            yield str(row_number), row, columns


def _splicebench_rows(dataset: Mapping, path: Path) -> Iterator[tuple[str, Mapping, Mapping]]:
    root = path if path.is_dir() else path.parent
    files = sorted(root.glob("*_scored.txt"))
    if not files:
        files = sorted(root.rglob("*_scored.txt"))
    include_files = set(str(name) for name in dataset.get("include_files", ()))
    if include_files:
        files = [table for table in files if table.name in include_files]
        missing_files = include_files - {table.name for table in files}
        if missing_files:
            raise ValueError(f"{path}: missing requested SpliceBench files {sorted(missing_files)}")
    if not files:
        raise ValueError(f"{path}: no SpliceBench *_scored.txt tables found")
    columns = {
        "chrom": "chrom",
        "pos": "pos",
        "ref": "ref",
        "alt": "alt",
        "gene": "gene",
        "transcript": "transcript",
        "label": "label",
        "outcome": "outcome",
        "cohort": "cohort",
    }
    for table in files:
        with table.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or ())
            required = {"chrom", "hg38_pos", "ref", "alt"}
            if not required <= fields:
                raise ValueError(f"{table}: missing {sorted(required - fields)}")
            fallback_gene = table.name.split("_", 1)[0].upper()
            for row_number, raw in enumerate(reader, 2):
                position = raw.get("hg38_pos", "")
                try:
                    position = str(int(float(position)))
                except (TypeError, ValueError):
                    position = ""
                gene = (
                    raw.get("gene_name")
                    or raw.get("gene_name_x")
                    or raw.get("gene_name_y")
                    or fallback_gene
                )
                transcript = (
                    raw.get("transcript_id")
                    or raw.get("transcript")
                    or raw.get("transcript_id_beta")
                    or ""
                )
                label = (
                    raw.get("sdv_fc2_cat")
                    or raw.get("lit_sdv_noptt_bool")
                    or raw.get("aberrant_spl_overall")
                    or raw.get("sdv")
                    or ""
                )
                outcome = (
                    raw.get("measured")
                    or raw.get("measured_abs")
                    or raw.get("aberrant_spl_overall")
                    or raw.get("sdv_fc2")
                    or ""
                )
                row = {
                    "chrom": raw.get("chrom", ""),
                    "pos": position,
                    "ref": raw.get("ref", ""),
                    "alt": raw.get("alt", ""),
                    "gene": gene,
                    "transcript": transcript,
                    "label": label,
                    "outcome": outcome,
                    "cohort": table.stem,
                }
                yield f"{table.name}:{row_number}", row, columns


def _locate_spip_files(path: Path) -> tuple[Path, Path]:
    if path.is_file() and path.name.endswith(".tar.gz"):
        archive = path
        candidates = list(path.parent.glob("*-converted.vcf"))
    else:
        candidates_tar = list(path.rglob("HUMU-43-2308-s003.csv.tar.gz"))
        candidates = list(path.rglob("HUMU-43-2308-s003-converted.vcf"))
        if not candidates_tar:
            raise ValueError(f"{path}: SPiP CSV archive not found")
        archive = candidates_tar[0]
    if not candidates:
        raise ValueError(f"{path}: SPiP converted GRCh38 VCF not found")
    return archive, candidates[0]


def _spip_rows(dataset: Mapping, path: Path) -> Iterator[tuple[str, Mapping, Mapping]]:
    archive, vcf = _locate_spip_files(path)
    experimental: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"{archive}: expected one CSV member")
        raw_handle = bundle.extractfile(members[0])
        if raw_handle is None:
            raise ValueError(f"{archive}: cannot read CSV member")
        handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
        header = None
        for line in handle:
            if line.startswith("variant\tuse\tvarID\t"):
                header = line
                break
        if header is None:
            raise ValueError(f"{archive}: SPiP table header not found")
        reader = csv.DictReader(chain((header,), handle), delimiter="\t")
        for row in reader:
            # Only RNA/minigene-evidenced variants; common SNP controls are not
            # experimental negatives.
            if row.get("variant") != "var":
                continue
            experimental[str(row.get("varID", ""))] = row
    columns = {
        "chrom": "chrom",
        "pos": "pos",
        "ref": "ref",
        "alt": "alt",
        "gene": "gene",
        "transcript": "transcript",
        "label": "label",
        "outcome": "outcome",
        "cohort": "cohort",
    }
    matched: set[str] = set()
    for row_number, record in enumerate(iter_vcf_records(vcf), 1):
        identifier = record.columns[2]
        evidence = experimental.get(identifier)
        if evidence is None:
            continue
        matched.add(identifier)
        row = {
            "chrom": record.key.chrom,
            "pos": str(record.key.pos),
            "ref": record.key.ref,
            "alt": record.key.alt,
            "gene": record.info.get("GeneName") or evidence.get("gene", ""),
            "transcript": evidence.get("transcript", ""),
            "label": evidence.get("class_splice", ""),
            "outcome": evidence.get("observation", ""),
            "cohort": evidence.get("source", "RNA_or_minigene"),
        }
        yield f"{vcf.name}:{row_number}", row, columns
    missing = set(experimental) - matched
    # Preserve conversion failures in the rejected-flow table instead of
    # silently shrinking the experimental denominator.
    for identifier in sorted(missing):
        evidence = experimental[identifier]
        yield (
            f"unmapped_varID:{identifier}",
            {
                "chrom": "",
                "pos": "",
                "ref": "",
                "alt": "",
                "gene": evidence.get("gene", ""),
                "transcript": evidence.get("transcript", ""),
                "label": evidence.get("class_splice", ""),
                "outcome": evidence.get("observation", ""),
                "cohort": evidence.get("source", "RNA_or_minigene"),
            },
            columns,
        )


def _xlsx_cell_rows(path: Path, sheet_number: int) -> Iterator[list[str]]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as workbook:
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(namespace + "t"))
            for item in shared_root
        ]
        root = ET.fromstring(workbook.read(f"xl/worksheets/sheet{sheet_number}.xml"))
        for xml_row in root.iter(namespace + "row"):
            values: dict[int, str] = {}
            for cell in xml_row.findall(namespace + "c"):
                reference = cell.attrib["r"]
                match = re.match(r"[A-Z]+", reference)
                if match is None:
                    continue
                column = 0
                for character in match.group(0):
                    column = column * 26 + ord(character) - 64
                value_node = cell.find(namespace + "v")
                value = "" if value_node is None else str(value_node.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values[column - 1] = value
            if values:
                yield [values.get(index, "") for index in range(max(values) + 1)]


def _riepe_rows(dataset: Mapping, path: Path) -> Iterator[tuple[str, Mapping, Mapping]]:
    workbook = path if path.is_file() else path / "data" / "variant_scores.xlsx"
    if not workbook.exists():
        candidates = list(path.rglob("variant_scores.xlsx")) if path.is_dir() else []
        if not candidates:
            raise ValueError(f"{path}: Riepe variant_scores.xlsx not found")
        workbook = candidates[0]
    sheets = {
        "riepe_abca4_noncanonical": (1, "chr1", "ABCA4", "ABCA4_NCSS"),
        "riepe_abca4_deep_intronic": (2, "chr1", "ABCA4", "ABCA4_DI"),
        "riepe_mybpc3": (3, "chr11", "MYBPC3", "MYBPC3_NCSS"),
    }
    if str(dataset["id"]) not in sheets:
        raise ValueError(f"unsupported Riepe dataset id: {dataset['id']}")
    sheet_number, chrom, gene, cohort = sheets[str(dataset["id"])]
    vcf_name = f"{cohort}_variants.vcf"
    vcf_path = (path / "data" / vcf_name) if path.is_dir() else path.parent / vcf_name
    if not vcf_path.exists() and path.is_dir():
        matches = list(path.rglob(vcf_name))
        if matches:
            vcf_path = matches[0]
    if not vcf_path.exists():
        raise ValueError(f"{path}: required Riepe VCF {vcf_name} not found")
    vcf_records = list(iter_vcf_records(vcf_path))
    rows = iter(_xlsx_cell_rows(workbook, sheet_number))
    header = next(rows)
    indices = {name: header.index(name) for name in ("cDNA variant", "genomic variant", "% Mutant RNA", "affects")}
    columns = {name: name for name in REQUIRED_COLUMNS}
    columns.update({"transcript": "transcript", "label": "label", "outcome": "outcome", "cohort": "cohort"})
    buffered: list[tuple[str, Mapping, Mapping]] = []
    for row_number, values in enumerate(rows, 2):
        genomic = values[indices["genomic variant"]]
        if not genomic:
            continue
        match = re.fullmatch(r"g\.(\d+)([ACGT])>([ACGT])", genomic)
        source_vcf_record = vcf_records[len(buffered)]
        mutant_rna = float(values[indices["% Mutant RNA"]] or 0)
        # This is the threshold used for all three sheets by the publication's
        # analysis_variants.py and functions.py. MYBPC3 happens to be encoded
        # as 0/100, but uses the same published rule.
        label = "splice_altering" if mutant_rna > 20 else "neutral"
        row = {
            "chrom": chrom,
            "pos": match.group(1) if match else str(source_vcf_record.key.pos),
            "ref": match.group(2) if match else source_vcf_record.key.ref,
            "alt": match.group(3) if match else source_vcf_record.key.alt,
            "gene": gene,
            "transcript": values[indices["cDNA variant"]],
            "label": label,
            "outcome": values[indices["% Mutant RNA"]],
            "cohort": cohort,
        }
        buffered.append((f"{workbook.name}:{sheet_number}:{row_number}", row, columns))
    if len(buffered) != len(vcf_records):
        raise ValueError(
            f"{cohort}: workbook/VCF row-count mismatch "
            f"({len(buffered)} != {len(vcf_records)})"
        )
    yield from buffered


def _dataset_rows(dataset: Mapping, path: Path) -> Iterator[tuple[str, Mapping, Mapping]]:
    adapter = str(dataset.get("adapter", "tabular"))
    if adapter == "tabular":
        yield from _tabular_rows(dataset, path)
    elif adapter == "splicebench_scored_directory":
        yield from _splicebench_rows(dataset, path)
    elif adapter == "spip_functional_archive":
        yield from _spip_rows(dataset, path)
    elif adapter == "riepe_xlsx":
        yield from _riepe_rows(dataset, path)
    else:
        raise ValueError(f"unknown external dataset adapter: {adapter}")


def _normalize_chrom(value: str) -> str:
    chrom = value.strip()
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"M", "MT"}:
        return "chrM"
    return f"chr{chrom}"


def _reverse_complement(base: str) -> str:
    return base.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class ChainLiftOver:
    """Minimal UCSC-chain point mapper for allele-aware SNV harmonization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.intervals: dict[str, list[tuple]] = {}
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as handle:
            header = None
            target_cursor = query_cursor = 0
            for line_number, raw in enumerate(handle, 1):
                line = raw.strip()
                if not line:
                    header = None
                    continue
                if line.startswith("chain "):
                    fields = line.split()
                    if len(fields) != 13:
                        raise ValueError(f"{self.path}:{line_number}: malformed chain header")
                    (
                        _word,
                        score,
                        target_name,
                        _target_size,
                        target_strand,
                        target_start,
                        _target_end,
                        query_name,
                        query_size,
                        query_strand,
                        query_start,
                        _query_end,
                        chain_id,
                    ) = fields
                    if target_strand != "+":
                        raise ValueError("point lifter requires '+' chain target strand")
                    header = (
                        target_name,
                        query_name,
                        int(query_size),
                        query_strand,
                        int(score),
                        chain_id,
                    )
                    target_cursor = int(target_start)
                    query_cursor = int(query_start)
                    continue
                if header is None:
                    raise ValueError(f"{self.path}:{line_number}: block outside chain")
                fields = line.split()
                if len(fields) not in (1, 3):
                    raise ValueError(f"{self.path}:{line_number}: malformed chain block")
                size = int(fields[0])
                target_name, query_name, query_size, query_strand, score, chain_id = header
                self.intervals.setdefault(_normalize_chrom(target_name), []).append(
                    (
                        target_cursor,
                        target_cursor + size,
                        _normalize_chrom(query_name),
                        query_cursor,
                        query_size,
                        query_strand,
                        score,
                        chain_id,
                    )
                )
                target_cursor += size
                query_cursor += size
                if len(fields) == 3:
                    target_cursor += int(fields[1])
                    query_cursor += int(fields[2])
        for intervals in self.intervals.values():
            intervals.sort(key=lambda value: (value[0], value[1], -value[6]))

    def map_snv(self, chrom: str, pos: int, ref: str, alt: str) -> tuple[str, int, str, str]:
        coordinate = pos - 1
        matches = []
        for start, end, query_name, query_start, query_size, strand, score, chain_id in self.intervals.get(
            _normalize_chrom(chrom), ()
        ):
            if start <= coordinate < end:
                offset = coordinate - start
                if strand == "+":
                    mapped = query_start + offset
                    mapped_ref, mapped_alt = ref, alt
                else:
                    mapped = query_size - (query_start + offset) - 1
                    mapped_ref, mapped_alt = _reverse_complement(ref), _reverse_complement(alt)
                matches.append((score, chain_id, query_name, mapped + 1, mapped_ref, mapped_alt))
        if not matches:
            raise ValueError("liftover_unmapped")
        matches.sort(reverse=True)
        best_score = matches[0][0]
        best = {value[2:] for value in matches if value[0] == best_score}
        if len(best) != 1:
            raise ValueError("liftover_ambiguous")
        return next(iter(best))


def _reference_reader(reference_fasta: Optional[str | Path]):
    if reference_fasta is None:
        return None
    try:
        import pysam
    except ImportError as exc:
        raise RuntimeError("REF verification requires pysam") from exc
    return pysam.FastaFile(str(reference_fasta))


def harmonize_external(
    manifest_path: str | Path,
    output_dir: str | Path,
    reference_fasta: Optional[str | Path] = None,
    annotation: Optional[str | Path] = None,
    openspliceai_models: Sequence[str] = (),
    include_spliceai: bool = True,
    source_overrides: Optional[Mapping[str, str]] = None,
    liftover_chain: Optional[str | Path] = None,
    score_mask: int = 0,
    allow_incomplete_sensitivity: bool = False,
) -> dict:
    if score_mask not in (0, 1):
        raise ValueError("score_mask must be 0 (unmasked) or 1 (masked)")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_source_manifest(manifest_path)
    liftover_chain, liftover_chain_sha256 = _verified_liftover_chain(
        manifest, manifest_path, liftover_chain
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    fasta = _reference_reader(reference_fasta)
    reference_contigs = (
        list(zip(fasta.references, fasta.lengths)) if fasta is not None else []
    )
    lifter = ChainLiftOver(liftover_chain) if liftover_chain else None
    source_overrides = source_overrides or {}
    accepted: list[dict] = []
    rejected: list[dict] = []
    source_status: list[dict] = []
    try:
        for dataset in manifest["datasets"]:
            identifier = str(dataset["id"])
            local_path = _resolve_local_path(manifest_path, dataset, source_overrides)
            status = {
                "id": identifier,
                "article_url": dataset["article_url"],
                "local_path": str(local_path) if local_path else None,
                "coordinate_build": dataset["coordinate_build"],
                "state": "ready",
                "rows": 0,
                "accepted": 0,
                "rejected": 0,
                "unprocessed": 0,
                "sha256": None,
                "lifted_rows": 0,
                "known_source_issues": dataset.get("known_source_issues", []),
                "verified_files_sha256": {},
            }
            if local_path is None or not local_path.exists():
                status["state"] = "requires_acquisition"
                source_status.append(status)
                continue
            status["sha256"] = _sha256_source(local_path)
            expected_source_sha = dataset.get("sha256")
            if expected_source_sha and status["sha256"] != expected_source_sha:
                raise ValueError(
                    f"{identifier}: source tree checksum mismatch: expected "
                    f"{expected_source_sha}, observed {status['sha256']}"
                )
            status["verified_files_sha256"] = _verify_source_files(
                manifest, dataset, local_path
            )
            if str(dataset["coordinate_build"]).upper() not in {
                "GRCH38",
                "HG38",
            }:
                if lifter is None:
                    # Parse and cross-check source assets even though they cannot
                    # enter the GRCh38 output until allele-aware liftover is supplied.
                    status["rows"] = sum(1 for _ in _dataset_rows(dataset, local_path))
                    status["unprocessed"] = status["rows"]
                    status["state"] = "requires_allele_aware_liftover"
                    source_status.append(status)
                    continue
                if fasta is None:
                    raise ValueError(
                        "--reference-fasta is required with --liftover-chain for allele validation"
                    )
                status["state"] = "ready_lifted_to_grch38"
            for row_number, row, columns in _dataset_rows(dataset, local_path):
                status["rows"] += 1
                reason = None
                try:
                    chrom = _normalize_chrom(str(row[columns["chrom"]]))
                    pos = int(row[columns["pos"]])
                    ref = str(row[columns["ref"]]).upper().strip()
                    alt = str(row[columns["alt"]]).upper().strip()
                    gene = str(row[columns["gene"]]).strip()
                    if (
                        pos < 1
                        or ref not in "ACGT"
                        or alt not in "ACGT"
                        or len(ref) != 1
                        or len(alt) != 1
                    ):
                        reason = "not_biallelic_snv"
                    elif ref == alt:
                        reason = "ref_equals_alt"
                    elif not gene:
                        reason = "missing_gene"
                    if (
                        reason is None
                        and str(dataset["coordinate_build"]).upper()
                        not in {"GRCH38", "HG38"}
                    ):
                        try:
                            chrom, pos, ref, alt = lifter.map_snv(chrom, pos, ref, alt)
                            status["lifted_rows"] += 1
                        except ValueError as exc:
                            reason = str(exc)
                    if reason is None and fasta is not None:
                        try:
                            observed = fasta.fetch(chrom, pos - 1, pos).upper()
                        except (KeyError, ValueError):
                            reason = "reference_contig_or_position_missing"
                        else:
                            if observed != ref:
                                reason = "reference_allele_mismatch"
                except (KeyError, TypeError, ValueError):
                    reason = "invalid_required_field"
                base = {
                    "dataset": identifier,
                    "source_row": row_number,
                    "reason": reason or "accepted",
                }
                if reason:
                    rejected.append(base)
                    status["rejected"] += 1
                    continue
                record = {
                    **base,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "transcript": str(row.get(columns.get("transcript", ""), "")),
                    "label": str(row.get(columns.get("label", ""), "")),
                    "outcome": str(row.get(columns.get("outcome", ""), "")),
                    "cohort": str(row.get(columns.get("cohort", ""), identifier)),
                }
                accepted.append(record)
                status["accepted"] += 1
            source_status.append(status)
    finally:
        if fasta is not None:
            fasta.close()

    harmonized_path = output_dir / "harmonized.tsv"
    fieldnames = [
        "dataset",
        "source_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "transcript",
        "label",
        "outcome",
        "cohort",
        "reason",
    ]
    with harmonized_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(accepted)
    rejected_path = output_dir / "rejected.tsv"
    with rejected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "source_row", "reason"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rejected)

    variants = sorted(
        {(r["chrom"], int(r["pos"]), r["ref"], r["alt"]) for r in accepted},
        key=lambda value: (value[0], value[1], value[2], value[3]),
    )
    vcf_path = output_dir / "validation_variants.hg38.vcf"
    with vcf_path.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=OpenSpliceAI-full-snv-concordance\n")
        # Both pysam and the original SpliceAI writer require every record
        # contig to be declared in the input header.  Prefer authoritative
        # FASTA lengths; a no-reference preparation still emits ID-only
        # declarations so its VCF remains round-trippable.
        if reference_contigs:
            for contig, length in reference_contigs:
                handle.write(f"##contig=<ID={contig},length={length}>\n")
        else:
            for contig in sorted({variant[0] for variant in variants}):
                handle.write(f"##contig=<ID={contig}>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, ref, alt in variants:
            handle.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")

    commands = build_scoring_plan(
        vcf_path,
        output_dir / "scores",
        reference_fasta,
        annotation,
        openspliceai_models,
        include_spliceai,
        score_mask,
    )
    dataset_completeness = _dataset_completeness(
        manifest, source_status, allow_incomplete_sensitivity
    )
    result = {
        "schema_version": 2,
        "kind": "external-validation-preparation",
        "created_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_fasta": str(reference_fasta) if reference_fasta else None,
        "reference_sha256": sha256_file(reference_fasta) if reference_fasta else None,
        "liftover_chain": str(liftover_chain) if liftover_chain else None,
        "liftover_chain_sha256": liftover_chain_sha256,
        "sources": source_status,
        "dataset_completeness": dataset_completeness,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "unique_variants": len(variants),
        "harmonized_tsv": str(harmonized_path),
        "rejected_tsv": str(rejected_path),
        "validation_vcf": str(vcf_path),
        "scoring_commands": commands,
        "score_mask": score_mask,
        "notes": [
            "Commands are a plan and are not executed by validate-external.",
            "Non-GRCh38 sources fail closed until allele-aware liftover is supplied.",
            (
                "All configured external datasets are complete and eligible for primary evaluation."
                if dataset_completeness["complete"]
                else (
                    "This plan is explicitly labelled as an incomplete-dataset sensitivity analysis."
                    if dataset_completeness["mode"] == "incomplete_sensitivity"
                    else "Scoring is blocked until every configured dataset is complete."
                )
            ),
            "Functional accuracy must be reported separately from predictor concordance.",
            (
                "External functional discrimination uses unmasked delta scores, the "
                "conventional primary estimand for max-DS thresholds."
                if score_mask == 0
                else "External scoring is masked (-M 1) and must be labelled as a sensitivity analysis."
            ),
        ],
    }
    atomic_write_json(output_dir / "validation_plan.json", result)
    return result


def build_scoring_plan(
    input_vcf: Path,
    score_dir: Path,
    reference_fasta: Optional[str | Path],
    annotation: Optional[str | Path],
    openspliceai_models: Sequence[str],
    include_spliceai: bool,
    score_mask: int = 0,
) -> list[dict]:
    if score_mask not in (0, 1):
        raise ValueError("score_mask must be 0 (unmasked) or 1 (masked)")
    commands: list[dict] = []
    score_dir.mkdir(parents=True, exist_ok=True)
    deterministic_environment = {
        "NVIDIA_TF32_OVERRIDE": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OSAI_TF32": "0",
        "OSAI_CUDNN_BENCH": "0",
        "OSAI_DETERMINISTIC": "1",
        "PYTHONHASHSEED": "0",
    }
    if include_spliceai:
        if reference_fasta:
            command = [
                "spliceai",
                "-I",
                str(input_vcf),
                "-O",
                str(score_dir / "spliceai.vcf"),
                "-R",
                str(reference_fasta),
                "-A",
                "grch38",
                "-D",
                "50",
                "-M",
                str(score_mask),
            ]
            commands.append(
                {
                    "name": "spliceai_official_five_model_ensemble",
                    "ready": True,
                    "argv": command,
                    "environment": {
                        **deterministic_environment,
                        "TF_DETERMINISTIC_OPS": "1",
                        "TF_CUDNN_DETERMINISTIC": "1",
                    },
                    "shell": shlex.join(
                        [
                            "env",
                            *(f"{key}={value}" for key, value in deterministic_environment.items()),
                            "TF_DETERMINISTIC_OPS=1",
                            "TF_CUDNN_DETERMINISTIC=1",
                            *command,
                        ]
                    ),
                }
            )
        else:
            commands.append(
                {
                    "name": "spliceai_official_five_model_ensemble",
                    "ready": False,
                    "missing": ["reference_fasta"],
                }
            )
    for specification in openspliceai_models:
        if "=" not in specification:
            raise ValueError("--openspliceai-model values must be NAME=PATH")
        name, model = specification.split("=", 1)
        missing = []
        if not reference_fasta:
            missing.append("reference_fasta")
        if not annotation:
            missing.append("annotation")
        if missing:
            commands.append({"name": name, "ready": False, "model": model, "missing": missing})
            continue
        command = [
            "openspliceai",
            "variant",
            "-R",
            str(reference_fasta),
            "-A",
            str(annotation),
            "--model",
            model,
            "-I",
            str(input_vcf),
            "-O",
            str(score_dir / f"openspliceai_{name}.vcf"),
            "--precision",
            "5",
            "-D",
            "50",
            "-f",
            "10000",
            "-M",
            str(score_mask),
            "-b",
            "128",
        ]
        commands.append(
            {
                "name": name,
                "ready": True,
                "argv": command,
                "environment": deterministic_environment,
                "shell": shlex.join(
                    [
                        "env",
                        *(f"{key}={value}" for key, value in deterministic_environment.items()),
                        *command,
                    ]
                ),
            }
        )
    return commands
