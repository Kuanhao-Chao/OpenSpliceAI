"""Freeze, verify, and execute the small external-validation scoring plan.

The genome-wide workflow intentionally never executes commands found in a JSON
file.  This module follows the same rule: it accepts only the seven expected
predictors, validates their arguments, replaces their executables with pinned
absolute paths, and records cryptographic provenance before a Slurm array can
start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from typing import Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 3
OFFICIAL_NAME = "spliceai_official_five_model_ensemble"
OSAI_NAMES = ("rs10", "rs11", "rs12", "rs13", "rs14", "ensemble")
EXPECTED_NAMES = (OFFICIAL_NAME,) + OSAI_NAMES
EXPECTED_INFO = {OFFICIAL_NAME: "SpliceAI", **{name: "OpenSpliceAI" for name in OSAI_NAMES}}
MIN_ANNOTATED_RECORD_FRACTION = 0.95
MIN_PREDICTION_COVERAGE = 0.90
RECEIPT_SCHEMA = "openspliceai.external_score_receipt"
RECEIPT_SCHEMA_VERSION = 1
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "OSAI_CUDNN_BENCH": "0",
    "OSAI_DETERMINISTIC": "1",
    "OSAI_TF32": "0",
    "PYTHONHASHSEED": "0",
    "TF_CUDNN_DETERMINISTIC": "1",
    "TF_DETERMINISTIC_OPS": "1",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a file or a directory tree, excluding generated Python caches."""

    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise ValueError(f"cannot fingerprint empty directory: {path}")
    for child in files:
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _path(value: object, label: str, *, directory: Optional[bool] = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    path = path.resolve()
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if directory is None and not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _flag(argv: Sequence[str], flag: str) -> tuple[int, str]:
    indices = [index for index, value in enumerate(argv[:-1]) if value == flag]
    if len(indices) != 1:
        raise ValueError(f"command must contain exactly one {flag}: {argv!r}")
    index = indices[0]
    return index + 1, argv[index + 1]


def _replace_flag(argv: Sequence[str], flag: str, value: str) -> list[str]:
    result = list(argv)
    index, _ = _flag(result, flag)
    result[index] = value
    return result


def _load_plan(path: str | Path) -> tuple[Path, dict]:
    plan_path = _path(path, "validation plan", directory=False)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("kind") != "external-validation-preparation":
        raise ValueError("not an external-validation preparation plan")
    if not isinstance(plan.get("scoring_commands"), list):
        raise ValueError("validation plan lacks a scoring_commands list")
    return plan_path, plan


def _validate_dataset_completeness(plan: Mapping) -> tuple[dict, list[dict]]:
    completeness = plan.get("dataset_completeness")
    sources = plan.get("sources")
    if not isinstance(completeness, Mapping) or not isinstance(sources, list):
        raise ValueError("validation plan lacks dataset completeness/source provenance")
    expected_ids = completeness.get("configured_dataset_ids")
    if not isinstance(expected_ids, list) or not expected_ids or not all(
        isinstance(value, str) and value for value in expected_ids
    ):
        raise ValueError("validation plan has invalid configured dataset IDs")
    if not all(isinstance(source, Mapping) for source in sources):
        raise ValueError("validation plan has a malformed dataset status")
    observed_ids = [str(source.get("id", "")) for source in sources]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("dataset statuses do not exactly match configured dataset IDs")
    ready_states = {"ready", "ready_lifted_to_grch38"}
    incomplete = []
    normalized_sources = []
    for source in sources:
        normalized = dict(source)
        try:
            rows = int(normalized.get("rows", 0))
            accepted = int(normalized.get("accepted", 0))
            rejected = int(normalized.get("rejected", 0))
            unprocessed = int(normalized.get("unprocessed", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dataset {normalized.get('id')}: invalid row counts") from exc
        if (
            min(rows, accepted, rejected, unprocessed) < 0
            or accepted + rejected + unprocessed != rows
        ):
            raise ValueError(f"dataset {normalized.get('id')}: inconsistent row accounting")
        reasons = []
        if normalized.get("state") not in ready_states:
            reasons.append(f"state={normalized.get('state')}")
        if rows == 0:
            reasons.append("zero source rows")
        if accepted == 0:
            reasons.append("zero accepted rows")
        if unprocessed:
            reasons.append(f"{unprocessed} unprocessed rows")
        if reasons:
            incomplete.append({"id": normalized["id"], "reasons": reasons})
        normalized_sources.append(normalized)
    complete = not incomplete
    declared_complete = completeness.get("complete") is True
    mode = completeness.get("mode")
    if declared_complete != complete:
        raise ValueError("declared dataset completeness disagrees with source statuses")
    if not complete and mode != "incomplete_sensitivity":
        detail = ", ".join(
            f"{item['id']} ({'; '.join(item['reasons'])})" for item in incomplete
        )
        raise ValueError(
            "primary external scoring requires every configured dataset to be acquired "
            f"with nonzero accepted rows; incomplete: {detail}"
        )
    if complete and mode != "complete":
        raise ValueError("complete source corpus must declare dataset completeness mode=complete")
    normalized = dict(completeness)
    if normalized.get("incomplete_datasets") != incomplete:
        raise ValueError("declared incomplete dataset details disagree with source statuses")
    return normalized, normalized_sources


def parse_plan(
    plan_path: str | Path,
    *,
    python_bin: str | Path,
    spliceai_bin: str | Path,
    openspliceai_bin: str | Path,
    official_package_dir: str | Path,
) -> dict:
    """Validate and normalize the seven permitted scoring commands."""

    plan_path, plan = _load_plan(plan_path)
    dataset_completeness, source_statuses = _validate_dataset_completeness(plan)
    manifest_path = _path(plan.get("manifest"), "external source manifest", directory=False)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != plan.get("manifest_sha256"):
        raise ValueError("external source manifest does not match validation_plan.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_ids = [str(dataset.get("id", "")) for dataset in manifest.get("datasets", [])]
    if manifest_ids != dataset_completeness["configured_dataset_ids"]:
        raise ValueError("configured dataset IDs do not exactly match the source manifest")
    python_bin = _path(python_bin, "Python executable", directory=False)
    spliceai_bin = _path(spliceai_bin, "SpliceAI executable", directory=False)
    openspliceai_bin = _path(openspliceai_bin, "OpenSpliceAI executable", directory=False)
    official_package_dir = _path(
        official_package_dir, "official SpliceAI package", directory=True
    )
    for executable in (python_bin, spliceai_bin, openspliceai_bin):
        if not os.access(executable, os.X_OK):
            raise ValueError(f"executable bit is not set: {executable}")

    entries: dict[str, Mapping] = {}
    for raw in plan["scoring_commands"]:
        if not isinstance(raw, Mapping):
            raise ValueError("each scoring command must be a JSON object")
        name = str(raw.get("name", ""))
        if name not in EXPECTED_NAMES:
            raise ValueError(f"unexpected predictor in scoring plan: {name!r}")
        if name in entries:
            raise ValueError(f"duplicate predictor in scoring plan: {name}")
        if raw.get("ready") is not True:
            raise ValueError(f"predictor is not ready: {name}")
        if not isinstance(raw.get("argv"), list) or not all(
            isinstance(value, str) for value in raw["argv"]
        ):
            raise ValueError(f"predictor {name} lacks a string argv list")
        entries[name] = raw
    missing = set(EXPECTED_NAMES) - set(entries)
    if missing:
        raise ValueError(f"scoring plan is missing predictors: {', '.join(sorted(missing))}")

    source_input = _path(plan.get("validation_vcf"), "validation VCF", directory=False)
    reference = _path(plan.get("reference_fasta"), "reference FASTA", directory=False)
    planned_reference_sha = plan.get("reference_sha256")
    if planned_reference_sha and sha256_file(reference) != planned_reference_sha:
        raise ValueError("reference FASTA does not match validation_plan.json")
    reference_fai = _path(
        Path(str(reference) + ".fai"), "reference FASTA index", directory=False
    )

    tasks: list[dict] = []
    score_masks: set[int] = set()
    annotations: set[Path] = set()
    outputs: set[Path] = set()
    individual_models: dict[str, Path] = {}
    ensemble_model: Optional[Path] = None
    for name in EXPECTED_NAMES:
        argv = list(entries[name]["argv"])
        if len(argv) < 2:
            raise ValueError(f"empty command for predictor {name}")
        if name == OFFICIAL_NAME:
            expected_flags = ["-I", "-O", "-R", "-A", "-D", "-M"]
            if len(argv) != 1 + 2 * len(expected_flags) or argv[1::2] != expected_flags:
                raise ValueError(
                    "official SpliceAI command must contain only the prespecified "
                    f"flags in order: {expected_flags}"
                )
        else:
            expected_flags = [
                "-R",
                "-A",
                "--model",
                "-I",
                "-O",
                "--precision",
                "-D",
                "-f",
                "-M",
                "-b",
            ]
            if (
                len(argv) != 2 + 2 * len(expected_flags)
                or argv[1:2] != ["variant"]
                or argv[2::2] != expected_flags
            ):
                raise ValueError(
                    f"{name} command must contain only the prespecified flags "
                    f"in order: {expected_flags}"
                )
        input_index, input_value = _flag(argv, "-I")
        output_index, output_value = _flag(argv, "-O")
        _, reference_value = _flag(argv, "-R")
        _, mask_value = _flag(argv, "-M")
        try:
            score_mask = int(mask_value)
        except ValueError as exc:
            raise ValueError(f"{name} -M must be 0 or 1") from exc
        if score_mask not in (0, 1):
            raise ValueError(f"{name} -M must be 0 or 1")
        score_masks.add(score_mask)
        if _path(input_value, f"{name} input", directory=False) != source_input:
            raise ValueError(f"{name} does not use validation_vcf")
        if _path(reference_value, f"{name} reference", directory=False) != reference:
            raise ValueError(f"{name} uses a different reference FASTA")
        output = Path(output_value).expanduser()
        if not output.is_absolute():
            raise ValueError(f"{name} output must be absolute")
        output = output.resolve(strict=False)
        if output == source_input or output in outputs:
            raise ValueError(f"unsafe or duplicate output for {name}: {output}")
        if not output.parent.is_dir():
            raise ValueError(f"planned score directory does not exist: {output.parent}")
        outputs.add(output)
        if name == OFFICIAL_NAME:
            if Path(argv[0]).name != "spliceai":
                raise ValueError("official command executable must be spliceai")
            argv[0] = str(spliceai_bin)
            if _flag(argv, "-A")[1].lower() != "grch38":
                raise ValueError("official SpliceAI command must use -A grch38")
            if _flag(argv, "-D")[1] != "50":
                raise ValueError("official SpliceAI command must use -D 50")
            model = None
            kind = "spliceai"
        else:
            if Path(argv[0]).name != "openspliceai" or argv[1:2] != ["variant"]:
                raise ValueError(f"{name} is not an openspliceai variant command")
            argv[0] = str(openspliceai_bin)
            _, annotation_value = _flag(argv, "-A")
            annotation = _path(annotation_value, f"{name} annotation", directory=False)
            annotations.add(annotation)
            expected_values = {
                "--precision": "5",
                "-D": "50",
                "-f": "10000",
                "-b": "128",
            }
            for flag, expected_value in expected_values.items():
                if _flag(argv, flag)[1] != expected_value:
                    raise ValueError(
                        f"{name} command must use {flag} {expected_value}"
                    )
            _, model_value = _flag(argv, "--model")
            model = _path(model_value, f"{name} model")
            if name == "ensemble":
                if not model.is_dir():
                    raise ValueError("ensemble model must be a directory")
                ensemble_model = model
            else:
                if not model.is_file() or name not in model.name:
                    raise ValueError(f"{name} must point to its named checkpoint file")
                individual_models[name] = model
            kind = "openspliceai"
        argv[input_index] = str(source_input)
        argv[output_index] = str(output)
        tasks.append(
            {
                "index": len(tasks),
                "name": name,
                "display_name": "SpliceAI" if name == OFFICIAL_NAME else name,
                "kind": kind,
                "info_key": EXPECTED_INFO[name],
                "argv": argv,
                "final_output": str(output),
                "model": str(model) if model else None,
                "score_mask": score_mask,
            }
        )
    if len(score_masks) != 1:
        raise ValueError("all predictors must use the same masking mode")
    score_mask = next(iter(score_masks))
    if plan.get("score_mask") is not None and int(plan["score_mask"]) != score_mask:
        raise ValueError("declared score_mask disagrees with scoring commands")
    if len(annotations) != 1:
        raise ValueError("all OpenSpliceAI predictors must use one annotation file")
    if ensemble_model is None or set(individual_models) != set(OSAI_NAMES[:-1]):
        raise ValueError("incomplete OpenSpliceAI model set")
    for name, model in individual_models.items():
        expected = ensemble_model / f"model_10000nt_{name}.pt"
        if model != expected.resolve():
            raise ValueError(f"{name} checkpoint is not the expected member of ensemble directory")

    official_models = []
    for number in range(1, 6):
        model = official_package_dir / "models" / f"spliceai{number}.h5"
        official_models.append(
            str(_path(model, f"official SpliceAI model {number}", directory=False))
        )
    harmonized = _path(plan.get("harmonized_tsv"), "harmonized TSV", directory=False)
    return {
        "plan_path": str(plan_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "source_input": str(source_input),
        "reference": str(reference),
        "reference_fai": str(reference_fai),
        "annotation": str(next(iter(annotations))),
        "harmonized": str(harmonized),
        "official_package_dir": str(official_package_dir),
        "official_models": official_models,
        "python_bin": str(python_bin),
        "spliceai_bin": str(spliceai_bin),
        "openspliceai_bin": str(openspliceai_bin),
        "tasks": tasks,
        "score_mask": score_mask,
        "dataset_completeness": dataset_completeness,
        "source_statuses": source_statuses,
    }


def _read_fai(path: Path) -> list[tuple[str, int]]:
    contigs: list[tuple[str, int]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: malformed FASTA index")
            name = fields[0]
            if not name or name in seen or any(char in name for char in ",<>"):
                raise ValueError(f"{path}:{line_number}: invalid/duplicate contig {name!r}")
            try:
                length = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid contig length") from exc
            if length < 1:
                raise ValueError(f"{path}:{line_number}: invalid contig length")
            seen.add(name)
            contigs.append((name, length))
    if not contigs:
        raise ValueError(f"empty FASTA index: {path}")
    return contigs


def write_header_complete_vcf(source: str | Path, destination: str | Path, fai: str | Path) -> dict:
    """Insert authoritative reference contig declarations without changing records."""

    source, destination, fai = Path(source), Path(destination), Path(fai)
    contigs = _read_fai(fai)
    contig_names = {name for name, _ in contigs}
    record_count = 0
    used_contigs: set[str] = set()
    saw_chrom = False
    with source.open("r", encoding="utf-8", newline="") as reader, destination.open(
        "x", encoding="utf-8", newline=""
    ) as writer:
        for line_number, line in enumerate(reader, 1):
            if line.startswith("##contig=<"):
                continue
            if line.startswith("#CHROM"):
                if saw_chrom:
                    raise ValueError(f"{source}:{line_number}: duplicate #CHROM header")
                for name, length in contigs:
                    writer.write(f"##contig=<ID={name},length={length}>\n")
                writer.write(line if line.endswith("\n") else line + "\n")
                saw_chrom = True
                continue
            if line.startswith("#"):
                if saw_chrom:
                    raise ValueError(f"{source}:{line_number}: header after #CHROM")
                writer.write(line if line.endswith("\n") else line + "\n")
                continue
            if not saw_chrom:
                raise ValueError(f"{source}:{line_number}: record before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"{source}:{line_number}: malformed VCF record")
            if fields[0] not in contig_names:
                raise ValueError(f"{source}:{line_number}: contig absent from reference index")
            used_contigs.add(fields[0])
            record_count += 1
            writer.write(line if line.endswith("\n") else line + "\n")
    if not saw_chrom or record_count == 0:
        raise ValueError(f"{source}: missing #CHROM header or data records")
    return {
        "records": record_count,
        "reference_contigs": len(contigs),
        "used_contigs": sorted(used_contigs),
    }


def _iter_vcf(path: Path) -> tuple[list[str], Iterable[tuple[int, list[str]]]]:
    # This workflow deliberately emits plain VCF.  Refuse a misleading gzip
    # suffix rather than accidentally validating compressed bytes as text.
    if path.suffix == ".gz":
        raise ValueError(f"gzip VCF is not supported by this scoring runner: {path}")
    handle = path.open("r", encoding="utf-8", newline="")
    header: list[str] = []

    def records() -> Iterable[tuple[int, list[str]]]:
        try:
            for line_number, line in enumerate(handle, 1):
                if line.startswith("#"):
                    header.append(line.rstrip("\r\n"))
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 8:
                    raise ValueError(f"{path}:{line_number}: malformed VCF record")
                yield line_number, fields
        finally:
            handle.close()

    return header, records()


def _validated_score_values(info_text: str, info_key: str, alt: str, distance: int) -> int:
    matches = [
        item.split("=", 1)[1]
        for item in info_text.split(";")
        if item.startswith(f"{info_key}=")
    ]
    if not matches:
        return 0
    if len(matches) != 1:
        raise ValueError(f"duplicate INFO/{info_key} value")
    annotations = matches[0].split(",")
    for annotation in annotations:
        fields = annotation.split("|")
        if len(fields) != 10:
            raise ValueError(f"INFO/{info_key} annotation does not have ten fields")
        if fields[0] != alt or not fields[1]:
            raise ValueError(f"INFO/{info_key} allele/gene does not match the record")
        try:
            scores = [float(value) for value in fields[2:6]]
            positions = [int(value) for value in fields[6:10]]
        except ValueError as exc:
            raise ValueError(f"INFO/{info_key} has a nonnumeric DS/DP value") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in scores):
            raise ValueError(f"INFO/{info_key} has a score outside [0,1]")
        if any(abs(value) > distance for value in positions):
            raise ValueError(f"INFO/{info_key} has a DP outside [-{distance},{distance}]")
    return len(annotations)


def validate_scored_vcf(
    input_vcf: str | Path,
    output_vcf: str | Path,
    info_key: str,
    distance: int = 50,
    minimum_annotated_fraction: float = MIN_ANNOTATED_RECORD_FRACTION,
) -> dict:
    input_vcf, output_vcf = Path(input_vcf), Path(output_vcf)
    if (
        not math.isfinite(minimum_annotated_fraction)
        or minimum_annotated_fraction < 0.0
        or minimum_annotated_fraction > 1.0
    ):
        raise ValueError("minimum annotated fraction must be finite and in [0,1]")
    if not output_vcf.is_file() or output_vcf.stat().st_size == 0:
        raise ValueError(f"missing or empty scored VCF: {output_vcf}")
    with output_vcf.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise ValueError(f"scored VCF lacks a final newline: {output_vcf}")

    input_header, input_records_iter = _iter_vcf(input_vcf)
    output_header, output_records_iter = _iter_vcf(output_vcf)
    input_records = list(input_records_iter)
    output_records = list(output_records_iter)
    if not any(line.startswith("#CHROM\t") for line in input_header):
        raise ValueError(f"input VCF lacks #CHROM header: {input_vcf}")
    if not any(line.startswith("#CHROM\t") for line in output_header):
        raise ValueError(f"scored VCF lacks #CHROM header: {output_vcf}")
    declaration = f"##INFO=<ID={info_key},"
    if not any(line.startswith(declaration) for line in output_header):
        raise ValueError(f"scored VCF lacks INFO/{info_key} declaration: {output_vcf}")
    if len(input_records) != len(output_records):
        raise ValueError(
            f"record count mismatch: input={len(input_records)} output={len(output_records)}"
        )
    annotated = 0
    annotations = 0
    for (input_line, left), (output_line, right) in zip(input_records, output_records):
        if left[:7] != right[:7]:
            raise ValueError(
                f"variant identity/order mismatch at input line {input_line}, "
                f"output line {output_line}"
            )
        count = _validated_score_values(right[7], info_key, right[4], distance)
        annotated += count > 0
        annotations += count
    annotated_fraction = annotated / len(input_records) if input_records else 0.0
    if annotated_fraction < minimum_annotated_fraction:
        raise ValueError(
            f"INFO/{info_key} annotation coverage {annotated}/{len(input_records)} "
            f"({annotated_fraction:.3f}) is below required {minimum_annotated_fraction:.3f}"
        )
    return {
        "records": len(output_records),
        "annotated_records": annotated,
        "annotations": annotations,
        "annotated_fraction": annotated_fraction,
        "minimum_annotated_fraction": minimum_annotated_fraction,
        "info_key": info_key,
        "output_sha256": sha256_file(output_vcf),
    }


def _resource(path: Path) -> dict:
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "sha256": sha256_tree(path),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_immutable_json(path: Path, value: object) -> str:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o222 or path.read_bytes() != content:
            raise FileExistsError(f"refusing changed pre-existing immutable JSON: {path}")
        return digest
    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    temporary = path.with_name(f".{path.name}.tmp.{token}")
    if temporary.exists():
        raise FileExistsError(f"refusing stale immutable-JSON temporary: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _code_resources(repo_root: Path, official_package_dir: Path) -> list[Path]:
    return [
        repo_root / "openspliceai",
        # Hash the parent package as well as this workflow so PYTHONPATH cannot
        # select an unverified validation/__init__.py during child startup.
        repo_root / "validation",
        official_package_dir,
    ]


def build_bundle(
    plan_path: str | Path,
    run_dir: str | Path,
    repo_root: str | Path,
    *,
    python_bin: str | Path,
    spliceai_bin: str | Path,
    openspliceai_bin: str | Path,
    official_package_dir: str | Path,
    persist: bool,
) -> dict:
    repo_root = _path(repo_root, "repository root", directory=True)
    parsed = parse_plan(
        plan_path,
        python_bin=python_bin,
        spliceai_bin=spliceai_bin,
        openspliceai_bin=openspliceai_bin,
        official_package_dir=official_package_dir,
    )
    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_absolute():
        raise ValueError("run directory must be absolute")
    run_dir = run_dir.resolve(strict=False)
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    resources = {
        "manifest": _resource(Path(parsed["manifest"])),
        "reference": _resource(Path(parsed["reference"])),
        "reference_fai": _resource(Path(parsed["reference_fai"])),
        "annotation": _resource(Path(parsed["annotation"])),
        "harmonized_source": _resource(Path(parsed["harmonized"])),
        "executables": {
            key: _resource(Path(parsed[key]))
            for key in ("python_bin", "spliceai_bin", "openspliceai_bin")
        },
        "models": {
            task["name"]: _resource(Path(task["model"]))
            for task in parsed["tasks"]
            if task["model"] is not None
        },
        "official_models": [_resource(Path(path)) for path in parsed["official_models"]],
        "code": [
            _resource(path)
            for path in _code_resources(repo_root, Path(parsed["official_package_dir"]))
        ],
    }
    summary = {
        "predictors": [task["name"] for task in parsed["tasks"]],
        "source_input": parsed["source_input"],
        "reference": parsed["reference"],
        "annotation": parsed["annotation"],
        "score_mask": parsed["score_mask"],
        "run_dir": str(run_dir),
        "resources": resources,
        "dataset_completeness": parsed["dataset_completeness"],
        "coverage_policy": {
            "minimum_annotated_record_fraction": MIN_ANNOTATED_RECORD_FRACTION,
            "minimum_prediction_coverage": MIN_PREDICTION_COVERAGE,
        },
    }
    if not persist:
        return summary

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.preparing.", dir=run_dir.parent))
    try:
        (stage / "input").mkdir()
        (stage / "logs").mkdir()
        (stage / "status").mkdir()
        (stage / "receipts").mkdir()
        (stage / "scores").mkdir()
        frozen_plan = stage / "validation_plan.json"
        shutil.copyfile(parsed["plan_path"], frozen_plan)
        frozen_input = stage / "input" / "validation_variants.hg38.vcf"
        input_summary = write_header_complete_vcf(
            parsed["source_input"], frozen_input, parsed["reference_fai"]
        )
        frozen_harmonized = stage / "input" / "harmonized.tsv"
        shutil.copyfile(parsed["harmonized"], frozen_harmonized)
        tasks = []
        for task in parsed["tasks"]:
            frozen = dict(task)
            frozen["argv"] = _replace_flag(task["argv"], "-I", str(run_dir / "input" / frozen_input.name))
            tasks.append(frozen)
        tasks_path = stage / "tasks.json"
        _write_json(tasks_path, {"schema_version": SCHEMA_VERSION, "tasks": tasks})
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "kind": "external-validation-score-run",
            "run_dir": str(run_dir),
            "task_count": len(tasks),
            "score_mask": parsed["score_mask"],
            "repo_root": str(repo_root),
            "dataset_completeness": parsed["dataset_completeness"],
            "source_statuses": parsed["source_statuses"],
            "coverage_policy": summary["coverage_policy"],
            "plan_source": _resource(Path(parsed["plan_path"])),
            "frozen_plan_sha256": sha256_file(frozen_plan),
            "source_input": _resource(Path(parsed["source_input"])),
            "frozen_input_sha256": sha256_file(frozen_input),
            "frozen_input_summary": input_summary,
            "frozen_harmonized_sha256": sha256_file(frozen_harmonized),
            "tasks_sha256": sha256_file(tasks_path),
            "resources": resources,
            "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
        }
        _write_json(stage / "provenance.json", provenance)
        bundle_sha256 = sha256_file(stage / "provenance.json")
        for frozen in (frozen_plan, frozen_input, frozen_harmonized, tasks_path, stage / "provenance.json"):
            frozen.chmod(0o444)
        os.replace(stage, run_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        **summary,
        "frozen_input_summary": input_summary,
        "bundle_sha256": bundle_sha256,
    }


def _load_run(
    run_dir: str | Path, expected_bundle_sha256: str
) -> tuple[Path, dict, list[dict], str, str]:
    run_dir = _path(run_dir, "external scoring run", directory=True)
    provenance_raw = (run_dir / "provenance.json").read_bytes()
    bundle_sha256 = hashlib.sha256(provenance_raw).hexdigest()
    expected = str(expected_bundle_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected bundle SHA-256 must be exactly 64 hexadecimal characters")
    if bundle_sha256 != expected:
        raise ValueError(
            f"external bundle anchor mismatch: observed={bundle_sha256} expected={expected}"
        )
    provenance = json.loads(provenance_raw)
    tasks_raw = (run_dir / "tasks.json").read_bytes()
    tasks_sha256 = hashlib.sha256(tasks_raw).hexdigest()
    tasks_document = json.loads(tasks_raw)
    tasks = tasks_document.get("tasks")
    if (
        provenance.get("kind") != "external-validation-score-run"
        or provenance.get("schema_version") != SCHEMA_VERSION
        or tasks_document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(tasks, list)
    ):
        raise ValueError("invalid frozen external scoring run")
    if tasks_sha256 != provenance.get("tasks_sha256"):
        raise ValueError("frozen tasks do not match the externally anchored provenance")
    return run_dir, provenance, tasks, bundle_sha256, tasks_sha256


def _bundle_anchor(run_dir: Path, expected_bundle_sha256: str) -> str:
    expected = str(expected_bundle_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected bundle SHA-256 must be exactly 64 hexadecimal characters")
    observed = sha256_file(run_dir / "provenance.json")
    if observed != expected:
        raise ValueError(
            f"external bundle anchor mismatch: observed={observed} expected={expected}"
        )
    return observed


def _validate_frozen_task_semantics(
    run_dir: Path,
    provenance: Mapping,
    tasks: Sequence[Mapping],
    frozen_plan: Mapping,
) -> None:
    if provenance.get("run_dir") != str(run_dir):
        raise ValueError("frozen provenance run_dir does not match the executed directory")
    if provenance.get("deterministic_environment") != DETERMINISTIC_ENVIRONMENT:
        raise ValueError("frozen deterministic environment is not the required allowlist")
    expected_coverage = {
        "minimum_annotated_record_fraction": MIN_ANNOTATED_RECORD_FRACTION,
        "minimum_prediction_coverage": MIN_PREDICTION_COVERAGE,
    }
    if provenance.get("coverage_policy") != expected_coverage:
        raise ValueError("frozen coverage policy is not the required publication policy")
    completeness, source_statuses = _validate_dataset_completeness(frozen_plan)
    if completeness != provenance.get("dataset_completeness"):
        raise ValueError("frozen plan completeness differs from run provenance")
    if source_statuses != provenance.get("source_statuses"):
        raise ValueError("frozen plan source statuses differ from run provenance")

    if len(tasks) != len(EXPECTED_NAMES):
        raise ValueError("frozen tasks are not the expected seven-predictor array")
    resources = provenance.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("frozen run lacks resource provenance")
    executables = resources.get("executables")
    models = resources.get("models")
    if not isinstance(executables, Mapping) or set(executables) != {
        "python_bin",
        "spliceai_bin",
        "openspliceai_bin",
    }:
        raise ValueError("frozen executable provenance is incomplete")
    if not isinstance(models, Mapping) or set(models) != set(OSAI_NAMES):
        raise ValueError("frozen model provenance is incomplete")
    frozen_input = str((run_dir / "input" / "validation_variants.hg38.vcf").resolve())
    reference = str(resources["reference"]["path"])
    annotation = str(resources["annotation"]["path"])
    expected_keys = {
        "index",
        "name",
        "display_name",
        "kind",
        "info_key",
        "argv",
        "final_output",
        "model",
        "score_mask",
    }
    outputs: set[str] = set()
    for index, (name, raw_task) in enumerate(zip(EXPECTED_NAMES, tasks)):
        if not isinstance(raw_task, Mapping) or set(raw_task) != expected_keys:
            raise ValueError(f"frozen task {index} has unexpected fields")
        task = dict(raw_task)
        expected_kind = "spliceai" if name == OFFICIAL_NAME else "openspliceai"
        expected_display = "SpliceAI" if name == OFFICIAL_NAME else name
        if (
            task["index"] != index
            or task["name"] != name
            or task["kind"] != expected_kind
            or task["display_name"] != expected_display
            or task["info_key"] != EXPECTED_INFO[name]
            or task["score_mask"] != provenance.get("score_mask")
        ):
            raise ValueError(f"frozen task {index} identity/provenance is invalid")
        argv = task["argv"]
        if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
            raise ValueError(f"frozen task {index} argv is not a string list")
        if name == OFFICIAL_NAME:
            flags = ["-I", "-O", "-R", "-A", "-D", "-M"]
            if len(argv) != 1 + 2 * len(flags) or argv[1::2] != flags:
                raise ValueError("frozen official task does not have the strict command shape")
            if argv[0] != executables["spliceai_bin"]["path"]:
                raise ValueError("frozen official task executable is not pinned")
            if _flag(argv, "-A")[1].lower() != "grch38" or _flag(argv, "-D")[1] != "50":
                raise ValueError("frozen official task scientific flags are invalid")
            expected_model = None
        else:
            flags = [
                "-R",
                "-A",
                "--model",
                "-I",
                "-O",
                "--precision",
                "-D",
                "-f",
                "-M",
                "-b",
            ]
            if (
                len(argv) != 2 + 2 * len(flags)
                or argv[1:2] != ["variant"]
                or argv[2::2] != flags
            ):
                raise ValueError(f"frozen task {name} does not have the strict command shape")
            if argv[0] != executables["openspliceai_bin"]["path"]:
                raise ValueError(f"frozen task {name} executable is not pinned")
            fixed = {"--precision": "5", "-D": "50", "-f": "10000", "-b": "128"}
            if any(_flag(argv, flag)[1] != value for flag, value in fixed.items()):
                raise ValueError(f"frozen task {name} scientific flags are invalid")
            if _flag(argv, "-A")[1] != annotation:
                raise ValueError(f"frozen task {name} annotation is not pinned")
            expected_model = models[name]["path"]
            if _flag(argv, "--model")[1] != expected_model:
                raise ValueError(f"frozen task {name} model is not pinned")
        final = Path(str(task["final_output"]))
        if not final.is_absolute() or str(final.resolve(strict=False)) != str(final):
            raise ValueError(f"frozen task {name} output is not a normalized absolute path")
        if not final.parent.is_dir() or str(final) in outputs or str(final) == frozen_input:
            raise ValueError(f"frozen task {name} output is unsafe or duplicated")
        outputs.add(str(final))
        if (
            _flag(argv, "-I")[1] != frozen_input
            or _flag(argv, "-O")[1] != str(final)
            or _flag(argv, "-R")[1] != reference
            or _flag(argv, "-M")[1] != str(provenance.get("score_mask"))
            or task["model"] != expected_model
        ):
            raise ValueError(f"frozen task {name} arguments disagree with frozen provenance")


def _verified_run_context(
    run_dir: str | Path, expected_bundle_sha256: str
) -> tuple[Path, dict, list[dict], str]:
    run_dir, provenance, tasks, bundle_sha256, tasks_sha256 = _load_run(
        run_dir, expected_bundle_sha256
    )
    frozen_plan_raw = (run_dir / "validation_plan.json").read_bytes()
    frozen_plan = json.loads(frozen_plan_raw)
    checks = {
        "frozen_plan": hashlib.sha256(frozen_plan_raw).hexdigest(),
        "frozen_input": sha256_file(run_dir / "input" / "validation_variants.hg38.vcf"),
        "frozen_harmonized": sha256_file(run_dir / "input" / "harmonized.tsv"),
        "tasks": tasks_sha256,
    }
    expected = {
        "frozen_plan": provenance["frozen_plan_sha256"],
        "frozen_input": provenance["frozen_input_sha256"],
        "frozen_harmonized": provenance["frozen_harmonized_sha256"],
        "tasks": provenance["tasks_sha256"],
    }
    if checks != expected:
        raise ValueError(f"frozen run checksum mismatch: observed={checks} expected={expected}")
    _validate_frozen_task_semantics(run_dir, provenance, tasks, frozen_plan)
    if not (run_dir / "receipts").is_dir():
        raise ValueError("frozen run lacks its receipt directory")

    def verify_resource(resource: Mapping) -> None:
        path = Path(str(resource["path"]))
        observed = sha256_tree(path)
        if observed != resource["sha256"]:
            raise ValueError(f"resource checksum mismatch: {path}")

    resources = provenance["resources"]
    for key in ("manifest", "reference", "reference_fai", "annotation"):
        verify_resource(resources[key])
    for group in ("executables", "models"):
        for resource in resources[group].values():
            verify_resource(resource)
    for group in ("official_models", "code"):
        for resource in resources[group]:
            verify_resource(resource)
    return run_dir, provenance, tasks, bundle_sha256


def verify_run(run_dir: str | Path, expected_bundle_sha256: str) -> dict:
    run_dir, _provenance, tasks, bundle_sha256 = _verified_run_context(
        run_dir, expected_bundle_sha256
    )
    return {
        "verified": True,
        "task_count": len(tasks),
        "run_dir": str(run_dir),
        "bundle_sha256": bundle_sha256,
    }


def _status(run_dir: Path, index: int, value: Mapping) -> None:
    path = run_dir / "status" / f"task-{index:02d}.json"
    _write_json(path, dict(value))


def _clean_child_environment(run_dir: Path, provenance: Mapping, *, plotting: bool = False) -> dict:
    """Construct a small deterministic environment without inheriting code-injection variables."""

    python_path = Path(provenance["resources"]["executables"]["python_bin"]["path"])
    repo_root = Path(str(provenance["repo_root"]))
    environment = {
        **DETERMINISTIC_ENVIRONMENT,
        "PATH": f"{python_path.parent}:/usr/bin:/bin",
        "PYTHONPATH": str(repo_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "8")
    if not cpus.isdigit() or int(cpus) < 1 or int(cpus) > 1024:
        raise ValueError(f"invalid SLURM_CPUS_PER_TASK: {cpus!r}")
    environment["OMP_NUM_THREADS"] = cpus
    environment["MKL_NUM_THREADS"] = cpus
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
    ):
        value = os.environ.get(key)
        if value:
            if any(character in value for character in "\0\r\n"):
                raise ValueError(f"unsafe scheduler environment value for {key}")
            environment[key] = value
    if plotting:
        cache = run_dir / "matplotlib-cache"
        cache.mkdir(exist_ok=True)
        environment["MPLCONFIGDIR"] = str(cache)
    return environment


def _write_receipt(run_dir: Path, index: int, value: Mapping) -> None:
    path = run_dir / "receipts" / f"task-{index:02d}.json"
    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    temporary = path.with_name(f".{path.name}.tmp.{token}")
    if temporary.exists():
        raise FileExistsError(f"refusing stale receipt temporary: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _matching_receipt(
    run_dir: Path, index: int, task: Mapping, expected_bundle_sha256: str
) -> Optional[dict]:
    path = run_dir / "receipts" / f"task-{index:02d}.json"
    if not path.exists():
        return None
    # A receipt is an immutable commit marker.  Once the pathname exists, a
    # malformed, writable, symlinked, or otherwise mismatched value must stop
    # the worker rather than being interpreted as "not yet present".  Treating
    # it as absent can trigger a second scorer invocation and eventually fail
    # on receipt publication, obscuring the original integrity violation.
    if not path.is_file():
        raise ValueError(f"receipt path is not a regular file: {path}")
    if path.is_symlink() or path.stat().st_mode & 0o222:
        raise ValueError(f"receipt is not an immutable regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid receipt JSON: {path}") from exc
    required_keys = {
        "schema",
        "schema_version",
        "kind",
        "state",
        "task_index",
        "task",
        "final_output",
        "score_mask",
        "bundle_sha256",
        "tasks_sha256",
        "snapshot_path",
        "snapshot_sha256",
        "snapshot_records",
        "records",
        "annotated_records",
        "annotations",
        "annotated_fraction",
        "minimum_annotated_fraction",
        "info_key",
        "output_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required_keys:
        raise ValueError(f"receipt schema/key set mismatch: {path}")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": "external-validation-score-task",
        "state": "published",
        "task_index": index,
        "task": task["name"],
        "final_output": task["final_output"],
        "score_mask": task["score_mask"],
        "bundle_sha256": expected_bundle_sha256,
        "tasks_sha256": sha256_file(run_dir / "tasks.json"),
        "info_key": task["info_key"],
        "minimum_annotated_fraction": MIN_ANNOTATED_RECORD_FRACTION,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"receipt identity mismatch: {path}")
    digest = receipt.get("output_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or receipt.get("snapshot_sha256") != digest
        or receipt.get("snapshot_path") != str(_snapshot_path(run_dir, index, task, digest))
    ):
        raise ValueError(f"receipt digest or snapshot identity mismatch: {path}")
    snapshot = Path(receipt["snapshot_path"])
    if snapshot.is_symlink() or snapshot.parent.resolve() != (run_dir / "scores").resolve():
        raise ValueError(f"receipt snapshot path escapes run scores directory: {path}")
    if (
        not isinstance(receipt.get("records"), int)
        or receipt["records"] < 0
        or receipt.get("snapshot_records") != receipt["records"]
        or not isinstance(receipt.get("annotated_records"), int)
        or not isinstance(receipt.get("annotations"), int)
        or not isinstance(receipt.get("annotated_fraction"), (int, float))
    ):
        raise ValueError(f"receipt count or fraction fields are invalid: {path}")
    return receipt


def _snapshot_path(run_dir: Path, index: int, task: Mapping, digest: str) -> Path:
    return run_dir / "scores" / f"{index:02d}-{task['name']}-{digest}.vcf"


def _quarantine_orphan_snapshots(run_dir: Path, index: int, task: Mapping) -> list[str]:
    """Preserve but retire snapshots left before an atomic receipt publication."""

    pattern = f"{index:02d}-{task['name']}-*.vcf"
    candidates = sorted((run_dir / "scores").glob(pattern))
    if not candidates:
        return []
    destination_dir = run_dir / "status" / "orphan-snapshots"
    destination_dir.mkdir(exist_ok=True)
    moved = []
    for source in candidates:
        token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
        destination = destination_dir / f"{source.name}.unreceipted.{token}"
        suffix = 0
        while destination.exists():
            suffix += 1
            destination = destination_dir / f"{source.name}.unreceipted.{token}.{suffix}"
        os.rename(source, destination)
        moved.append(str(destination))
    return moved


def _publish_final_from_snapshot(
    frozen_input: Path,
    final: Path,
    snapshot: Path,
    task: Mapping,
    expected_validation: Mapping,
) -> dict:
    """Publish/recover the convenience output after snapshot+receipt commit."""

    if final.exists():
        observed = validate_scored_vcf(
            frozen_input,
            final,
            task["info_key"],
            minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
        )
        if observed != expected_validation:
            raise ValueError(f"pre-existing convenience output differs from receipt: {final}")
        return observed

    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    temporary = final.with_name(f".{final.name}.publishing.{token}")
    if temporary.exists():
        raise FileExistsError(f"refusing stale convenience-output temporary: {temporary}")
    try:
        with snapshot.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        copied = validate_scored_vcf(
            frozen_input,
            temporary,
            task["info_key"],
            minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
        )
        if copied != expected_validation:
            raise ValueError("convenience-output copy differs from receipt-bound snapshot")
        try:
            os.link(temporary, final)
        except FileExistsError:
            # A concurrent publisher is acceptable only when it published the
            # exact receipt-bound bytes; validate it below without overwriting.
            pass
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    observed = validate_scored_vcf(
        frozen_input,
        final,
        task["info_key"],
        minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
    )
    if observed != expected_validation:
        raise ValueError(f"convenience output differs from receipt-bound snapshot: {final}")
    return observed


def _publish_snapshot(
    run_dir: Path,
    index: int,
    task: Mapping,
    source: Path,
    expected_validation: Mapping,
) -> tuple[Path, dict]:
    digest = str(expected_validation["output_sha256"])
    destination = _snapshot_path(run_dir, index, task, digest)
    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    temporary = destination.with_name(f".{destination.name}.tmp.{token}")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing pre-existing score snapshot: {destination}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        snapshot_validation = validate_scored_vcf(
            run_dir / "input" / "validation_variants.hg38.vcf",
            temporary,
            task["info_key"],
            minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
        )
        if snapshot_validation != expected_validation:
            raise ValueError("run-local score snapshot differs from the validated scorer output")
        temporary.chmod(0o444)
        os.link(temporary, destination)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, snapshot_validation


def run_task(run_dir: str | Path, task_index: int, expected_bundle_sha256: str) -> dict:
    expected_bundle_sha256 = str(expected_bundle_sha256).lower()
    run_dir, provenance, tasks, _bundle_sha256 = _verified_run_context(
        run_dir, expected_bundle_sha256
    )
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task index outside [0,{len(tasks) - 1}]: {task_index}")
    task = tasks[task_index]
    frozen_input = run_dir / "input" / "validation_variants.hg38.vcf"
    final = Path(task["final_output"])
    receipt_path = run_dir / "receipts" / f"task-{task_index:02d}.json"
    receipt = _matching_receipt(run_dir, task_index, task, expected_bundle_sha256)
    if receipt_path.exists() and receipt is None:
        raise ValueError(f"invalid frozen-run receipt for predictor {task['name']}")
    if receipt is not None:
        snapshot = _snapshot_path(run_dir, task_index, task, receipt["output_sha256"])
        snapshot_validation = validate_scored_vcf(
            frozen_input,
            snapshot,
            task["info_key"],
            minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
        )
        if (
            str(snapshot) != receipt.get("snapshot_path")
            or snapshot_validation["output_sha256"] != receipt.get("snapshot_sha256")
            or snapshot_validation["records"] != receipt.get("snapshot_records")
            or snapshot.stat().st_mode & 0o222
        ):
            raise ValueError(f"score snapshot differs from its frozen-run receipt: {snapshot}")
        _publish_final_from_snapshot(
            frozen_input, final, snapshot, task, snapshot_validation
        )
        result = dict(receipt)
        result["state"] = "recovered_or_skipped_same_run_valid"
        _status(run_dir, task_index, result)
        return result
    if final.exists():
        raise FileExistsError(
            f"refusing pre-existing score without a receipt from this frozen run: {final}"
        )

    _quarantine_orphan_snapshots(run_dir, task_index, task)

    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{task_index}.{os.getpid()}"
    temporary = final.with_name(f".{final.name}.tmp.{token}")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary output: {temporary}")
    argv = _replace_flag(task["argv"], "-O", str(temporary))
    environment = _clean_child_environment(run_dir, provenance)
    process: Optional[subprocess.Popen] = None

    def terminate(signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)
        raise SystemExit(128 + signum)

    old_handlers = {
        signum: signal.signal(signum, terminate) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        process = subprocess.Popen(argv, env=environment, start_new_session=True)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, argv)
        validation = validate_scored_vcf(
            frozen_input,
            temporary,
            task["info_key"],
            minimum_annotated_fraction=MIN_ANNOTATED_RECORD_FRACTION,
        )
        snapshot, snapshot_validation = _publish_snapshot(
            run_dir, task_index, task, temporary, validation
        )
        result = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": "external-validation-score-task",
            "state": "published",
            "task_index": task_index,
            "task": task["name"],
            "final_output": str(final),
            "score_mask": task["score_mask"],
            "bundle_sha256": expected_bundle_sha256,
            "tasks_sha256": sha256_file(run_dir / "tasks.json"),
            "snapshot_path": str(snapshot),
            "snapshot_sha256": snapshot_validation["output_sha256"],
            "snapshot_records": snapshot_validation["records"],
            **validation,
        }
        _write_receipt(run_dir, task_index, result)
        _publish_final_from_snapshot(
            frozen_input, final, snapshot, task, snapshot_validation
        )
        temporary.unlink()
        _status(run_dir, task_index, result)
        return result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if temporary.exists():
            temporary.unlink()


def run_evaluation(
    run_dir: str | Path, bootstrap_replicates: int, expected_bundle_sha256: str
) -> int:
    expected_bundle_sha256 = str(expected_bundle_sha256).lower()
    run_dir, provenance, tasks, _bundle_sha256 = _verified_run_context(
        run_dir, expected_bundle_sha256
    )
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    frozen_input = run_dir / "input" / "validation_variants.hg38.vcf"
    argv = [
        provenance["resources"]["executables"]["python_bin"]["path"],
        "-m",
        "validation.full_snv_concordance",
        "evaluate-external",
        "--harmonized",
        str(run_dir / "input" / "harmonized.tsv"),
        "--harmonized-sha256",
        str(provenance["frozen_harmonized_sha256"]),
        "--validation-plan",
        str(run_dir / "validation_plan.json"),
        "--validation-plan-sha256",
        str(provenance["frozen_plan_sha256"]),
        "--output-dir",
        str(run_dir / "evaluation"),
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--score-mask",
        str(provenance["score_mask"]),
        "--minimum-prediction-coverage",
        str(provenance["coverage_policy"]["minimum_prediction_coverage"]),
    ]
    receipt_entries = []
    for index, task in enumerate(tasks):
        receipt = _matching_receipt(run_dir, index, task, expected_bundle_sha256)
        if receipt is None:
            raise ValueError(f"missing frozen-run receipt for predictor {task['name']}")
        snapshot = _snapshot_path(run_dir, index, task, str(receipt.get("output_sha256")))
        if str(snapshot) != receipt.get("snapshot_path"):
            raise ValueError(f"receipt snapshot path is invalid for predictor {task['name']}")
        validation = validate_scored_vcf(
            frozen_input,
            snapshot,
            task["info_key"],
            minimum_annotated_fraction=provenance["coverage_policy"][
                "minimum_annotated_record_fraction"
            ],
        )
        if (
            validation["output_sha256"] != receipt.get("snapshot_sha256")
            or validation["records"] != receipt.get("snapshot_records")
            or any(receipt.get(key) != value for key, value in validation.items())
            or snapshot.is_symlink()
            or snapshot.stat().st_mode & 0o222
        ):
            raise ValueError(f"score snapshot differs from receipt for predictor {task['name']}")
        receipt_path = run_dir / "receipts" / f"task-{index:02d}.json"
        receipt_entries.append(
            {
                "task_index": index,
                "task": task["name"],
                "display_name": task["display_name"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "snapshot_path": str(snapshot),
                "snapshot_sha256": validation["output_sha256"],
                "snapshot_records": validation["records"],
                "info_key": task["info_key"],
            }
        )
        argv.extend(("--score", f"{task['display_name']}={snapshot}"))
        argv.extend(("--score-info", f"{task['display_name']}={task['info_key']}"))
        argv.extend(
            ("--score-sha256", f"{task['display_name']}={validation['output_sha256']}")
        )
        argv.extend(("--score-records", f"{task['display_name']}={validation['records']}"))
    submission_path, submission, submission_sha256 = _load_committed_submission(
        run_dir, expected_bundle_sha256
    )
    if int(submission["bootstrap_replicates"]) != bootstrap_replicates:
        raise ValueError("evaluation bootstrap count differs from committed submission")
    receipt_manifest_bytes = json.dumps(
        receipt_entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run_identity = {
        "schema_version": 1,
        "kind": "external-validation-evaluation-inputs",
        "run_dir": str(run_dir),
        "bundle_sha256": expected_bundle_sha256,
        "tasks_sha256": sha256_file(run_dir / "tasks.json"),
        "frozen_plan_sha256": provenance["frozen_plan_sha256"],
        "frozen_harmonized_sha256": provenance["frozen_harmonized_sha256"],
        "score_mask": provenance["score_mask"],
        "coverage_policy": provenance["coverage_policy"],
        "submission": {
            "path": str(submission_path),
            "sha256": submission_sha256,
            "document": submission,
        },
        "receipt_manifest_sha256": hashlib.sha256(receipt_manifest_bytes).hexdigest(),
        "receipts": receipt_entries,
    }
    run_identity_path = run_dir / "evaluation_inputs.json"
    run_identity_sha256 = _write_immutable_json(run_identity_path, run_identity)
    argv.extend(("--run-provenance", str(run_identity_path)))
    argv.extend(("--run-provenance-sha256", run_identity_sha256))
    environment = _clean_child_environment(run_dir, provenance, plotting=True)
    return subprocess.run(argv, env=environment, check=False).returncode


def _submission_job_id(value: object, label: str, *, optional: bool = False) -> Optional[str]:
    if optional and value is None:
        return None
    candidate = str(value)
    if not candidate.isdigit() or int(candidate) < 1:
        raise ValueError(f"{label} must be a positive numeric Slurm job ID")
    return candidate


def _load_committed_submission(
    run_dir: Path, expected_bundle_sha256: str
) -> tuple[Path, dict, str]:
    path = run_dir / "submission.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("external scoring run lacks a regular committed submission record")
    raw = path.read_bytes()
    document = json.loads(raw)
    required = {
        "schema_version",
        "kind",
        "state",
        "score_array_job",
        "evaluation_job",
        "bootstrap_replicates",
        "bundle_sha256",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("external scoring submission record has an invalid schema")
    if (
        document["schema_version"] != 1
        or document["kind"] != "external-validation-slurm-submission"
        or document["state"] != "committed"
        or document["bundle_sha256"] != expected_bundle_sha256
        or not isinstance(document["bootstrap_replicates"], int)
        or document["bootstrap_replicates"] < 0
    ):
        raise ValueError("external scoring submission is not a committed anchored run")
    _submission_job_id(document["score_array_job"], "score array job")
    _submission_job_id(document["evaluation_job"], "evaluation job", optional=True)
    return path, document, hashlib.sha256(raw).hexdigest()


def record_submission(
    run_dir: str | Path,
    score_job: str,
    evaluation_job: str,
    replicates: int,
    expected_bundle_sha256: str,
) -> None:
    run_dir = _path(run_dir, "external scoring run", directory=True)
    _bundle_anchor(run_dir, expected_bundle_sha256)
    score_job = _submission_job_id(score_job, "score array job") or ""
    evaluation_job = _submission_job_id(
        evaluation_job or None, "evaluation job", optional=True
    )
    if replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    submission_path = run_dir / "submission.json"
    if submission_path.exists():
        raise FileExistsError(f"refusing pre-existing submission record: {submission_path}")
    _write_json(
        submission_path,
        {
            "schema_version": 1,
            "kind": "external-validation-slurm-submission",
            "state": "pending_release",
            "score_array_job": score_job,
            "evaluation_job": evaluation_job,
            "bootstrap_replicates": replicates,
            "bundle_sha256": expected_bundle_sha256,
        },
    )


def commit_submission(run_dir: str | Path, expected_bundle_sha256: str) -> None:
    run_dir = _path(run_dir, "external scoring run", directory=True)
    _bundle_anchor(run_dir, expected_bundle_sha256)
    path = run_dir / "submission.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("external scoring run lacks its pending submission record")
    document = json.loads(path.read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("bundle_sha256") != expected_bundle_sha256
        or document.get("state") not in {"pending_release", "committed"}
    ):
        raise ValueError("cannot commit an invalid external submission record")
    if document["state"] == "pending_release":
        document["state"] = "committed"
        _write_json(path, document)
    _load_committed_submission(run_dir, expected_bundle_sha256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", required=True)
    common.add_argument("--run-dir", required=True)
    common.add_argument("--repo-root", required=True)
    common.add_argument("--python-bin", required=True)
    common.add_argument("--spliceai-bin", required=True)
    common.add_argument("--openspliceai-bin", required=True)
    common.add_argument("--official-package-dir", required=True)
    subparsers.add_parser("inspect", parents=[common])
    subparsers.add_parser("prepare", parents=[common])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--expected-bundle-sha256", required=True)
    task = subparsers.add_parser("run-task")
    task.add_argument("--run-dir", required=True)
    task.add_argument("--task-index", required=True, type=int)
    task.add_argument("--expected-bundle-sha256", required=True)
    evaluation = subparsers.add_parser("run-evaluation")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--bootstrap-replicates", required=True, type=int)
    evaluation.add_argument("--expected-bundle-sha256", required=True)
    record = subparsers.add_parser("record-submission")
    record.add_argument("--run-dir", required=True)
    record.add_argument("--score-job", required=True)
    record.add_argument("--evaluation-job", default="")
    record.add_argument("--bootstrap-replicates", required=True, type=int)
    record.add_argument("--expected-bundle-sha256", required=True)
    commit = subparsers.add_parser("commit-submission")
    commit.add_argument("--run-dir", required=True)
    commit.add_argument("--expected-bundle-sha256", required=True)
    validate = subparsers.add_parser("validate-vcf")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--info-key", required=True, choices=("SpliceAI", "OpenSpliceAI"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"inspect", "prepare"}:
        result = build_bundle(
            args.plan,
            args.run_dir,
            args.repo_root,
            python_bin=args.python_bin,
            spliceai_bin=args.spliceai_bin,
            openspliceai_bin=args.openspliceai_bin,
            official_package_dir=args.official_package_dir,
            persist=args.command == "prepare",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "verify":
        print(
            json.dumps(
                verify_run(args.run_dir, args.expected_bundle_sha256), sort_keys=True
            )
        )
    elif args.command == "run-task":
        print(
            json.dumps(
                run_task(args.run_dir, args.task_index, args.expected_bundle_sha256),
                sort_keys=True,
            )
        )
    elif args.command == "run-evaluation":
        return run_evaluation(
            args.run_dir, args.bootstrap_replicates, args.expected_bundle_sha256
        )
    elif args.command == "record-submission":
        record_submission(
            args.run_dir,
            args.score_job,
            args.evaluation_job,
            args.bootstrap_replicates,
            args.expected_bundle_sha256,
        )
    elif args.command == "commit-submission":
        commit_submission(args.run_dir, args.expected_bundle_sha256)
    elif args.command == "validate-vcf":
        print(json.dumps(validate_scored_vcf(args.input, args.output, args.info_key), sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
