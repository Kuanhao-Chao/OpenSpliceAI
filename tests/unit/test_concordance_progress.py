import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from validation.concordance_study import progress


FIELDS = (
    "chunk_id",
    "state",
    "input_path",
    "output_path",
    "input_size",
    "output_size",
    "input_mtime_ns",
    "output_mtime_ns",
    "provenance_state",
)


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _row(chunk_id: int, state: str, source: Path, output: Path, provenance: str):
    source_identity = _identity(source) if source.exists() else (-1, -1)
    output_identity = _identity(output) if output.exists() else (-1, -1)
    return {
        "chunk_id": str(chunk_id),
        "state": state,
        "input_path": str(source),
        "output_path": str(output),
        "input_size": str(source_identity[0]),
        "output_size": str(output_identity[0]),
        "input_mtime_ns": str(source_identity[1]),
        "output_mtime_ns": str(output_identity[1]),
        "provenance_state": provenance,
    }


def _write_seed(campaign: Path, seed: str, rows: list[dict]) -> tuple[Path, Path]:
    manifests = campaign / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / f"{seed}.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    state_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for row in rows:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
        provenance = row["provenance_state"]
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
    meta = manifests / f"{seed}.tsv.meta.json"
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "created_at": "2026-08-08T17:15:21.467034+00:00",
                "chunk_count": len(rows),
                "manifest_sha256": digest,
                "state_counts": state_counts,
                "provenance_state_counts": provenance_counts,
            }
        ),
        encoding="utf-8",
    )
    return manifest, meta


def _file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_collect_seed_reports_independent_audit_and_current_counts(tmp_path):
    source1 = _file(tmp_path / "files/source1.vcf", "source-one")
    output1 = _file(tmp_path / "files/output1.vcf", "output-one")
    source2 = _file(tmp_path / "files/source2.vcf", "source-two")
    missing2 = tmp_path / "files/output2.vcf"
    source3 = _file(tmp_path / "files/source3.vcf", "source-three")
    output3 = _file(tmp_path / "files/output3.vcf", "truncated")
    rows = [
        _row(1, "valid", source1, output1, "legacy_unprovenanced"),
        _row(2, "missing", source2, missing2, "not_applicable"),
        _row(3, "truncated", source3, output3, "not_applicable"),
    ]
    _write_seed(tmp_path, "rs10", rows)

    result = progress.collect_seed(tmp_path, "rs10", workers=2, progress_every=1)

    assert result["total_chunks"] == 3
    assert result["audit_at_utc"] == "2026-08-08T17:15:21.467034+00:00"
    assert result["audit_counts"] == {"valid": 1, "missing": 1, "invalid": 1}
    assert result["audit_state_counts"] == {"missing": 1, "truncated": 1, "valid": 1}
    assert result["provenance_state_counts"] == {
        "legacy_unprovenanced": 1,
        "not_applicable": 2,
    }
    assert result["current_counts"] == {
        "valid": 1,
        "missing": 1,
        "invalid": 1,
        "unverified": 0,
    }
    assert result["filesystem"]["status"] == "checked"
    assert result["filesystem"]["changed_chunks"] == []
    assert result["filesystem"]["errors"] == []


def test_changed_newly_missing_and_newly_created_files_are_not_valid(tmp_path):
    pairs = []
    for chunk in range(1, 4):
        source = _file(tmp_path / f"files/source{chunk}.vcf", f"source-{chunk}")
        output = _file(tmp_path / f"files/output{chunk}.vcf", f"output-{chunk}")
        pairs.append((source, output))
    absent_output = tmp_path / "files/output4.vcf"
    source4 = _file(tmp_path / "files/source4.vcf", "source-4")
    rows = [
        _row(1, "valid", *pairs[0], "legacy_unprovenanced"),
        _row(2, "valid", *pairs[1], "legacy_unprovenanced"),
        _row(3, "valid", *pairs[2], "legacy_unprovenanced"),
        _row(4, "missing", source4, absent_output, "not_applicable"),
    ]
    _write_seed(tmp_path, "rs10", rows)
    pairs[0][1].write_text("output-1 changed", encoding="utf-8")
    pairs[1][1].unlink()
    absent_output.write_text("new output", encoding="utf-8")

    result = progress.collect_seed(tmp_path, "rs10", workers=2)

    assert result["current_counts"] == {
        "valid": 1,
        "missing": 1,
        "invalid": 0,
        "unverified": 2,
    }
    assert result["filesystem"]["changed_chunks"] == [1, 4]
    assert result["filesystem"]["errors"] == [
        {
            "chunk_id": 2,
            "kind": "missing",
            "path_role": "output",
            "message": f"file not found: {pairs[1][1]}",
        }
    ]


def test_skip_filesystem_check_marks_counts_unchecked(tmp_path):
    source = _file(tmp_path / "source.vcf", "source")
    output = _file(tmp_path / "output.vcf", "output")
    _write_seed(
        tmp_path,
        "rs10",
        [_row(1, "valid", source, output, "legacy_unprovenanced")],
    )

    result = progress.collect_seed(tmp_path, "rs10", check_filesystem=False)

    assert result["current_counts"] is None
    assert result["filesystem"] == {
        "status": "unchecked",
        "checked_at_utc": None,
        "changed_chunks": None,
        "errors": None,
    }


@pytest.mark.parametrize(
    "chunk_ids, message",
    [([1, 1], "duplicate chunk_id 1"), ([1, 3], "chunk IDs are not complete")],
)
def test_collect_seed_rejects_non_unique_or_incomplete_chunk_ids(
    tmp_path, chunk_ids, message
):
    rows = []
    for index, chunk_id in enumerate(chunk_ids):
        source = _file(tmp_path / f"source{index}.vcf", "source")
        output = _file(tmp_path / f"output{index}.vcf", "output")
        rows.append(_row(chunk_id, "valid", source, output, "legacy_unprovenanced"))
    _write_seed(tmp_path, "rs10", rows)

    with pytest.raises(progress.ManifestError, match=message):
        progress.collect_seed(tmp_path, "rs10", check_filesystem=False)


def test_collect_seed_rejects_manifest_checksum_mismatch(tmp_path):
    source = _file(tmp_path / "source.vcf", "source")
    output = _file(tmp_path / "output.vcf", "output")
    manifest, _ = _write_seed(
        tmp_path,
        "rs10",
        [_row(1, "valid", source, output, "legacy_unprovenanced")],
    )
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(progress.ManifestError, match="manifest SHA-256 mismatch"):
        progress.collect_seed(tmp_path, "rs10", check_filesystem=False)


def test_query_scheduler_reports_unavailable_instead_of_no_jobs():
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=kwargs["timeout"])

    scheduler, raw = progress.query_scheduler(run=unavailable, timeout=3)

    assert scheduler["status"] == "unavailable"
    assert scheduler["error"] == "squeue timed out after 3 seconds"
    assert scheduler["jobs"] == []
    assert raw is None


def test_query_scheduler_filters_uid_and_exact_seed_job_prefixes():
    raw = "\n".join(
        [
            "29611225|osai_rs10|RUNNING|None|1239",
            "29611226_7|osai_rs13_retry|PENDING|Resources|1239",
            "42|osai_rs100|RUNNING|None|1239",
            "43|osai_rs10|RUNNING|None|9999",
            "44|other|RUNNING|None|1239",
        ]
    )

    def available(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=raw, stderr="")

    scheduler, captured = progress.query_scheduler(run=available)

    assert scheduler["status"] == "available"
    assert scheduler["error"] is None
    assert scheduler["jobs"] == [
        {"job_id": "29611225", "name": "osai_rs10", "state": "RUNNING", "reason": "None"},
        {
            "job_id": "29611226_7",
            "name": "osai_rs13_retry",
            "state": "PENDING",
            "reason": "Resources",
        },
    ]
    assert captured == raw


def test_cli_writes_schema_and_groups_scheduler_jobs_by_seed(tmp_path):
    for seed in ("rs10", "rs13"):
        source = _file(tmp_path / f"{seed}-source.vcf", "source")
        output = _file(tmp_path / f"{seed}-output.vcf", "output")
        _write_seed(
            tmp_path,
            seed,
            [_row(1, "valid", source, output, "legacy_unprovenanced")],
        )
    raw = "\n".join(
        [
            "29611225|osai_rs10_retry|RUNNING|None|1239",
            "29611226|osai_rs13|PENDING|Priority|1239",
        ]
    )

    def available(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=raw, stderr="")

    output_path = tmp_path / "snapshot.json"
    raw_path = tmp_path / "scheduler.txt"
    exit_code = progress.main(
        [
            "--campaign-root",
            str(tmp_path),
            "--output",
            str(output_path),
            "--scheduler-raw-output",
            str(raw_path),
            "--skip-filesystem-check",
        ],
        scheduler_run=available,
    )

    assert exit_code == 0
    snapshot = json.loads(output_path.read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == 1
    assert set(snapshot) == {"schema_version", "observed_at_utc", "scheduler", "seeds"}
    assert snapshot["scheduler"]["status"] == "available"
    assert set(snapshot["scheduler"]) == {"checked_at_utc", "status", "error"}
    assert snapshot["seeds"]["rs10"]["jobs"] == [
        {
            "job_id": "29611225",
            "name": "osai_rs10_retry",
            "state": "RUNNING",
            "reason": "None",
        }
    ]
    assert snapshot["seeds"]["rs13"]["jobs"] == [
        {
            "job_id": "29611226",
            "name": "osai_rs13",
            "state": "PENDING",
            "reason": "Priority",
        }
    ]
    assert snapshot["seeds"]["rs10"]["filesystem"]["status"] == "unchecked"
    assert raw_path.read_text(encoding="utf-8") == raw + "\n"
