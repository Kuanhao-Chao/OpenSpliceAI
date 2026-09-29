"""Read-only progress snapshots; metadata checks do not re-audit VCF contents."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ManifestError(ValueError):
    """Raised when an audit manifest fails its identity contract."""


def category(state):
    return "valid" if state == "valid" else "missing" if state in ("missing", "missing_input") else "invalid"


def check_row(row):
    errors, changed, missing = [], False, False
    chunk = int(row["chunk_id"])
    for role in ("input", "output"):
        path = Path(row[role + "_path"])
        expected = (int(row[role + "_size"]), int(row[role + "_mtime_ns"]))
        try:
            stat = path.stat()
            changed |= (stat.st_size, stat.st_mtime_ns) != expected
        except FileNotFoundError:
            missing = True
            if expected != (-1, -1):
                errors.append(dict(chunk_id=chunk, kind="missing", path_role=role,
                                   message=f"file not found: {path}"))
        except OSError as exc:
            errors.append(dict(chunk_id=chunk, kind="unavailable", path_role=role, message=str(exc)))
            changed = True
    state = "unverified" if changed else "missing" if missing else category(row["state"])
    return chunk, state, changed, errors


def collect_seed(campaign, seed, workers=16, progress_every=5000, check_filesystem=True):
    manifest = Path(campaign) / "manifests" / (seed + ".tsv")
    metadata = json.loads(manifest.with_suffix(".tsv.meta.json").read_text())
    raw = manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != metadata["manifest_sha256"]:
        raise ManifestError("manifest SHA-256 mismatch")
    rows = list(csv.DictReader(raw.decode().splitlines(), delimiter="\t"))
    ids = [int(row["chunk_id"]) for row in rows]
    duplicates = [chunk for chunk, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ManifestError(f"duplicate chunk_id {duplicates[0]}")
    if set(ids) != set(range(1, metadata["chunk_count"] + 1)):
        raise ManifestError("chunk IDs are not complete")
    states = dict(Counter(row["state"] for row in rows))
    provenance = dict(Counter(row["provenance_state"] for row in rows))
    if metadata["seed"] != seed or states != metadata["state_counts"] or provenance != metadata["provenance_state_counts"]:
        raise ManifestError("manifest and audit metadata disagree")
    audit = {state: sum(n for s, n in states.items() if category(s) == state)
             for state in ("valid", "missing", "invalid")}
    result = dict(total_chunks=len(rows), audit_at_utc=metadata["created_at"],
                  manifest_sha256=digest, audit_counts=audit, audit_state_counts=states,
                  provenance_state_counts=provenance, current_counts=None,
                  filesystem=dict(status="unchecked", checked_at_utc=None, changed_chunks=None, errors=None))
    if check_filesystem:
        counts = dict.fromkeys(("valid", "missing", "invalid", "unverified"), 0)
        changed, errors = [], []
        print(f"{seed}: checking {len(rows):,} chunk input/output metadata", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (chunk, state, change, row_errors) in enumerate(pool.map(check_row, rows), 1):
                counts[state] += 1
                if change:
                    changed.append(chunk)
                errors.extend(row_errors)
                if progress_every and i % progress_every == 0:
                    print(f"{seed}: {i:,}/{len(rows):,} checked", flush=True)
        result["current_counts"] = counts
        result["filesystem"] = dict(status="checked", checked_at_utc=now(),
                                    changed_chunks=sorted(changed), errors=errors)
    return result


def query_scheduler(run=subprocess.run, timeout=30):
    scheduler = dict(checked_at_utc=now(), status="unavailable", error=None, jobs=[])
    try:
        completed = run(["/cm/shared/apps/slurm/current/bin/squeue", "-h", "-u", str(os.getuid()),
                         "-o", "%i|%j|%T|%r|%U"], capture_output=True, text=True, timeout=timeout, check=True)
        raw = completed.stdout
        for line in raw.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 5:
                raise ValueError("unexpected squeue row")
            job_id, name, state, reason, uid = parts
            if uid == str(os.getuid()) and re.match(r"^osai_rs(?:10|13)($|_)", name):
                scheduler["jobs"].append(dict(job_id=job_id, name=name, state=state, reason=reason))
        scheduler["status"] = "available"
        return scheduler, raw
    except subprocess.TimeoutExpired:
        scheduler["error"] = f"squeue timed out after {timeout} seconds"
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        scheduler["error"] = str(exc)
    scheduler["jobs"] = []
    return scheduler, None


def main(argv=None, scheduler_run=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-raw-output", type=Path)
    parser.add_argument("--skip-filesystem-check", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    seeds = {s: collect_seed(args.campaign_root, s, workers=args.workers,
                              check_filesystem=not args.skip_filesystem_check) for s in ("rs10", "rs13")}
    scheduler, raw = query_scheduler(run=scheduler_run)
    jobs = scheduler.pop("jobs")
    for seed, data in seeds.items():
        data["jobs"] = [j for j in jobs if re.match(r"^osai_" + seed + r"($|_)", j["name"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema_version=1, observed_at_utc=now(),
                                         scheduler=scheduler, seeds=seeds), indent=2) + "\n")
    if args.scheduler_raw_output and raw is not None:
        args.scheduler_raw_output.write_text(raw.rstrip("\n") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
