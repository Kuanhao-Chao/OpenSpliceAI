"""Freeze a self-contained CPU run, then submit its map/reduce dependency.

Dry-run by default. A submitted run uses a copied code tree, annotation, and
pairs file: later report edits cannot invalidate queued computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("primary", "matched"), required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--sites-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--chunks-per-task", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=120)
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--reduce-mem", default="96G")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    if min(a.chunks_per_task, a.concurrency, a.hours) < 1:
        p.error("task size, concurrency and hours must be positive")
    pairs, sites, out = (Path(v).resolve() for v in (a.pairs_file, a.sites_file, a.output_dir))
    with pairs.open() as handle:
        count = sum(1 for _ in handle)-1
    if count < 1:
        p.error("empty pairs file")
    tasks = math.ceil(count/a.chunks_per_task)
    plan = {"kind": a.kind, "source_pairs": str(pairs), "source_sites": str(sites),
            "pairs_sha256": sha(pairs), "sites_sha256": sha(sites),
            "output": str(out), "chunks": count, "tasks": tasks,
            "chunks_per_task": a.chunks_per_task, "concurrency": a.concurrency,
            "hours": a.hours, "python": str(Path(a.python).resolve()), "status": "dry-run"}
    plan["reduce_mem"] = a.reduce_mem
    print(json.dumps(plan, indent=2), flush=True)
    if not a.submit:
        return 0
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    code = out / "code"
    for relative in ("validation/full_snv_concordance", "validation/concordance_study"):
        shutil.copytree(repo/relative, code/relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(repo/"validation/__init__.py", code/"validation/__init__.py")
    shutil.copy2(pairs, out/"pairs.tsv")
    shutil.copy2(sites, out/"sites.tsv")
    for name in ("maps", "logs"):
        (out/name).mkdir()
    # The manifest is itself pinned in the sbatch arguments, preventing an edit
    # to both a frozen file and its expected checksum from passing verification.
    common = ["--pairs-file", str(out/"pairs.tsv"), "--sites-file", str(out/"sites.tsv")]
    q = shlex.quote
    worker = out/"worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {q(str(code))}\n"
        f"test \"$(sha256sum {q(str(out/'checks.sha256'))} | cut -d' ' -f1)\" = \"$2\"\n"
        f"sha256sum --check --status {q(str(out/'checks.sha256'))}\n"
        "if [ \"$1\" = map ]; then\n"
        f"  exec >{q(str(out/'logs'))}/map-${{SLURM_ARRAY_TASK_ID}}.out 2>&1\n"
        f"  args=(map-{a.kind} --task-index \"$SLURM_ARRAY_TASK_ID\" --chunks-per-task {a.chunks_per_task}"
        f" --output {q(str(out/'maps'))}/map-${{SLURM_ARRAY_TASK_ID}}.json)\n"
        "else\n"
        f"  exec >{q(str(out/'logs/reduce.out'))} 2>&1\n"
        f"  args=(reduce-{a.kind} --input-dir {q(str(out/'maps'))} --expected-task-count {tasks}"
        f" --output {q(str(out/'summary.json'))})\nfi\n"
        f"exec env -i PATH={q(str(Path(a.python).resolve().parent)+':/usr/bin:/bin')}"
        f" PYTHONPATH={q(str(code))} PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1"
        " OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/osai-depth-mpl"
        f" {q(plan['python'])} -m validation.concordance_study.depth \"${{args[@]}}\""
        f" {shlex.join(common)}\n"
    )
    frozen = sorted(p for p in code.rglob("*") if p.is_file()) + [out/"pairs.tsv", out/"sites.tsv", worker]
    manifest = out/"checks.sha256"
    manifest.write_text("".join(f"{sha(path)}  {path}\n" for path in frozen))
    manifest_sha = sha(manifest)
    for path in frozen+[manifest]:
        path.chmod(0o444)
    plan.update(status="prepared", checks_sha256=manifest_sha)
    (out/"launch.json").write_text(json.dumps(plan, indent=2))
    jobs = []
    sbatch = "/cm/shared/apps/slurm/current/bin/sbatch"
    try:
        base = [sbatch, "--parsable", "--account=ssalzbe1-chess", "--partition=parallel", "--export=NONE",
                f"--chdir={code}", f"--time={a.hours}:00:00", f"--output={out}/logs/slurm-%j.out"]
        def submit(args):
            value = subprocess.check_output(base+args, text=True).strip().split(";")[0]
            if not value.isdigit():
                raise ValueError(f"invalid Slurm job id: {value!r}")
            jobs.append(value)
            return value
        map_id = submit(["--hold", f"--array=0-{tasks-1}%{a.concurrency}", "--cpus-per-task=1",
                         "--mem=8G" if a.kind == "matched" else "--mem=4G",
                         f"--job-name=depth_{a.kind}", str(worker), "map", manifest_sha])
        reduce_id = submit([f"--dependency=afterok:{map_id}", "--cpus-per-task=4", f"--mem={a.reduce_mem}",
                            f"--job-name=depth_reduce_{a.kind}", str(worker), "reduce", manifest_sha])
        plan.update(status="submitted", map_job=map_id, reduce_job=reduce_id)
        (out/"launch.json").write_text(json.dumps(plan, indent=2))
        subprocess.run(["/cm/shared/apps/slurm/current/bin/scontrol", "release", map_id], check=True)
    except BaseException as exc:
        cancelled = subprocess.run(["/cm/shared/apps/slurm/current/bin/scancel", *jobs], capture_output=True, text=True) if jobs else None
        plan.update(status="submission_failed", jobs=jobs, error=str(exc),
                    rollback_returncode=cancelled.returncode if cancelled else None,
                    rollback_stderr=cancelled.stderr if cancelled else "")
        (out/"launch.json").write_text(json.dumps(plan, indent=2))
        raise
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
