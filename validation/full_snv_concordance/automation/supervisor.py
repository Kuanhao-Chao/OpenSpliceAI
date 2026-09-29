"""Resumable scoring/analysis supervisor.

The scheduler-facing policy lives in this small, testable module.  Every
transition is checkpointed and every external command is logged; completion is
never inferred from filenames alone.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


UTC = dt.timezone.utc
STATES = (
    "WAIT_SCHEDULER", "CLEAN_LEGACY", "AUDIT", "PLAN", "SUBMIT_SCORE",
    "MONITOR_SCORE", "RECONCILE", "BUILD_PAIRS", "SUBMIT_ANALYSIS",
    "MONITOR_ANALYSIS", "PUBLISH_SUMMARY", "COMPLETE", "PAUSED_ERROR",
)
TERMINAL_JOB_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED",
}
SUCCESS_JOB_STATES = {"COMPLETED"}
JOB_ID_RE = re.compile(r"^(?P<id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
SUBMITTED_SCORE_RE = re.compile(r"\[(rs10|rs13)\]\s+submitted job\s+([0-9]+)")
SUBMITTED_ANALYSIS_RE = re.compile(
    r"Submitted\s+map=(?P<map>[0-9]+)\s+reduce=(?P<reduce>[0-9]+)\s+report=(?P<report>[0-9]+)"
)


def utc_now() -> str:
    return dt.datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_job_id(raw: str) -> str:
    match = JOB_ID_RE.fullmatch(raw.strip())
    if not match:
        raise SchedulerError(f"invalid parsable Slurm job ID: {raw!r}")
    return match.group("id")


class SchedulerError(RuntimeError):
    """A scheduler operation failed or returned an unverifiable result."""


class CampaignGuardError(RuntimeError):
    """A safety invariant failed; this is not a transient scheduler outage."""


@dataclass
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str], Optional[Path]], CommandResult]


def subprocess_runner(argv: Sequence[str], cwd: Optional[Path] = None) -> CommandResult:
    completed = subprocess.run(
        list(argv), cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return CommandResult(tuple(str(item) for item in argv), completed.returncode,
                         completed.stdout, completed.stderr)


@dataclass
class AutomationConfig:
    repo_root: Path
    fgs_root: Path
    state_root: Path
    python_bin: Path
    sb_root: Path = Path("/cm/shared/apps/slurm/current/bin")
    seeds: tuple[str, ...] = ("rs10", "rs13")
    audit_workers: int = 24
    poll_seconds: int = 60
    backoff_initial: int = 30
    backoff_cap: int = 600
    supervisor_walltime: str = "48:00:00"
    supervisor_signal_lead: int = 600
    account: str = "ssalzbe1-chess"
    partition: str = "parallel"
    mailto: str = "kuanhao.chao@gmail.com"
    legacy_job_id: str = "26719761"
    legacy_array: str = "2347-5000%6"
    legacy_name: str = "osai_rs13"
    runner: Runner = field(default=subprocess_runner, repr=False)

    @classmethod
    def from_environment(cls, repo_root: Optional[Path] = None) -> "AutomationConfig":
        root = repo_root or Path(__file__).resolve().parents[3]
        fgs_root = Path(os.environ.get(
            "FGS_ROOT",
            "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI_precompute_scores/src/openspliceai_prediction_workflow/full_genome_scoring",
        ))
        state_root = Path(os.environ.get(
            "AUTOMATION_STATE_ROOT",
            "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI_precompute_scores/results/full_genome_scoring_campaign/automation",
        ))
        return cls(
            repo_root=root, fgs_root=fgs_root, state_root=state_root,
            python_bin=Path(os.environ.get(
                "AUTOMATION_PYTHON", "/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python"
            )),
            sb_root=Path(os.environ.get("SLURM_BIN", "/cm/shared/apps/slurm/current/bin")),
            audit_workers=int(os.environ.get("AUDIT_WORKERS", "24")),
            poll_seconds=int(os.environ.get("AUTOMATION_POLL_SECONDS", "60")),
            backoff_initial=int(os.environ.get("AUTOMATION_BACKOFF_INITIAL", "30")),
            backoff_cap=int(os.environ.get("AUTOMATION_BACKOFF_CAP", "600")),
            supervisor_walltime=os.environ.get("SUPERVISOR_WALLTIME", "48:00:00"),
            supervisor_signal_lead=int(os.environ.get("SUPERVISOR_SIGNAL_LEAD", "600")),
            account=os.environ.get("AUTOMATION_ACCOUNT", "ssalzbe1-chess"),
            partition=os.environ.get("AUTOMATION_PARTITION", "parallel"),
            mailto=os.environ.get("MAILTO", "kuanhao.chao@gmail.com"),
            legacy_job_id=os.environ.get("LEGACY_JOB_ID", "26719761"),
            legacy_array=os.environ.get("LEGACY_ARRAY", "2347-5000%6"),
            legacy_name=os.environ.get("LEGACY_JOB_NAME", "osai_rs13"),
        )

    @property
    def campaign_script(self) -> Path:
        return self.fgs_root / "campaign.sh"

    @property
    def manifest_root(self) -> Path:
        # fgs_root is .../src/.../full_genome_scoring; project root is parents[2].
        return self.fgs_root.parents[2] / "results" / "full_genome_scoring_campaign" / "manifests"

    def slurm_tool(self, name: str) -> Path:
        return self.sb_root / name


class Scheduler:
    def __init__(self, config: AutomationConfig):
        self.config = config

    def _run(self, argv: Sequence[str]) -> CommandResult:
        result = self.config.runner(argv, self.config.repo_root)
        if result.returncode != 0:
            raise SchedulerError(
                f"scheduler command failed ({result.returncode}): {shlex.join(result.argv)}\n"
                f"{result.stderr.strip()}"
            )
        return result

    def ping(self) -> bool:
        result = self.config.runner(
            (str(self.config.slurm_tool("scontrol")), "ping"), self.config.repo_root
        )
        return result.returncode == 0 and "UP" in result.stdout.upper()

    def show_job(self, job_id: str) -> str:
        return self._run((str(self.config.slurm_tool("scontrol")), "show", "job", job_id)).stdout

    def cancel(self, job_id: str) -> str:
        return self._run((str(self.config.slurm_tool("scancel")), job_id)).stdout

    def job_state(self, job_id: str) -> str:
        result = self.config.runner((str(self.config.slurm_tool("sacct")), "-j", job_id,
                                     "-X", "-n", "-o", "State"), self.config.repo_root)
        if result.returncode != 0:
            raise SchedulerError(result.stderr.strip() or "sacct failed")
        states = [line.strip().split()[0].split("+")[0].upper()
                  for line in result.stdout.splitlines() if line.strip()]
        return states[-1] if states else "UNKNOWN"


class CampaignSupervisor:
    def __init__(self, config: AutomationConfig, *, dry_run: bool = False, once: bool = False):
        self.config = config
        self.scheduler = Scheduler(config)
        self.dry_run = dry_run
        self.once = once
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.config.state_root / "current_state.json"
        self.events_path = self.config.state_root / "events.jsonl"
        self.log_dir = self.config.state_root / "commands"
        self.lock_path = self.config.state_root / "supervisor.lock"
        self._lock_handle = None

    def _initial_state(self) -> dict:
        return {
            "schema_version": 1, "state": "WAIT_SCHEDULER",
            "created_at": utc_now(), "updated_at": utc_now(),
            "transitions": 0, "jobs": {}, "artifacts": {}, "last_error": None,
        }

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return self._initial_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("state") not in STATES:
            raise ValueError(f"invalid automation state: {self.state_path}")
        return state

    def save_state(self, state: Mapping) -> None:
        atomic_json(self.state_path, state)

    def event(self, event: str, **fields: object) -> None:
        record = {"timestamp": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def acquire_lock(self) -> None:
        self._lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another supervisor holds {self.lock_path}") from exc

    def release_lock(self) -> None:
        if self._lock_handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def notify(self, subject: str, body: str) -> None:
        self.event("notification", subject=subject, body=body)
        if self.dry_run or not self.config.mailto:
            return
        mail = Path("/usr/bin/mail")
        if not mail.exists():
            self.event("notification_failed", reason="/usr/bin/mail not found")
            return
        result = subprocess.run([str(mail), "-s", subject, self.config.mailto], input=body,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            self.event("notification_failed", returncode=result.returncode, stderr=result.stderr)

    def command(self, state: dict, name: str, argv: Sequence[str], *, cwd: Optional[Path] = None) -> CommandResult:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        index = len(state.get("commands", []))
        log_path = self.log_dir / f"{index:04d}-{name}.log"
        self.event("command_started", name=name, argv=list(argv))
        result = self.config.runner(argv, cwd or self.config.repo_root)
        log_path.write_text(
            f"$ {shlex.join(tuple(str(item) for item in argv))}\nexit={result.returncode}\n\n"
            f"{result.stdout}\n{result.stderr}", encoding="utf-8"
        )
        state.setdefault("commands", []).append({
            "name": name, "argv": list(argv), "exit": result.returncode,
            "log": str(log_path), "at": utc_now(),
        })
        self.save_state(state)
        self.event("command", name=name, argv=list(argv), returncode=result.returncode, log=str(log_path))
        return result

    def transition(self, state: dict, next_state: str, reason: str) -> None:
        if next_state not in STATES:
            raise ValueError(next_state)
        previous = state["state"]
        state["state"] = next_state
        state["updated_at"] = utc_now()
        if next_state != "PAUSED_ERROR":
            state["last_error"] = None
        state["transitions"] = int(state.get("transitions", 0)) + 1
        state["last_transition"] = {"from": previous, "to": next_state, "reason": reason, "at": utc_now()}
        self.save_state(state)
        self.event("transition", previous=previous, state=next_state, reason=reason)
        self.notify(f"OpenSpliceAI campaign: {next_state}", f"Transitioned {previous} -> {next_state}: {reason}")

    def fail(self, state: dict, error: Exception) -> None:
        state["last_error"] = {"type": type(error).__name__, "message": str(error), "at": utc_now()}
        self.save_state(state)
        self.event("error", error_type=type(error).__name__, message=str(error))
        self.notify("OpenSpliceAI campaign paused", str(error))
        self.transition(state, "PAUSED_ERROR", str(error))

    def _run_campaign(self, state: dict, name: str, command: str, *args: str) -> CommandResult:
        return self.command(state, name, (str(self.config.campaign_script), command, *args), cwd=self.config.fgs_root)

    def _manifest(self, seed: str) -> Path:
        return self.config.manifest_root / f"{seed}.tsv"

    def manifest_complete(self, seed: str) -> bool:
        meta = Path(str(self._manifest(seed)) + ".meta.json")
        if not meta.is_file():
            return False
        data = json.loads(meta.read_text(encoding="utf-8"))
        counts = data.get("state_counts", {})
        return (data.get("chunk_count") == 100000 and counts.get("valid") == 100000
                and data.get("retry_chunks") == 0
                and counts.get("missing", 0) == 0 and counts.get("empty", 0) == 0
                and counts.get("truncated", 0) == 0)

    def _parse_score_jobs(self, output: str) -> dict[str, str]:
        return {seed: job for seed, job in SUBMITTED_SCORE_RE.findall(output)}

    def _parse_analysis_jobs(self, output: str) -> tuple[str, str, str]:
        match = SUBMITTED_ANALYSIS_RE.search(output)
        if not match:
            raise SchedulerError(f"could not parse analysis job IDs from output: {output!r}")
        return match.group("map"), match.group("reduce"), match.group("report")

    def _legacy_matches(self, raw: str) -> bool:
        tokens = raw.split()
        if not all(item in tokens for item in (
            f"JobName={self.config.legacy_name}", "JobState=PENDING", "Reason=JobHeldUser"
        )):
            return False
        array_ok = (f"ArrayTaskId={self.config.legacy_array}" in raw
                    or f"_{self.config.legacy_array}" in raw)
        if not array_ok:
            return False
        markers = ("run_selected.sh", "submit.sh", "robust_submit.sh", "fill_and_dedup.sh", "nuke_submit.sh", "score_chunk.sbatch")
        return not ("Command=" in raw or "WorkDir=" in raw) or any(marker in raw for marker in markers)

    def step_wait_scheduler(self, state: dict) -> None:
        if not self.scheduler.ping():
            current = int(state.get("scheduler_backoff", self.config.backoff_initial))
            state["scheduler"] = {"state": "unavailable", "checked_at": utc_now()}
            state["scheduler_backoff"] = min(current * 2, self.config.backoff_cap)
            self.save_state(state)
            self.event("scheduler_unavailable", next_retry_seconds=current)
            return
        state["scheduler"] = {"state": "up", "checked_at": utc_now()}
        state["scheduler_backoff"] = self.config.backoff_initial
        self.save_state(state)
        resume = state.pop("resume_state", None)
        self.save_state(state)
        self.transition(state, resume or "CLEAN_LEGACY", "Slurm controller is reachable")

    def step_clean_legacy(self, state: dict) -> None:
        try:
            raw = self.scheduler.show_job(self.config.legacy_job_id)
        except SchedulerError as exc:
            if "not found" in str(exc).lower() or "invalid job id" in str(exc).lower():
                self.event("legacy_absent", job_id=self.config.legacy_job_id)
                self.transition(state, "AUDIT", "legacy job absent")
                return
            raise
        if not self._legacy_matches(raw):
            raise CampaignGuardError(f"legacy job {self.config.legacy_job_id} does not match exact cancellation signature")
        self.scheduler.cancel(self.config.legacy_job_id)
        state.setdefault("legacy", {})["canceled"] = {"job_id": self.config.legacy_job_id, "evidence": raw, "at": utc_now()}
        self.save_state(state)
        self.event("legacy_canceled", job_id=self.config.legacy_job_id)
        self.notify("OpenSpliceAI legacy job canceled", f"Canceled exact legacy job {self.config.legacy_job_id}")
        self.transition(state, "AUDIT", "exact incompatible held legacy job canceled")

    def step_audit(self, state: dict) -> None:
        result = self._run_campaign(state, "audit", "audit", "--workers", str(self.config.audit_workers), *self.config.seeds)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        for seed in self.config.seeds:
            manifest = self._manifest(seed)
            meta = Path(str(manifest) + ".meta.json")
            state.setdefault("artifacts", {})[f"{seed}_manifest"] = {
                "path": str(manifest), "sha256": sha256_file(manifest) if manifest.is_file() else None,
                "meta_sha256": sha256_file(meta) if meta.is_file() else None,
            }
        self.save_state(state)
        self.transition(state, "PLAN", "fresh manifests written")

    def step_plan(self, state: dict) -> None:
        result = self._run_campaign(state, "plan", "plan-resume", *self.config.seeds)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        self.transition(state, "SUBMIT_SCORE", "both immutable resume plans validated")

    def step_submit_score(self, state: dict) -> None:
        result = self._run_campaign(state, "submit-score", "submit-resume", *self.config.seeds)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        jobs = self._parse_score_jobs(result.stdout)
        state.setdefault("jobs", {}).setdefault("score", {}).update(jobs)
        state["score_submission_output"] = result.stdout
        if not jobs and all(self.manifest_complete(seed) for seed in self.config.seeds):
            self.transition(state, "BUILD_PAIRS", "both seeds were already complete")
            return
        if set(jobs) != set(self.config.seeds):
            raise SchedulerError(f"score submission did not return both seed job IDs: {jobs}")
        self.save_state(state)
        self.transition(state, "MONITOR_SCORE", "serialized scoring arrays submitted")

    def step_monitor_score(self, state: dict) -> None:
        jobs = state.get("jobs", {}).get("score", {})
        if not jobs:
            self.transition(state, "RECONCILE", "no score jobs recorded")
            return
        states = {seed: self.scheduler.job_state(str(job)) for seed, job in jobs.items()}
        state.setdefault("jobs", {})["score_states"] = states
        self.save_state(state)
        if any(value not in TERMINAL_JOB_STATES for value in states.values()):
            self.event("score_jobs_running", states=states)
            return
        self.transition(state, "RECONCILE", f"score arrays terminal: {states}")

    def step_reconcile(self, state: dict) -> None:
        result = self._run_campaign(state, "reconcile", "reconcile", *self.config.seeds)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        if all(self.manifest_complete(seed) for seed in self.config.seeds):
            self.transition(state, "BUILD_PAIRS", "both fresh manifests contain 100,000 valid chunks")
            return
        jobs = self._parse_score_jobs(result.stdout)
        if jobs:
            state.setdefault("jobs", {})["score"] = jobs
            self.save_state(state)
            self.transition(state, "MONITOR_SCORE", "reconcile submitted invalid chunks")
            return
        self.event("reconcile_incomplete", manifests={seed: self.manifest_complete(seed) for seed in self.config.seeds})

    def step_build_pairs(self, state: dict) -> None:
        pair_root = self.config.state_root / "pairs"
        pair_root.mkdir(parents=True, exist_ok=True)
        rs10, rs13, seeds = pair_root / "rs10_spliceai.tsv", pair_root / "rs13_spliceai.tsv", pair_root / "rs10_rs13.tsv"
        base = (str(self.config.python_bin), "-m", "validation.full_snv_concordance", "build-pairs")
        result = self.command(state, "build-pairs-rs10-seeds", (*base, "--left-manifest", str(self._manifest("rs10")), "--concordance-output", str(rs10), "--right-manifest", str(self._manifest("rs13")), "--seeds-output", str(seeds), "--require-left-count", "100000", "--require-overlap-count", "100000"))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        result = self.command(state, "build-pairs-rs13", (*base, "--left-manifest", str(self._manifest("rs13")), "--concordance-output", str(rs13), "--require-left-count", "100000"))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        for name, path in (("rs10_spliceai", rs10), ("rs13_spliceai", rs13), ("rs10_rs13", seeds)):
            if not path.is_file():
                raise FileNotFoundError(path)
            state.setdefault("artifacts", {})[f"pairs_{name}"] = {"path": str(path), "sha256": sha256_file(path)}
        self.save_state(state)
        self.transition(state, "SUBMIT_ANALYSIS", "frozen complete pair files created")

    def _analysis_command(self, kind: str, pairs: Path, output: Path, label: str, overlap: Optional[int] = None) -> tuple[str, ...]:
        script = self.config.repo_root / "validation" / "full_snv_concordance" / "slurm" / "run_map_reduce.sh"
        command = (str(script), "--kind", kind, "--pairs-file", str(pairs), "--output-dir", str(output), "--run-label", label, "--finality", "final", "--expected-total-chunks", "100000", "--python", str(self.config.python_bin), "--submit")
        return command + (("--expected-overlap-count", str(overlap)) if overlap is not None else ())

    def step_submit_analysis(self, state: dict) -> None:
        timestamp = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        pair_root, analysis_root = self.config.state_root / "pairs", self.config.state_root / "analysis" / timestamp
        runs = (("rs10_spliceai", "concordance", pair_root / "rs10_spliceai.tsv", analysis_root / "rs10_spliceai", None), ("rs13_spliceai", "concordance", pair_root / "rs13_spliceai.tsv", analysis_root / "rs13_spliceai", None), ("rs10_rs13", "seeds", pair_root / "rs10_rs13.tsv", analysis_root / "rs10_rs13", 100000))
        analysis_jobs = {}
        for label, kind, pairs, output, overlap in runs:
            result = self.command(state, f"submit-analysis-{label}", self._analysis_command(kind, pairs, output, label, overlap))
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            map_job, reduce_job, report_job = self._parse_analysis_jobs(result.stdout)
            analysis_jobs[label] = {"map": map_job, "reduce": reduce_job, "report": report_job, "output": str(output)}
            state.setdefault("artifacts", {})[f"analysis_{label}"] = {
                "path": str(output),
                "report_markdown": str(output / "report" / "report.md"),
                "report_html": str(output / "report" / "report.html"),
            }
        state.setdefault("jobs", {})["analysis"] = analysis_jobs
        self.save_state(state)
        self.transition(state, "MONITOR_ANALYSIS", "three final analysis chains submitted")

    def step_monitor_analysis(self, state: dict) -> None:
        analysis = state.get("jobs", {}).get("analysis", {})
        if not analysis:
            raise SchedulerError("no analysis jobs recorded")
        states = {}
        all_terminal = True
        for label, jobs in analysis.items():
            states[label] = {}
            for kind in ("map", "reduce", "report"):
                current = self.scheduler.job_state(str(jobs[kind]))
                states[label][kind] = current
                all_terminal = all_terminal and current in TERMINAL_JOB_STATES
        state.setdefault("jobs", {})["analysis_states"] = states
        self.save_state(state)
        if not all_terminal:
            self.event("analysis_jobs_running", states=states)
            return
        failures = [f"{label}:{kind}={value}" for label, entries in states.items() for kind, value in entries.items() if value not in SUCCESS_JOB_STATES]
        if failures:
            raise RuntimeError("analysis jobs failed: " + ", ".join(failures))
        for jobs in analysis.values():
            report_dir = Path(jobs["output"]) / "report"
            for filename in ("report.md", "report.html"):
                if not (report_dir / filename).is_file():
                    raise FileNotFoundError(report_dir / filename)
        self.transition(state, "PUBLISH_SUMMARY", "all final reports completed")

    def step_publish_summary(self, state: dict) -> None:
        summary = {"schema_version": 1, "kind": "full-snv-concordance-campaign", "status": "complete", "created_at": state.get("created_at"), "completed_at": utc_now(), "jobs": state.get("jobs", {}), "artifacts": state.get("artifacts", {}), "scope": ["rs10_vs_spliceai", "rs13_vs_spliceai", "rs10_vs_rs13"], "external_database_used": False}
        summary_json = self.config.state_root / "summary.json"
        atomic_json(summary_json, summary)
        summary_md = self.config.state_root / "summary.md"
        lines = ["# Full-SNV scoring and concordance campaign", "", "Status: **COMPLETE**", "", "## Comparisons", "", "- rs10 versus SpliceAI", "- rs13 versus SpliceAI", "- rs10 versus rs13", "", "## Artifacts", ""]
        for key, value in sorted(summary["artifacts"].items()):
            if isinstance(value, Mapping) and value.get("path"):
                lines.append(f"- `{key}`: `{value['path']}` ({value.get('sha256', 'unhashed')})")
        summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary_html = self.config.state_root / "summary.html"
        escaped = "\n".join(f"<li><code>{line}</code></li>" for line in lines[5:])
        summary_html.write_text("<!doctype html><meta charset='utf-8'><title>Full-SNV concordance campaign</title><h1>Full-SNV scoring and concordance campaign</h1><p>Status: <strong>COMPLETE</strong></p><ul>" + escaped + "</ul>", encoding="utf-8")
        state.setdefault("artifacts", {})["summary"] = {"json": str(summary_json), "markdown": str(summary_md), "html": str(summary_html), "sha256": sha256_file(summary_json)}
        self.save_state(state)
        self.transition(state, "COMPLETE", "campaign summary published")

    def step(self, state: dict) -> None:
        handlers = {"WAIT_SCHEDULER": self.step_wait_scheduler, "CLEAN_LEGACY": self.step_clean_legacy, "AUDIT": self.step_audit, "PLAN": self.step_plan, "SUBMIT_SCORE": self.step_submit_score, "MONITOR_SCORE": self.step_monitor_score, "RECONCILE": self.step_reconcile, "BUILD_PAIRS": self.step_build_pairs, "SUBMIT_ANALYSIS": self.step_submit_analysis, "MONITOR_ANALYSIS": self.step_monitor_analysis, "PUBLISH_SUMMARY": self.step_publish_summary}
        handler = handlers.get(state["state"])
        if handler:
            handler(state)

    def run(self) -> int:
        self.acquire_lock()
        try:
            state = self.load_state()
            self.save_state(state)
            self.event("supervisor_started", state=state["state"], dry_run=self.dry_run, once=self.once)
            while state["state"] not in {"COMPLETE", "PAUSED_ERROR"}:
                before = state["state"]
                try:
                    self.step(state)
                except Exception as exc:
                    if before == "WAIT_SCHEDULER" and isinstance(exc, SchedulerError):
                        self.event("scheduler_wait", message=str(exc))
                    elif isinstance(exc, SchedulerError):
                        state["resume_state"] = before
                        state["state"] = "WAIT_SCHEDULER"
                        state["last_error"] = {"type": type(exc).__name__, "message": str(exc), "at": utc_now()}
                        self.save_state(state)
                        self.event("scheduler_lost", previous_state=before, message=str(exc))
                        self.notify("OpenSpliceAI campaign waiting for Slurm", str(exc))
                    else:
                        self.fail(state, exc)
                        break
                state = self.load_state()
                if self.once:
                    break
                if state["state"] == before:
                    delay = int(state.get("scheduler_backoff", self.config.backoff_initial)) if before == "WAIT_SCHEDULER" else self.config.poll_seconds
                    time.sleep(delay)
            if state["state"] == "COMPLETE":
                self.notify("OpenSpliceAI campaign complete", str(self.config.state_root / "summary.md"))
                return 0
            return 1 if state["state"] == "PAUSED_ERROR" else 0
        finally:
            self.release_lock()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-root", type=Path)
    result.add_argument("--repo-root", type=Path)
    result.add_argument("--fgs-root", type=Path)
    result.add_argument("--python", dest="python_bin", type=Path)
    result.add_argument("--once", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--show-state", action="store_true")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    config = AutomationConfig.from_environment(args.repo_root)
    if args.state_root:
        config.state_root = args.state_root
    if args.fgs_root:
        config.fgs_root = args.fgs_root
    if args.python_bin:
        config.python_bin = args.python_bin
    if args.show_state:
        path = config.state_root / "current_state.json"
        print(path.read_text(encoding="utf-8") if path.exists() else json.dumps({"state": "WAIT_SCHEDULER"}, indent=2))
        return 0
    return CampaignSupervisor(config, dry_run=args.dry_run, once=args.once).run()


if __name__ == "__main__":
    raise SystemExit(main())
