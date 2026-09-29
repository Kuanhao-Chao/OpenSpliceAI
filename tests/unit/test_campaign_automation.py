import json
from pathlib import Path

import pytest

from validation.full_snv_concordance.automation.supervisor import (
    AutomationConfig,
    CampaignGuardError,
    CampaignSupervisor,
    CommandResult,
    SchedulerError,
    parse_job_id,
)


def config_for(tmp_path, runner):
    project = tmp_path / "project"
    fgs = project / "src" / "workflow" / "full_genome_scoring"
    state = tmp_path / "automation"
    return AutomationConfig(
        repo_root=tmp_path,
        fgs_root=fgs,
        state_root=state,
        python_bin=Path("/usr/bin/python3"),
        runner=runner,
        mailto="",
        poll_seconds=0,
    )


def test_parse_job_id_requires_parsable_numeric_value():
    assert parse_job_id("12345") == "12345"
    assert parse_job_id("12345;cluster") == "12345"
    with pytest.raises(SchedulerError):
        parse_job_id("Submitted batch job 12345")


def test_legacy_signature_is_exact_and_does_not_match_new_job(tmp_path):
    supervisor = CampaignSupervisor(config_for(tmp_path, lambda *_: CommandResult((), 0)))
    exact = (
        "JobId=26719761_[2347-5000%6] JobName=osai_rs13 "
        "JobState=PENDING Reason=JobHeldUser ArrayTaskId=2347-5000%6 "
        "Command=/old/run_selected.sh"
    )
    assert supervisor._legacy_matches(exact)
    assert not supervisor._legacy_matches(exact.replace("JobName=osai_rs13", "JobName=osai_rs13_resume"))
    assert not supervisor._legacy_matches(exact.replace("Reason=JobHeldUser", "Reason=Resources"))
    assert not supervisor._legacy_matches(exact.replace("2347-5000%6", "1-5000%6"))


def test_legacy_guard_mismatch_is_not_treated_as_scheduler_outage(tmp_path):
    def runner(argv, _cwd):
        if list(argv)[-2:] == ["show", "job"]:
            return CommandResult(tuple(argv), 0, "JobName=osai_rs13 JobState=PENDING Reason=Resources", "")
        if "show" in argv and "job" in argv:
            return CommandResult(tuple(argv), 0, "JobName=osai_rs13 JobState=PENDING Reason=Resources", "")
        return CommandResult(tuple(argv), 0)

    supervisor = CampaignSupervisor(config_for(tmp_path, runner))
    with pytest.raises(CampaignGuardError):
        supervisor.step_clean_legacy(supervisor.load_state())


def test_scheduler_outage_checkpoints_backoff_without_advancing(tmp_path):
    def runner(argv, _cwd):
        if Path(argv[0]).name == "scontrol":
            return CommandResult(tuple(argv), 1, "", "controller unavailable")
        return CommandResult(tuple(argv), 0)

    config = config_for(tmp_path, runner)
    supervisor = CampaignSupervisor(config, once=True)
    state = supervisor.load_state()
    supervisor.step_wait_scheduler(state)
    state = supervisor.load_state()
    assert state["state"] == "WAIT_SCHEDULER"
    assert state["scheduler"]["state"] == "unavailable"
    assert state["scheduler_backoff"] == 60


def test_manifest_complete_requires_all_clean_states(tmp_path):
    supervisor = CampaignSupervisor(config_for(tmp_path, lambda *_: CommandResult((), 0)))
    manifest = supervisor._manifest("rs10")
    manifest.parent.mkdir(parents=True)
    manifest.write_text("chunk\n", encoding="utf-8")
    Path(str(manifest) + ".meta.json").write_text(
        json.dumps({"chunk_count": 100000, "retry_chunks": 0, "state_counts": {"valid": 100000}}),
        encoding="utf-8",
    )
    assert supervisor.manifest_complete("rs10")
    data = json.loads(Path(str(manifest) + ".meta.json").read_text())
    data["state_counts"]["missing"] = 1
    Path(str(manifest) + ".meta.json").write_text(json.dumps(data), encoding="utf-8")
    assert not supervisor.manifest_complete("rs10")


def test_score_submission_records_both_seed_ids(tmp_path):
    def runner(argv, _cwd):
        if "submit-resume" in argv:
            return CommandResult(tuple(argv), 0, "[rs10] submitted job 111\n[rs13] submitted job 222\n")
        return CommandResult(tuple(argv), 0)

    supervisor = CampaignSupervisor(config_for(tmp_path, runner))
    state = supervisor.load_state()
    supervisor.step_submit_score(state)
    state = supervisor.load_state()
    assert state["state"] == "MONITOR_SCORE"
    assert state["jobs"]["score"] == {"rs10": "111", "rs13": "222"}


def test_duplicate_supervisors_are_locked(tmp_path):
    config = config_for(tmp_path, lambda *_: CommandResult((), 0))
    first = CampaignSupervisor(config)
    second = CampaignSupervisor(config)
    first.acquire_lock()
    try:
        with pytest.raises(RuntimeError):
            second.acquire_lock()
    finally:
        first.release_lock()
