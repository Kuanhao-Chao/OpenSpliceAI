import json
from pathlib import Path
import stat

import pytest
import validation.full_snv_concordance.external_score_plan as score_plan_module

from validation.full_snv_concordance.external_score_plan import (
    EXPECTED_NAMES,
    OFFICIAL_NAME,
    _clean_child_environment,
    build_bundle,
    commit_submission,
    parse_plan,
    record_submission,
    run_evaluation,
    run_task,
    sha256_file,
    validate_scored_vcf,
    verify_run,
    write_header_complete_vcf,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fixture_plan(tmp_path: Path) -> tuple[Path, dict]:
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nAAAA\n>chr2\nCCCC\n", encoding="utf-8")
    (tmp_path / "reference.fa.fai").write_text(
        "chr1\t4\t6\t4\t5\nchr2\t4\t17\t4\t5\n", encoding="utf-8"
    )
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("#NAME\tCHROM\nGENE\tchr1\n", encoding="utf-8")
    source = tmp_path / "variants.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tC\t.\t.\t.\n",
        encoding="utf-8",
    )
    harmonized = tmp_path / "harmonized.tsv"
    harmonized.write_text(
        "dataset\tsource_row\tchrom\tpos\tref\talt\tgene\tlabel\tcohort\n"
        "smith\t1\tchr1\t2\tA\tC\tGENE\tSDV\tcohort\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores"
    scores.mkdir()
    models = tmp_path / "osai_models"
    models.mkdir()
    for name in ("rs10", "rs11", "rs12", "rs13", "rs14"):
        (models / f"model_10000nt_{name}.pt").write_bytes(name.encode("ascii"))
    official_package = tmp_path / "spliceai_package"
    (official_package / "models").mkdir(parents=True)
    (official_package / "__init__.py").write_text("", encoding="utf-8")
    for number in range(1, 6):
        (official_package / "models" / f"spliceai{number}.h5").write_bytes(bytes([number]))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executables = {
        "python_bin": _executable(bin_dir / "python"),
        "spliceai_bin": _executable(bin_dir / "spliceai"),
        "openspliceai_bin": _executable(bin_dir / "openspliceai"),
    }
    commands = [
        {
            "name": OFFICIAL_NAME,
            "ready": True,
            "argv": [
                "spliceai", "-I", str(source), "-O", str(scores / "spliceai.vcf"),
                "-R", str(reference), "-A", "grch38", "-D", "50", "-M", "1",
            ],
        }
    ]
    for name in ("rs10", "rs11", "rs12", "rs13", "rs14", "ensemble"):
        model = models if name == "ensemble" else models / f"model_10000nt_{name}.pt"
        commands.append(
            {
                "name": name,
                "ready": True,
                "argv": [
                    "openspliceai", "variant", "-R", str(reference), "-A", str(annotation),
                    "--model", str(model), "-I", str(source), "-O", str(scores / f"{name}.vcf"),
                    "--precision", "5", "-D", "50", "-f", "10000", "-M", "1",
                    "-b", "128",
                ],
            }
        )
    source_status = {
        "id": "smith",
        "state": "ready",
        "rows": 1,
        "accepted": 1,
        "rejected": 0,
    }
    manifest = tmp_path / "external_sources.json"
    manifest.write_text(json.dumps({"datasets": [{"id": "smith"}]}), encoding="utf-8")
    document = {
        "kind": "external-validation-preparation",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "validation_vcf": str(source),
        "harmonized_tsv": str(harmonized),
        "reference_fasta": str(reference),
        "reference_sha256": sha256_file(reference),
        "scoring_commands": commands,
        "sources": [source_status],
        "dataset_completeness": {
            "mode": "complete",
            "configured_dataset_ids": ["smith"],
            "complete": True,
            "incomplete_datasets": [],
        },
    }
    plan = tmp_path / "validation_plan.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    return plan, {**executables, "official_package_dir": official_package}


def _parse(plan: Path, paths: dict) -> dict:
    return parse_plan(plan, **paths)


def _scored(path: Path, info_key: str) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        f'##INFO=<ID={info_key},Number=.,Type=String,Description="score">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        f"chr1\t2\t.\tA\tC\t.\t.\t{info_key}=C|GENE|0|0|0|0|0|0|0|0\n",
        encoding="utf-8",
    )


def _mock_scorer(path: Path, info_key: str) -> None:
    payload = (
        "##fileformat=VCFv4.2\n"
        f'##INFO=<ID={info_key},Number=.,Type=String,Description="score">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        f"chr1\t2\t.\tA\tC\t.\t.\t{info_key}=C|GENE|0|0|0|0|0|0|0|0\n"
    )
    path.write_text(
        "#!/bin/sh\n"
        "output=\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-O\" ]; then output=$2; break; fi\n"
        "  shift\n"
        "done\n"
        f"/usr/bin/printf %s '{payload}' > \"$output\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_parse_plan_requires_exact_seven_and_pins_executables(tmp_path):
    plan, paths = _fixture_plan(tmp_path)
    parsed = _parse(plan, paths)
    assert [task["name"] for task in parsed["tasks"]] == list(EXPECTED_NAMES)
    assert parsed["score_mask"] == 1
    assert parsed["tasks"][0]["argv"][0] == str(paths["spliceai_bin"].resolve())
    assert all(
        task["argv"][0] == str(paths["openspliceai_bin"].resolve())
        for task in parsed["tasks"][1:]
    )

    document = json.loads(plan.read_text(encoding="utf-8"))
    document["scoring_commands"].pop()
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing predictors"):
        _parse(plan, paths)

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    plan, paths = _fixture_plan(mixed)
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["scoring_commands"][1]["argv"][
        document["scoring_commands"][1]["argv"].index("-M") + 1
    ] = "0"
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="same masking mode"):
        _parse(plan, paths)

    altered = tmp_path / "altered"
    altered.mkdir()
    plan, paths = _fixture_plan(altered)
    document = json.loads(plan.read_text(encoding="utf-8"))
    command = document["scoring_commands"][1]["argv"]
    command[command.index("-D") + 1] = "5000"
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="must use -D 50"):
        _parse(plan, paths)

    injected = tmp_path / "injected"
    injected.mkdir()
    plan, paths = _fixture_plan(injected)
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["scoring_commands"][0]["argv"].extend(["--unexpected", "value"])
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="only the prespecified flags"):
        _parse(plan, paths)


def test_header_completion_preserves_record_and_declares_reference(tmp_path):
    plan, _ = _fixture_plan(tmp_path)
    document = json.loads(plan.read_text(encoding="utf-8"))
    destination = tmp_path / "complete.vcf"
    summary = write_header_complete_vcf(
        document["validation_vcf"], destination, document["reference_fasta"] + ".fai"
    )
    text = destination.read_text(encoding="utf-8")
    assert "##contig=<ID=chr1,length=4>\n" in text
    assert "##contig=<ID=chr2,length=4>\n" in text
    assert text.count("chr1\t2\t.\tA\tC\t.\t.\t.\n") == 1
    assert summary == {"records": 1, "reference_contigs": 2, "used_contigs": ["chr1"]}


def test_bundle_verification_and_unreceipted_existing_score_is_rejected(tmp_path):
    plan, paths = _fixture_plan(tmp_path)
    run_dir = tmp_path / "run"
    prepared = build_bundle(plan, run_dir, REPO_ROOT, persist=True, **paths)
    anchor = prepared["bundle_sha256"]
    assert verify_run(run_dir, anchor)["task_count"] == 7
    frozen = run_dir / "input" / "validation_variants.hg38.vcf"
    assert "##contig=<ID=chr1,length=4>" in frozen.read_text(encoding="utf-8")

    tasks = json.loads((run_dir / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    output = Path(tasks[0]["final_output"])
    _scored(output, "SpliceAI")
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="without a receipt"):
        run_task(run_dir, 0, anchor)
    assert output.read_bytes() == before


def test_existing_malformed_receipt_fails_closed(tmp_path):
    plan, paths = _fixture_plan(tmp_path)
    run_dir = tmp_path / "run"
    anchor = build_bundle(plan, run_dir, REPO_ROOT, persist=True, **paths)[
        "bundle_sha256"
    ]
    receipt_path = run_dir / "receipts" / "task-00.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt_path.chmod(0o444)
    with pytest.raises(ValueError, match="schema/key set mismatch"):
        run_task(run_dir, 0, anchor)


def test_external_bundle_anchor_and_worker_semantics_reject_post_freeze_tamper(tmp_path):
    plan, paths = _fixture_plan(tmp_path)
    run_dir = tmp_path / "run"
    anchor = build_bundle(plan, run_dir, REPO_ROOT, persist=True, **paths)[
        "bundle_sha256"
    ]
    tasks_path = run_dir / "tasks.json"
    provenance_path = run_dir / "provenance.json"
    tasks_path.chmod(0o644)
    provenance_path.chmod(0o644)
    document = json.loads(tasks_path.read_text(encoding="utf-8"))
    document["tasks"][0]["argv"] = [
        "/bin/sh",
        "-c",
        "exit 0",
        "ignored",
        "-O",
        document["tasks"][0]["final_output"],
    ]
    tasks_path.write_text(json.dumps(document), encoding="utf-8")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["tasks_sha256"] = sha256_file(tasks_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle anchor mismatch"):
        verify_run(run_dir, anchor)
    # Even if a caller incorrectly anchors the rewritten provenance, workers
    # reconstruct and reject the task's non-prescribed command semantics.
    with pytest.raises(ValueError, match="strict command shape"):
        verify_run(run_dir, sha256_file(provenance_path))


def test_incomplete_dataset_plan_requires_explicit_sensitivity_label(tmp_path):
    plan, paths = _fixture_plan(tmp_path)
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["sources"][0].update(
        state="requires_allele_aware_liftover",
        rows=1,
        accepted=0,
        rejected=0,
        unprocessed=1,
    )
    incomplete = [
        {
            "id": "smith",
            "reasons": [
                "state=requires_allele_aware_liftover",
                "zero accepted rows",
                "1 unprocessed rows",
            ],
        }
    ]
    document["dataset_completeness"] = {
        "mode": "incomplete_blocked",
        "configured_dataset_ids": ["smith"],
        "complete": False,
        "incomplete_datasets": incomplete,
    }
    plan.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="primary external scoring requires every"):
        _parse(plan, paths)

    document["dataset_completeness"]["mode"] = "incomplete_sensitivity"
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert _parse(plan, paths)["dataset_completeness"]["mode"] == "incomplete_sensitivity"


def test_child_environment_is_allowlisted_and_pins_repo_imports(tmp_path, monkeypatch):
    plan, paths = _fixture_plan(tmp_path)
    run_dir = tmp_path / "run"
    build_bundle(plan, run_dir, REPO_ROOT, persist=True, **paths)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    monkeypatch.setenv("PYTHONPATH", "/tmp/attacker")
    monkeypatch.setenv("PYTHONHOME", "/tmp/attacker-home")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/attacker.so")
    environment = _clean_child_environment(run_dir, provenance)
    assert environment["PYTHONPATH"] == str(REPO_ROOT)
    assert "PYTHONHOME" not in environment
    assert "LD_PRELOAD" not in environment


def test_tasks_publish_receipt_bound_run_local_snapshots_and_evaluate_only_them(
    tmp_path, monkeypatch
):
    plan, paths = _fixture_plan(tmp_path)
    _mock_scorer(paths["spliceai_bin"], "SpliceAI")
    _mock_scorer(paths["openspliceai_bin"], "OpenSpliceAI")
    run_dir = tmp_path / "run"
    anchor = build_bundle(plan, run_dir, REPO_ROOT, persist=True, **paths)[
        "bundle_sha256"
    ]
    loaded = score_plan_module._load_run(run_dir, anchor)
    context = (loaded[0], loaded[1], loaded[2], loaded[3])
    monkeypatch.setattr(
        score_plan_module, "_verified_run_context", lambda *_args, **_kwargs: context
    )
    receipts = [run_task(run_dir, index, anchor) for index in range(7)]
    for receipt in receipts:
        snapshot = Path(receipt["snapshot_path"])
        assert snapshot.parent == run_dir / "scores"
        assert sha256_file(snapshot) == receipt["snapshot_sha256"]
        assert snapshot.stat().st_mode & 0o222 == 0

    # Changing a separately published convenience output cannot change the
    # content-addressed score bytes handed to the evaluator.
    first_final = Path(receipts[0]["final_output"])
    first_final.write_text(
        first_final.read_text(encoding="utf-8").replace(
            "C|GENE|0|0|0|0|0|0|0|0", "C|GENE|0.9|0|0|0|0|0|0|0"
        ),
        encoding="utf-8",
    )
    captured = {}

    class Completed:
        returncode = 0

    def fake_run(argv, *, env, check):
        captured.update(argv=list(argv), env=dict(env), check=check)
        return Completed()

    monkeypatch.setattr(score_plan_module.subprocess, "run", fake_run)
    record_submission(run_dir, "101", "102", 0, anchor)
    commit_submission(run_dir, anchor)
    assert run_evaluation(run_dir, 0, anchor) == 0
    score_arguments = [
        captured["argv"][index + 1]
        for index, value in enumerate(captured["argv"][:-1])
        if value == "--score"
    ]
    assert len(score_arguments) == 7
    assert all(f"={run_dir / 'scores'}/" in value for value in score_arguments)
    assert "PYTHONHOME" not in captured["env"]
    assert "LD_PRELOAD" not in captured["env"]
    provenance_path = Path(
        captured["argv"][captured["argv"].index("--run-provenance") + 1]
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["bundle_sha256"] == anchor
    assert len(provenance["receipts"]) == 7
    assert provenance["submission"]["document"]["state"] == "committed"
    assert (
        captured["argv"][captured["argv"].index("--run-provenance-sha256") + 1]
        == sha256_file(provenance_path)
    )


def test_validate_scored_vcf_fails_on_truncation_count_and_info(tmp_path):
    plan, _ = _fixture_plan(tmp_path)
    source = Path(json.loads(plan.read_text(encoding="utf-8"))["validation_vcf"])
    output = tmp_path / "score.vcf"
    _scored(output, "OpenSpliceAI")
    assert validate_scored_vcf(source, output, "OpenSpliceAI")["records"] == 1
    output.write_bytes(output.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="final newline"):
        validate_scored_vcf(source, output, "OpenSpliceAI")

    _scored(output, "OpenSpliceAI")
    output.write_text(
        output.read_text(encoding="utf-8").replace(
            "C|GENE|0|0|0|0|0|0|0|0", "C|GENE|1.1|0|0|0|0|0|0|0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        validate_scored_vcf(source, output, "OpenSpliceAI")


def test_validate_scored_vcf_rejects_grossly_sparse_annotations(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tC\t.\t.\t.\n"
        "chr1\t2\t.\tA\tC\t.\t.\t.\n",
        encoding="utf-8",
    )
    output = tmp_path / "score.vcf"
    output.write_text(
        "##fileformat=VCFv4.2\n"
        '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="score">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tC\t.\t.\tOpenSpliceAI=C|GENE|0|0|0|0|0|0|0|0\n"
        "chr1\t2\t.\tA\tC\t.\t.\t.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="annotation coverage"):
        validate_scored_vcf(source, output, "OpenSpliceAI")
    assert validate_scored_vcf(
        source, output, "OpenSpliceAI", minimum_annotated_fraction=0.5
    )["annotated_fraction"] == 0.5


def test_external_slurm_scripts_have_valid_bash_syntax():
    import subprocess

    scripts = REPO_ROOT / "validation" / "full_snv_concordance" / "slurm"
    for name in (
        "external_score_array.sbatch",
        "external_evaluate.sbatch",
        "run_external_scoring.sh",
    ):
        subprocess.run(["bash", "-n", str(scripts / name)], check=True)
    launcher = (scripts / "run_external_scoring.sh").read_text(encoding="utf-8")
    score_worker = (scripts / "external_score_array.sbatch").read_text(encoding="utf-8")
    evaluation_worker = (scripts / "external_evaluate.sbatch").read_text(encoding="utf-8")
    assert launcher.count("--export=NONE") == 2
    assert "prepare_result=" in launcher
    assert "--expected-bundle-sha256" in launcher
    assert "--expected-bundle-sha256" in score_worker
    assert "--expected-bundle-sha256" in evaluation_worker
    assert "/usr/bin/env -i" in score_worker
    assert "/usr/bin/env -i" in evaluation_worker
