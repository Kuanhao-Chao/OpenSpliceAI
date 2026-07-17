"""Coverage for the top-level CLI dispatch in openspliceai/openspliceai.py::main.

Each subcommand entry point is monkeypatched so we assert the dispatch wiring (which
``args.command`` calls which function) without running the heavy pipelines.

Note: ``main()`` imports each subcommand's module lazily inside its own dispatch
branch (so the CLI never loads the heavy stack unless a subcommand runs; see
GitHub issue #19), so the patch targets are the functions in their *source*
modules -- exactly what ``from openspliceai.<pkg> import <mod>`` resolves to."""
import argparse
import types

import pytest

import openspliceai.openspliceai as osa


@pytest.mark.parametrize("command,target", [
    ("train", "openspliceai.train.train.train"),
    ("calibrate", "openspliceai.calibrate.calibrate.calibrate"),
    ("transfer", "openspliceai.transfer.transfer.transfer"),
    ("predict", "openspliceai.predict.predict.predict_cli"),
    ("variant", "openspliceai.variant.variant.variant"),
])
def test_main_dispatches_to_subcommand(monkeypatch, command, target):
    hit = {}
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command=command, verify_h5=False))
    monkeypatch.setattr(target, lambda args: hit.setdefault("ok", True))
    osa.main([command])
    assert hit.get("ok") is True


def test_main_create_data_runs_both_stages_and_verify(monkeypatch):
    seq = []
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command="create-data", verify_h5=True))
    monkeypatch.setattr("openspliceai.create_data.create_datafile.create_datafile", lambda args: seq.append("datafile"))
    monkeypatch.setattr("openspliceai.create_data.create_dataset.create_dataset", lambda args: seq.append("dataset"))
    monkeypatch.setattr("openspliceai.create_data.verify_h5_file.verify_h5", lambda args: seq.append("verify"))
    osa.main(["create-data"])
    assert seq == ["datafile", "dataset", "verify"]


def test_main_create_data_without_verify(monkeypatch):
    seq = []
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command="create-data", verify_h5=False))
    monkeypatch.setattr("openspliceai.create_data.create_datafile.create_datafile", lambda args: seq.append("datafile"))
    monkeypatch.setattr("openspliceai.create_data.create_dataset.create_dataset", lambda args: seq.append("dataset"))
    monkeypatch.setattr("openspliceai.create_data.verify_h5_file.verify_h5", lambda args: seq.append("verify"))
    osa.main(["create-data"])
    assert seq == ["datafile", "dataset"]   # verify skipped


def test_parse_args_requires_a_subcommand():
    with pytest.raises(SystemExit):
        osa.parse_args([])


def test_parse_args_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        osa.parse_args(["definitely-not-a-command"])


def test_parse_args_test_registers_disabled_test_subparser():
    """parse_args_test is defined but not wired into parse_args; call it directly so the
    (still-shipped) registration code is exercised."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    osa.parse_args_test(subparsers)
    ns = parser.parse_args(["test", "-m", "x", "-o", "o", "-p", "p", "-test", "t"])
    assert ns.command == "test" and ns.flanking_size == 80
