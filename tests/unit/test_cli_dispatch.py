"""Coverage for the top-level CLI dispatch in openspliceai/openspliceai.py::main.

Each subcommand entry point is monkeypatched so we assert the dispatch wiring (which
``args.command`` calls which function) without running the heavy pipelines."""
import argparse
import types

import pytest

import openspliceai.openspliceai as osa


@pytest.mark.parametrize("command,module_attr,fn_attr", [
    ("train", "train", "train"),
    ("calibrate", "calibrate", "calibrate"),
    ("transfer", "transfer", "transfer"),
    ("predict", "predict", "predict_cli"),
    ("variant", "variant", "variant"),
])
def test_main_dispatches_to_subcommand(monkeypatch, command, module_attr, fn_attr):
    hit = {}
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command=command, verify_h5=False))
    monkeypatch.setattr(getattr(osa, module_attr), fn_attr, lambda args: hit.setdefault("ok", True))
    osa.main([command])
    assert hit.get("ok") is True


def test_main_create_data_runs_both_stages_and_verify(monkeypatch):
    seq = []
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command="create-data", verify_h5=True))
    monkeypatch.setattr(osa.create_datafile, "create_datafile", lambda args: seq.append("datafile"))
    monkeypatch.setattr(osa.create_dataset, "create_dataset", lambda args: seq.append("dataset"))
    monkeypatch.setattr(osa.verify_h5_file, "verify_h5", lambda args: seq.append("verify"))
    osa.main(["create-data"])
    assert seq == ["datafile", "dataset", "verify"]


def test_main_create_data_without_verify(monkeypatch):
    seq = []
    monkeypatch.setattr(osa, "parse_args",
                        lambda a=None: types.SimpleNamespace(command="create-data", verify_h5=False))
    monkeypatch.setattr(osa.create_datafile, "create_datafile", lambda args: seq.append("datafile"))
    monkeypatch.setattr(osa.create_dataset, "create_dataset", lambda args: seq.append("dataset"))
    monkeypatch.setattr(osa.verify_h5_file, "verify_h5", lambda args: seq.append("verify"))
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
