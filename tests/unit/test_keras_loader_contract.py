"""Loader failure contracts are testable without installing the optional backend."""
import sys
import types

import pytest

from openspliceai.variant.utils import load_keras_models


@pytest.fixture
def fake_keras(monkeypatch):
    def load(path):
        if "invalid" in str(path):
            raise ValueError("invalid model")
        return str(path)
    keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=load))
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace(keras=keras))


@pytest.mark.parametrize("kind", ["missing", "empty", "corrupt", "partial"])
def test_keras_loader_errors_are_failures(tmp_path, fake_keras, kind):
    if kind == "missing":
        path = tmp_path / "missing.h5"
    elif kind == "corrupt":
        path = tmp_path / "invalid.h5"
        path.touch()
    else:
        path = tmp_path
        if kind == "partial":
            (path / "valid.h5").touch()
            (path / "invalid.h5").touch()
    with pytest.raises(SystemExit) as exc:
        load_keras_models(str(path))
    assert exc.value.code == 1


def test_keras_directory_and_file_load_complete_requested_set(tmp_path, fake_keras):
    for name in ("one.h5", "two.h5"):
        (tmp_path / name).touch()
    assert set(load_keras_models(str(tmp_path))) == {str(tmp_path / "one.h5"), str(tmp_path / "two.h5")}
    assert load_keras_models(str(tmp_path / "one.h5")) == [str(tmp_path / "one.h5")]
