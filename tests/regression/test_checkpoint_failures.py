"""Invalid ensemble members must never silently change the requested model set."""
import shutil

import pytest
import torch

from openspliceai.predict.predict import load_pytorch_models as load_predict
from openspliceai.variant.utils import load_pytorch_models as load_variant
from openspliceai.calibrate.model_utils import initialize_model_and_optim


@pytest.fixture(params=["predict", "variant"])
def loader(request):
    if request.param == "predict":
        return lambda path: load_predict(str(path), torch.device("cpu"), 5000, 80)[0]
    return lambda path: load_variant(str(path), 80)


def test_missing_checkpoint_returns_failure(loader, tmp_path):
    with pytest.raises(SystemExit) as exc:
        loader(tmp_path / "missing.pt")
    assert exc.value.code == 1


@pytest.mark.parametrize("invalid_kind", ["corrupt", "empty_state", "wrong_shape"])
def test_one_valid_member_cannot_hide_invalid_member(loader, tmp_path, packaged_80nt_state, invalid_kind):
    shutil.copyfile(packaged_80nt_state, tmp_path / "valid.pt")
    bad = tmp_path / "invalid.pt"
    if invalid_kind == "corrupt":
        bad.write_text("not a checkpoint")
    elif invalid_kind == "empty_state":
        torch.save({}, bad)
    else:
        state = torch.load(packaged_80nt_state, map_location="cpu", weights_only=True)
        state["initial_conv.weight"] = torch.zeros(1)
        torch.save(state, bad)
    with pytest.raises(SystemExit) as exc:
        loader(tmp_path)
    assert exc.value.code == 1


def test_pth_extension_loads_and_preserves_weights(loader, tmp_path, packaged_80nt_state):
    shutil.copyfile(packaged_80nt_state, tmp_path / "valid.pth")
    models = loader(tmp_path)
    assert len(models) == 1
    expected = torch.load(packaged_80nt_state, map_location="cpu", weights_only=True)
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, expected[key])


@pytest.mark.parametrize("wrong_flank", [400, 2000, 10000])
def test_calibration_rejects_incompatible_checkpoint(packaged_80nt_state, wrong_flank):
    with pytest.raises(RuntimeError):
        initialize_model_and_optim(torch.device("cpu"), wrong_flank, packaged_80nt_state)


def test_calibration_rejects_empty_checkpoint(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({}, path)
    with pytest.raises(RuntimeError):
        initialize_model_and_optim(torch.device("cpu"), 80, path)
