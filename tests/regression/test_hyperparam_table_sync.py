"""Published architecture schedules and every caller's context stay consistent."""
import numpy as np
import pytest

from openspliceai.model_config import model_hyperparameters
from openspliceai.train_base.openspliceai import SpliceAI

SCHEDULES = [
    (80, [11]*4, [1]*4, 36),
    (400, [11]*8, [1]*4 + [4]*4, 36),
    (2000, [11]*8 + [21]*4, [1]*4 + [4]*4 + [10]*4, 24),
    (10000, [11]*8 + [21]*4 + [41]*4, [1]*4 + [4]*4 + [10]*4 + [25]*4, 12),
]


@pytest.mark.parametrize("flank,windows,rates,batch", SCHEDULES)
def test_configuration_preserves_published_schedule(flank, windows, rates, batch):
    width, devices, actual_windows, actual_rates, actual_batch = model_hyperparameters(flank)
    assert (width, devices, actual_batch) == (32, 2, batch)
    np.testing.assert_array_equal(actual_windows, windows)
    np.testing.assert_array_equal(actual_rates, rates)
    assert 2*np.sum(actual_rates*(actual_windows-1)) == flank


@pytest.mark.parametrize("flank,windows,rates,batch", SCHEDULES)
def test_inference_loaders_match_complete_checkpoint(tmp_path, flank, windows, rates, batch):
    import torch
    from openspliceai.predict.predict import load_pytorch_models as predict
    from openspliceai.variant.utils import load_pytorch_models as variant
    reference = SpliceAI(32, np.asarray(windows), np.asarray(rates)).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(reference.state_dict(), path)
    predicted, params = predict(str(path), torch.device("cpu"), 5000, flank)
    assert params["CL"] == flank and params["BATCH_SIZE"] == batch
    for model in predicted + variant(str(path), flank):
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.state_dict()[name])


def test_configurations_are_independent_arrays():
    first = model_hyperparameters(80)
    first[2][0] = 99
    assert model_hyperparameters(80)[2][0] == 11


@pytest.mark.parametrize("flank", [0, 81, 999, -80])
def test_unsupported_context_is_rejected(flank):
    with pytest.raises(ValueError, match="Unsupported flanking"):
        model_hyperparameters(flank)
