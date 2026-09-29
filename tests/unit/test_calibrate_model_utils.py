"""Calibration loads every parameter of a checkpoint with the requested context."""
import numpy as np
import pytest
import torch

import openspliceai.calibrate.model_utils as cmu
from openspliceai.train_base.openspliceai import SpliceAI


@pytest.mark.parametrize("flank,windows,rates", [
    (80, [11]*4, [1]*4),
    (400, [11]*8, [1]*4 + [4]*4),
    (2000, [11]*8 + [21]*4, [1]*4 + [4]*4 + [10]*4),
    (10000, [11]*8 + [21]*4 + [41]*4, [1]*4 + [4]*4 + [10]*4 + [25]*4),
])
def test_initialize_model_and_optim_builds_each_flanking(tmp_path, flank, windows, rates):
    expected = SpliceAI(32, np.asarray(windows), np.asarray(rates), apply_softmax=False)
    checkpoint = tmp_path / "model.pt"
    torch.save(expected.state_dict(), checkpoint)
    model, params = cmu.initialize_model_and_optim(torch.device("cpu"), flank, checkpoint)
    assert params["CL"] == flank and params["SL"] == 5000 and params["L"] == 32
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])


def test_calibration_rejects_unsupported_context(packaged_80nt_state):
    with pytest.raises(ValueError, match="Unsupported flanking"):
        cmu.initialize_model_and_optim(torch.device("cpu"), 81, packaged_80nt_state)
