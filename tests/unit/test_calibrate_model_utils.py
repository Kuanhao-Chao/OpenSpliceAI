"""Unit coverage for openspliceai/calibrate/model_utils.py (the calibrate model builder,
a 5th copy of the hyperparameter table; the table-sync is pinned separately)."""
import pytest
import torch

import openspliceai.calibrate.model_utils as cmu
from openspliceai.train_base.openspliceai import SpliceAI


@pytest.mark.parametrize("flank", [80, 400, 2000, 10000])
def test_initialize_model_and_optim_builds_each_flanking(packaged_80nt_state, flank):
    """Builds the correct architecture for every flanking size; the 80nt checkpoint loads
    with strict=False so larger sizes just keep the matching (or zero) keys."""
    model, params = cmu.initialize_model_and_optim(torch.device("cpu"), flank, packaged_80nt_state)
    assert isinstance(model, SpliceAI)
    assert params["CL"] == flank and params["SL"] == 5000 and params["L"] == 32
