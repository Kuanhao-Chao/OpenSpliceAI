"""Coverage for the temperature-scaling optimisation loop + loaders in
openspliceai/calibrate/temperature_scaling.py (the existing test covers the wrapper's
scale/clamp/ECE; this drives set_temperature/load_temperature over a real model+loader)."""
import h5py
import numpy as np
import pytest
import torch

import openspliceai.calibrate.model_utils as cmu
import openspliceai.calibrate.temperature_scaling as ts


@pytest.fixture
def logits_model(packaged_80nt_state):
    """A no-softmax SpliceAI (logits) + params, as the calibrate CLI builds it."""
    return cmu.initialize_model_and_optim(torch.device("cpu"), 80, packaged_80nt_state)


@pytest.fixture
def valid_loader(calibrate_datasets):
    with h5py.File(calibrate_datasets["validation"], "r") as h5f:
        yield ts.get_validation_loader(h5f, [0], batch_size=2)


def test_get_validation_loader_builds_batches(valid_loader):
    batches = list(valid_loader)
    assert batches and batches[0][0].shape[1] == 4   # (N, channels=4, SL+CL_max)


def test_set_temperature_optimises_within_clamp(logits_model, valid_loader):
    model, params = logits_model
    mwt = ts.ModelWithTemperature(model, 3)
    mwt.set_temperature(valid_loader, params)
    t = mwt.temperature.detach()
    assert torch.isfinite(t).all()
    assert float(t.min()) >= 0.05 - 1e-6 and float(t.max()) <= 5.0 + 1e-6
    assert hasattr(mwt, "logits") and hasattr(mwt, "labels")


def test_load_temperature_restores_and_collects(logits_model, valid_loader, tmp_path):
    model, params = logits_model
    src = ts.ModelWithTemperature(model, 3)
    f = tmp_path / "temperature.pt"
    src.save_temperature(str(f))
    dst = ts.ModelWithTemperature(model, 3)
    dst.load_temperature(str(f), valid_loader, params)
    assert dst.logits.ndim == 2 and dst.logits.shape[1] == 3
    assert dst.labels.ndim == 1


def test_compute_ece_nll_returns_finite(logits_model):
    model, _ = logits_model
    mwt = ts.ModelWithTemperature(model, 3)
    nll, ece = mwt.compute_ece_nll(torch.randn(40, 3), torch.randint(0, 3, (40,)))
    assert np.isfinite(nll) and np.isfinite(ece) and ece >= 0.0
