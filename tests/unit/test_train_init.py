"""Unit tests for openspliceai/train/train.py model/optimizer setup and the
top-k statistics helper in openspliceai/train_base/utils.py.

Every expected value below was grounded by running the function on real inputs
(the (CL, BATCH_SIZE) table, ResidualUnit counts, and the print_topl_statistics
return shape were all observed before being asserted).
"""
import numpy as np
import pytest
import torch
import torch.optim as optim

import openspliceai.train_base.utils as tbu
from openspliceai.train import train
from openspliceai.train_base.openspliceai import ResidualUnit, Skip


# flanking_size -> (n ResidualUnit modules, BATCH_SIZE) from the duplicated hyperparam table.
FLANK_TABLE = {
    80: (4, 36),
    400: (8, 36),
    2000: (12, 24),
    10000: (16, 12),
}


@pytest.mark.parametrize("flanking_size", [80, 400, 2000, 10000])
def test_initialize_model_and_optim_table(flanking_size):
    device = torch.device("cpu")
    expected_units, expected_batch = FLANK_TABLE[flanking_size]

    model, optimizer, scheduler, params = train.initialize_model_and_optim(
        device, flanking_size, epochs=10, scheduler="MultiStepLR"
    )

    # Context length must equal the flanking size exactly (CL = 2*sum(AR*(W-1))).
    assert params["CL"] == flanking_size
    # The SL output window is the fixed constant from constants.py.
    assert params["SL"] == 5000
    assert params["BATCH_SIZE"] == expected_batch

    # residual_units holds ResidualUnit AND Skip modules; count only the real units.
    n_residual = sum(1 for m in model.residual_units if isinstance(m, ResidualUnit))
    assert n_residual == expected_units
    # A Skip is inserted after every 4 residual units.
    n_skip = sum(1 for m in model.residual_units if isinstance(m, Skip))
    assert n_skip == expected_units // 4

    # The model's own CL attribute agrees with the returned params.
    assert int(model.CL) == flanking_size

    # Optimizer is AdamW at the train learning rate.
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_initialize_model_and_optim_multisteplr():
    device = torch.device("cpu")
    _, _, scheduler, _ = train.initialize_model_and_optim(
        device, 80, epochs=10, scheduler="MultiStepLR"
    )
    assert isinstance(scheduler, optim.lr_scheduler.MultiStepLR)


def test_initialize_model_and_optim_cosine():
    device = torch.device("cpu")
    _, _, scheduler, _ = train.initialize_model_and_optim(
        device, 80, epochs=10, scheduler="CosineAnnealingWarmRestarts"
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


def test_initialize_model_forward_runs():
    """Sanity: the 80nt model runs a forward pass producing 3 softmax channels.

    The model crops only by its own CL (=flanking size); the training loop first clips the
    raw (SL+CL_max) window down to (SL+CL) via clip_datapoints, so feeding SL+CL here yields
    the SL-length output the model actually trains on.
    """
    device = torch.device("cpu")
    model, _, _, params = train.initialize_model_and_optim(
        device, 80, epochs=10, scheduler="MultiStepLR"
    )
    model.eval()
    from openspliceai.constants import SL
    x = torch.zeros(2, 4, SL + params["CL"])
    x[:, 0, :] = 1.0  # all-A one-hot
    with torch.no_grad():
        out = model(x)
    # (N, 3, SL) softmax over channel dim.
    assert out.shape == (2, 3, SL)
    assert torch.allclose(out.sum(dim=1), torch.ones(2, SL), atol=1e-4)


# --- print_topl_statistics ---------------------------------------------------------

def _make_topl_inputs(n=1000, n_true=20, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.zeros(n, dtype=np.int64)
    true_idx = rng.choice(n, size=n_true, replace=False)
    y_true[true_idx] = 1
    y_pred = rng.random(n).astype(np.float64)
    # Make predictions informative so AUPRC > baseline.
    y_pred[true_idx] += 0.5
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return y_true, y_pred, true_idx


def test_print_topl_statistics_return_types_and_ranges(tmp_path):
    y_true, y_pred, _ = _make_topl_inputs()
    out_file = tmp_path / "topk.txt"

    topk, auprc = tbu.print_topl_statistics(
        y_true, y_pred, str(out_file), ss_type="acceptor", print_top_k=False
    )

    # Return is (top-1L accuracy, AUPRC). Both are probabilities in [0, 1].
    assert isinstance(topk, float)
    assert 0.0 <= topk <= 1.0
    assert 0.0 <= float(auprc) <= 1.0
    # Informative predictions => AUPRC strictly above the prevalence baseline (~0.02).
    assert float(auprc) > 0.02


def test_print_topl_statistics_appends_one_line_per_call(tmp_path):
    y_true, y_pred, true_idx = _make_topl_inputs()
    out_file = tmp_path / "topk.txt"

    tbu.print_topl_statistics(y_true, y_pred, str(out_file), ss_type="acceptor")
    assert out_file.exists()
    lines = out_file.read_text().splitlines()
    assert len(lines) == 1

    # Calling again appends rather than truncating.
    tbu.print_topl_statistics(y_true, y_pred, str(out_file), ss_type="donor")
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2

    # Each line is 10 tab-separated fields; the last is the count of true positives.
    fields = lines[0].split("\t")
    assert len(fields) == 10
    assert int(fields[-1]) == len(true_idx)
