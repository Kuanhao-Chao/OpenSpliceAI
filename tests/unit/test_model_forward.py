"""Unit tests for the SpliceAI model definition (CPU, untrained)."""
import numpy as np
import torch

from openspliceai.train_base.openspliceai import SpliceAI, ResidualUnit, Skip


def _model(apply_softmax=True):
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    return SpliceAI(32, W, AR, apply_softmax=apply_softmax).to("cpu").eval()


def test_forward_shape_and_cropping():
    model = _model()
    # 80nt model crops CL=80 (40 each side); input width 280 -> output 200
    x = torch.randn(2, 4, 280)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, 200)


def test_forward_softmax_normalised():
    model = _model(apply_softmax=True)
    with torch.no_grad():
        y = model(torch.randn(2, 4, 280))
    sums = y.sum(dim=1)            # over the 3 class channels
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_forward_logits_when_softmax_disabled():
    model = _model(apply_softmax=False)
    with torch.no_grad():
        y = model(torch.randn(2, 4, 280))
    sums = y.sum(dim=1)
    # raw logits should not generally sum to 1 across classes
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_residual_units_interleave_skip_layers():
    """Locks the structure that Fix #2 depends on: Skip layers are interleaved after
    every 4 ResidualUnits, so residual_units[-1] is a Skip, NOT the last ResidualUnit."""
    model = _model()
    assert len(model.residual_units) == 5
    assert isinstance(model.residual_units[-1], Skip)
    res_idx = [i for i, m in enumerate(model.residual_units) if isinstance(m, ResidualUnit)]
    assert res_idx == [0, 1, 2, 3]
