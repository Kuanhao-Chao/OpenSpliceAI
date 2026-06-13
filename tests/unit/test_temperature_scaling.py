"""Unit tests for openspliceai/calibrate/temperature_scaling.py and calibrate/calibrate_utils.py.

All expected values were grounded by running the real functions on the synthetic inputs first.
These are CPU-only and use a tiny identity-style stand-in model where a full SpliceAI is unnecessary.
"""
import numpy as np
import torch

from openspliceai.calibrate.temperature_scaling import ModelWithTemperature, _ECELoss
from openspliceai.calibrate.calibrate_utils import compute_calibration_curve


class _Identity(torch.nn.Module):
    """Trivial inner model so we can construct ModelWithTemperature without SpliceAI."""

    def forward(self, x):
        return x


def test_temperature_is_parameter_of_shape_num_classes_initialised_to_ones():
    m = ModelWithTemperature(_Identity(), num_classes=3)
    assert isinstance(m.temperature, torch.nn.Parameter)
    assert tuple(m.temperature.shape) == (3,)
    assert torch.equal(m.temperature.data, torch.ones(3))


def test_temperature_scale_divides_logits_by_temperature_broadcast():
    m = ModelWithTemperature(_Identity(), num_classes=3)
    # Set a known temperature vector (all within the [0.05, 5.0] clamp range).
    m.temperature.data = torch.tensor([1.0, 2.0, 4.0])
    logits = torch.tensor([[2.0, 4.0, 8.0], [1.0, 1.0, 1.0]])
    out = m.temperature_scale(logits)
    expected = logits / torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(out, expected)


def test_temperature_scale_clamps_temperature_range():
    """temperature_scale clamps temperatures into [0.05, 5.0] before dividing."""
    m = ModelWithTemperature(_Identity(), num_classes=3)
    m.temperature.data = torch.tensor([0.001, 10.0, 1.0])  # out-of-range low/high
    logits = torch.ones(1, 3)
    out = m.temperature_scale(logits)
    # 1/0.05 = 20, 1/5.0 = 0.2, 1/1.0 = 1.0
    expected = torch.tensor([[1.0 / 0.05, 1.0 / 5.0, 1.0]])
    assert torch.allclose(out, expected)


def test_compute_ece_nll_returns_two_finite_nonneg_floats():
    m = ModelWithTemperature(_Identity(), num_classes=3)
    torch.manual_seed(0)
    logits = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    nll, ece = m.compute_ece_nll(logits, labels)
    assert isinstance(nll, float) and isinstance(ece, float)
    assert np.isfinite(nll) and np.isfinite(ece)
    assert nll >= 0.0
    assert ece >= 0.0


def test_ece_loss_returns_scalar_tensor_nonneg():
    torch.manual_seed(0)
    logits = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    ece = _ECELoss()(logits, labels)
    assert isinstance(ece, torch.Tensor)
    assert ece.numel() == 1
    assert ece.item() >= 0.0


def test_ece_loss_zero_for_perfectly_calibrated_confident_correct():
    """If the model is 100% confident and always correct, ECE collapses to 0."""
    # Hugely positive logit on the true class => softmax ~ 1.0 confidence, all correct.
    logits = torch.tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
    labels = torch.tensor([0, 1, 2])
    ece = _ECELoss()(logits, labels)
    assert ece.item() < 1e-4


def test_save_temperature_roundtrip(tmp_path):
    m = ModelWithTemperature(_Identity(), num_classes=3)
    m.temperature.data = torch.tensor([0.5, 1.5, 2.5])
    fp = tmp_path / "temperature.pt"
    m.save_temperature(str(fp))
    assert fp.exists()

    loaded = torch.load(str(fp))
    assert isinstance(loaded, torch.Tensor)
    assert tuple(loaded.shape) == (3,)
    assert torch.equal(loaded, torch.tensor([0.5, 1.5, 2.5]))


def test_compute_calibration_curve_uniform_shapes_and_bin_counts():
    rng = np.random.default_rng(0)
    n = 200
    probs = rng.random(n)
    labels = (rng.random(n) < probs).astype(int)

    prob_true, prob_pred, bin_counts = compute_calibration_curve(
        labels, probs, n_bins=5, strategy="uniform"
    )
    # sklearn drops empty bins, so prob_true/prob_pred have <= n_bins entries and match each other.
    assert prob_true.shape == prob_pred.shape
    assert prob_true.shape[0] <= 5
    # bin_counts always has exactly n_bins entries.
    assert bin_counts.shape == (5,)
    assert bin_counts.sum() <= len(probs)
    assert (bin_counts >= 0).all()


def test_compute_calibration_curve_quantile_strategy():
    rng = np.random.default_rng(1)
    n = 200
    probs = rng.random(n)
    labels = (rng.random(n) < probs).astype(int)

    prob_true, prob_pred, bin_counts = compute_calibration_curve(
        labels, probs, n_bins=5, strategy="quantile"
    )
    assert prob_true.shape == prob_pred.shape
    assert bin_counts.shape == (5,)
    assert bin_counts.sum() <= len(probs)
    # Calibration probabilities are valid probabilities.
    assert ((prob_true >= 0) & (prob_true <= 1)).all()
    assert ((prob_pred >= 0) & (prob_pred <= 1)).all()
