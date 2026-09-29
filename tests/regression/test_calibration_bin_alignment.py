"""Reliability-curve counts must describe the same occupied bins as the means."""
import numpy as np
import pytest

from openspliceai.calibrate.calibrate_utils import compute_calibration_curve


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_endpoint_observations_are_conserved(strategy):
    true, pred, counts = compute_calibration_curve(
        [0, 1, 1, 1], [0., 0., 1., 1.], n_bins=10, strategy=strategy
    )
    np.testing.assert_array_equal(true, [.5, 1.])
    np.testing.assert_array_equal(pred, [0., 1.])
    np.testing.assert_array_equal(counts, [2, 2])


def test_uniform_edges_belong_to_lower_bin():
    true, pred, counts = compute_calibration_curve(
        [0, 1, 0, 1], [0., .25, .5, 1.], n_bins=4, strategy="uniform"
    )
    np.testing.assert_array_equal(true, [.5, 0., 1.])
    np.testing.assert_array_equal(pred, [.125, .5, 1.])
    np.testing.assert_array_equal(counts, [2, 1, 1])


def test_repeated_quantiles_produce_one_occupied_bin():
    true, pred, counts = compute_calibration_curve(
        [0, 1, 0, 1], [.5, .5, .5, .5], n_bins=10, strategy="quantile"
    )
    np.testing.assert_array_equal(true, [.5])
    np.testing.assert_array_equal(pred, [.5])
    np.testing.assert_array_equal(counts, [4])


def test_plot_uses_independent_counts_for_calibrated_curve(tmp_path, monkeypatch):
    from openspliceai.calibrate import visualization
    from openspliceai.calibrate.calibrate_utils import compute_confidence_intervals
    seen = []

    def record(prob_true, counts):
        seen.append(np.asarray(counts).tolist())
        return compute_confidence_intervals(prob_true, counts)

    monkeypatch.setattr(visualization, "compute_confidence_intervals", record)
    original = (np.array([.25, .75]), np.array([.2, .8]), np.array([3, 7]))
    scaled = (np.array([.5]), np.array([.5]), np.array([10]))
    visualization.plot_calibration_curves([original]*3, [scaled]*3,
                                          ["Non-splice", "Acceptor", "Donor"], str(tmp_path))
    assert seen == [[3, 7], [10]]*3


def test_evaluation_saves_calibrated_counts_from_calibrated_predictions(tmp_path, monkeypatch):
    import torch
    from openspliceai.calibrate import calibrate
    # Independent curves have different numbers of occupied bins after scaling.
    curves = [(np.array([0., 1.]), np.array([0., 1.]), np.array([2, 2])),
              (np.array([.5]), np.array([.5]), np.array([4]))]*3
    iterator = iter(curves)
    monkeypatch.setattr(calibrate, "compute_calibration_curve", lambda *a, **k: next(iterator))
    for name in ("plot_score_distribution", "plot_calibration_curves", "plot_brier_scores", "plot_calibration_map"):
        monkeypatch.setattr(calibrate, name, lambda *a, **k: None)
    saved = []
    monkeypatch.setattr(calibrate, "save_calibration_data", lambda *a: saved.append((a[-1], a[-2])))

    class Base(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros(2, 3, 2)

    class Calibrated:
        model = Base()
        def compute_ece_nll(self, *args):
            return 0., 0.

        def temperature_scale(self, logits):
            return logits


    loader = [(torch.zeros(2, 4, 10082), torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]]*2))]
    calibrate.evaluate_and_visualize(Calibrated(), loader, torch.device("cpu"), str(tmp_path),
                                     "test", {"CL": 80, "N_GPUS": 2}, 80)
    assert [suffix for suffix, _ in saved] == ["original", "calibrated"]*3
    for (_, original), (_, scaled) in zip(saved[::2], saved[1::2]):
        np.testing.assert_array_equal(original, [2, 2])
        np.testing.assert_array_equal(scaled, [4])
