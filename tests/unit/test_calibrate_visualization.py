"""Smoke coverage for openspliceai/calibrate/visualization.py.

Plotting is verified at the "runs without error + writes a PNG" level (Agg backend), not
pixel content, per the coverage policy in tests/README.md."""
import numpy as np
import pytest
import torch

import openspliceai.calibrate.visualization as viz
from openspliceai.calibrate.temperature_scaling import ModelWithTemperature

CLASSES = ["Non-splice", "Acceptor", "Donor"]


def _probs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((n, 3))
    return p / p.sum(axis=1, keepdims=True)


def test_plot_score_distribution(tmp_path):
    p = _probs()
    labels = np.random.default_rng(1).integers(0, 3, 60)
    viz.plot_score_distribution(p, p, labels, str(tmp_path), 1)
    assert (tmp_path / "prob_dist_acceptor.png").exists()


@pytest.mark.parametrize("index,fname", [(0, "prob_dist_neither.png"),
                                         (1, "prob_dist_acceptor.png"),
                                         (2, "prob_dist_donor.png")])
def test_score_frequency_distribution(tmp_path, index, fname):
    p = _probs()
    labels = np.random.default_rng(2).integers(0, 3, 60)
    viz.score_frequency_distribution(p, p, labels, str(tmp_path), index)
    assert (tmp_path / fname).exists()


def test_plot_calibration_curves(tmp_path):
    b = 5
    cd = [(np.linspace(0.1, 0.9, b), np.linspace(0.1, 0.9, b), np.full(b, 10)) for _ in range(3)]
    viz.plot_calibration_curves(cd, cd, CLASSES, str(tmp_path))
    assert (tmp_path / "calibration_curve.png").exists()


def test_plot_brier_scores(tmp_path):
    viz.plot_brier_scores([0.1, 0.2, 0.3], [0.05, 0.1, 0.15], CLASSES, str(tmp_path))
    assert (tmp_path / "brier_scores.png").exists()


def test_plot_calibration_map(tmp_path, model_80nt):
    mwt = ModelWithTemperature(model_80nt, 3)
    viz.plot_calibration_map(mwt, torch.device("cpu"), str(tmp_path))
    assert (tmp_path / "calibration_map.png").exists()
