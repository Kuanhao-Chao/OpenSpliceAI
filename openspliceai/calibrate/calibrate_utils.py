"""
Filename: calibrate_utils.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Utility functions for calibrating models.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def compute_calibration_curve(labels, probs, n_bins=10, strategy='quantile'):
    """Return aligned occupied-bin fractions, probabilities and counts."""
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy=strategy)
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(probs, quantiles)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    # Match sklearn's interior-edge convention, including probabilities 0 and 1.
    # calibration_curve omits empty bins; counts must use the same occupied bins.
    bin_indices = np.searchsorted(bin_edges[1:-1], probs)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = bin_counts[bin_counts > 0]
    return prob_true, prob_pred, bin_counts


def compute_confidence_intervals(prob_true, bin_counts, z=1.96):
    """Return Wilson binomial intervals for the provided reliability bins."""
    if np.shape(prob_true) != np.shape(bin_counts):
        raise ValueError("Calibration probabilities and counts must have matching shapes")
    ci_lower, ci_upper = [], []
    for p, n in zip(prob_true, bin_counts):
        std_error = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        delta = z * std_error
        ci_lower.append(max(p - delta, 0))
        ci_upper.append(min(p + delta, 1))
    return np.array(ci_lower), np.array(ci_upper)


def reverse_softmax(probs):
    """Recover logits up to a shared additive constant from probabilities."""
    return np.log(np.clip(probs, 1e-8, 1.0))


def save_calibration_data(output_dir, class_name, flanking_size, prob_true, prob_pred, bin_counts, suffix):
    """Write aligned reliability statistics and confidence intervals as CSV."""
    np.savez(
        f"{output_dir}/calibration_data_{class_name}_{suffix}_{flanking_size}nt.npz",
        prob_true=prob_true,
        prob_pred=prob_pred,
        bin_counts=bin_counts
    )


def calculate_brier_scores(labels, probs, probs_scaled):
    """Return mean squared probability error independently for each class."""
    return (
        [brier_score_loss((labels == i).astype(int), probs[:, i]) for i in range(3)],
        [brier_score_loss((labels == i).astype(int), probs_scaled[:, i]) for i in range(3)]
    )