"""
Filename: visualization.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Functions for visualizing calibration results.
"""

import matplotlib.pyplot as plt
import numpy as np
from openspliceai.calibrate.calibrate_utils import *
import torch

def plot_score_distribution(probs, probs_scaled, labels, output_dir, index):
    """Save bounded before/after class-score histograms split by true class."""
    index_names = {
        0: ('non-splice site', 'neither'),
        1: ('acceptor site', 'acceptor'),
        2: ('donor site', 'donor')
    }
    
    positive_indices = (labels == index)
    probs_positive = probs[:, index][positive_indices]
    probs_scaled_positive = probs_scaled[:, index][positive_indices]

    plt.figure(figsize=(5.5, 4))
    plt.hist(probs_positive, bins=30, alpha=0.5, label='Original')
    plt.hist(probs_scaled_positive, bins=30, alpha=0.5, label='Calibrated')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.title(f'Score Distribution for {index_names[index][0]}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prob_dist_{index_names[index][1]}.png")
    plt.close()


def score_frequency_distribution(probs, probs_scaled, labels, outdir, index=1):
    # Flatten the tensors for simplicity
    """Bin class probabilities and plot their empirical frequency distribution."""
    probabilities_flat = probs[:, index]
    probabilities_scaled_flat = probs_scaled[:, index]
    print("probabilities_flat.shape: ", probabilities_flat.shape)
    print("probabilities_scaled_flat.shape: ", probabilities_scaled_flat.shape)
    print("labels.shape: ", labels.shape)
    # Filter out cases where the label is zero (i.e., keep only positive cases for Class 1)
    positive_indices = (labels == index)  # True for positive cases
    print("positive_indices: ", positive_indices)
    print("len(positive_indices): ", len(positive_indices))
    probabilities_positive = probabilities_flat[positive_indices]
    probabilities_scaled_positive = probabilities_scaled_flat[positive_indices]
    labels_positive = labels[positive_indices]
    print("len(probabilities_positive): ", len(probabilities_positive))
    print("probabilities_positive: ", probabilities_positive)
    print("len(probabilities_scaled_positive): ", len(probabilities_scaled_positive))
    print("probabilities_scaled_positive: ", probabilities_scaled_positive)
    print("len(labels_positive): ", len(labels_positive))
    print("labels_positive: ", labels_positive)
    # Since we're now only looking at positive cases, the labels will be all ones,
    # and we're primarily interested in the distribution of predicted probabilities.
    # Plotting
    plt.figure(figsize=(5.5, 4))
    # Probability distribution for Class 1 from logits (only positive cases)
    plt.hist(probabilities_positive, bins=30, alpha=0.5, label='Predicted Probabilities')
    plt.hist(probabilities_scaled_positive, bins=30, alpha=0.5, label='Predicted Calibrated Probabilities')
    # For positive cases, since all labels are 1, we can simply indicate the average or density with a line or annotation
    # plt.axvline(x=np.mean(labels_positive), color='r', linestyle='dashed', linewidth=2, label='Actual Labels')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    if index == 0:
        plt.title('Score Distribution for non-splice site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/prob_dist_neither.png")
    if index == 1:
        plt.title('Score Distribution for acceptor site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/prob_dist_acceptor.png")
    elif index == 2:
        plt.title('Score Distribution for donor site')
        plt.tight_layout()
        plt.savefig(f"{outdir}/prob_dist_donor.png")
    plt.clf()


def plot_calibration_curves(calibration_data, calibration_data_scaled, classes, output_dir):
    """Save before/after reliability curves for all three model classes."""
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    for i, (class_name, (orig_data, scaled_data)) in enumerate(zip(classes, zip(calibration_data, calibration_data_scaled))):
        prob_true, prob_pred, bin_counts = orig_data
        prob_true_s, prob_pred_s, bin_counts_s = scaled_data
        
        ci_lower, ci_upper = compute_confidence_intervals(prob_true, bin_counts)
        ci_lower_s, ci_upper_s = compute_confidence_intervals(prob_true_s, bin_counts_s)
        
        axs[i].plot(prob_pred, prob_true, 'o-', color='blue', label='Original')
        axs[i].fill_between(prob_pred, ci_lower, ci_upper, color='blue', alpha=0.1)
        axs[i].plot(prob_pred_s, prob_true_s, 'o-', color='green', label='Calibrated')
        axs[i].fill_between(prob_pred_s, ci_lower_s, ci_upper_s, color='green', alpha=0.1)
        axs[i].plot([0, 1], [0, 1], 'k--')
        axs[i].set_title(class_name)
        axs[i].set_xlabel('Predicted Probability')
        axs[i].set_ylabel('Empirical Probability')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curve.png", dpi=300)
    plt.close()


def plot_brier_scores(brier_uncal, brier_cal, classes, output_dir):
    """Save the original and calibrated per-class Brier score comparison."""
    plt.figure(figsize=(8, 6))
    x = np.arange(len(classes))
    plt.bar(x - 0.2, brier_uncal, 0.4, label='Uncalibrated')
    plt.bar(x + 0.2, brier_cal, 0.4, label='Calibrated')
    plt.xticks(x, classes)
    plt.ylabel('Brier Score')
    plt.title('Brier Score Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/brier_scores.png", dpi=300)
    plt.close()


def plot_calibration_map(scaled_model, device, output_dir):
    """Save the class-temperature probability transformation over a probability grid."""
    p1d = np.linspace(0, 1, 20)
    p0, p1 = np.meshgrid(p1d, p1d)
    p2 = 1 - p0 - p1
    mask = (p2 >= 0) & (p2 <= 1)
    p = np.vstack([p0[mask], p1[mask], p2[mask]]).T
    
    logits = reverse_softmax(p)
    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(torch.tensor(logits).float().to(device))
    probs_scaled = torch.nn.functional.softmax(logits_scaled, dim=1).cpu().numpy()

    plt.figure(figsize=(4.8, 4.8))
    colors = ['red', 'green', 'blue']
    for i in range(p.shape[0]):
        color = colors[np.argmax(p[i])]
        plt.arrow(p[i][1], p[i][2], 
                 probs_scaled[i][1]-p[i][1], probs_scaled[i][2]-p[i][2],
                 color=color, head_width=0.01, length_includes_head=True)
    
    plt.plot([0, 1], [0, 0], 'k-')
    plt.plot([1, 0], [0, 1], 'k-')
    plt.plot([0, 0], [1, 0], 'k-')
    plt.xlabel("Acceptor Probability")
    plt.ylabel("Donor Probability")
    plt.title("Calibration Map")
    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', label='Non-splice'),
        plt.Line2D([0], [0], color='green', label='Acceptor'),
        plt.Line2D([0], [0], color='blue', label='Donor')
    ])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_map.png", dpi=300)
    plt.close()