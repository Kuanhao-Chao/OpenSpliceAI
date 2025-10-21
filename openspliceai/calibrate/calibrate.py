"""
Filename: calibrate.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Calibrate the OpenSpliceAI model.
"""

import os
import time
import torch
from torch.nn import functional as F
from openspliceai.train_base.utils import *
from openspliceai.calibrate.calibrate_utils import *
from openspliceai.calibrate.visualization import *
from openspliceai.calibrate.temperature_scaling import *
from openspliceai.calibrate.model_utils import initialize_model_and_optim

def get_logits_labels(model, loader, device, params):
    """
    Same as before.
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Collecting logits and labels'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = clip_datapoints(inputs, labels, params["CL"], CL_max, params["N_GPUS"])
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list).permute(0, 2, 1).contiguous()
    labels = torch.cat(labels_list).permute(0, 2, 1).contiguous().argmax(dim=-1)

    # Flatten logits and labels
    N, L, C = logits.shape
    logits = logits.view(-1, C)
    labels = labels.view(-1).long()
    return logits, labels


def evaluate_and_visualize(calibrated_model, data_loader, device, output_base_dir, dataset_name, params, flanking_size):
    print(f"\n--- Evaluating on {dataset_name} set ---")
    
    results_dir = os.path.join(output_base_dir, "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    calib_data_dir = os.path.join(results_dir, "calibration_data")
    os.makedirs(calib_data_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Compute logits and labels using the base model
    base_model = calibrated_model.model
    logits, labels = get_logits_labels(base_model, data_loader, device, params)

    # Apply temperature scaling
    logits_scaled = calibrated_model.temperature_scale(logits)

    # Compute metrics
    metric_original_file = os.path.join(results_dir, "metrics_original.txt")
    metric_calibrated_file = os.path.join(results_dir, "metrics_calibrated.txt")
    
    original_nll, original_ece = calibrated_model.compute_ece_nll(logits, labels)
    with open(metric_original_file, 'w') as f:
        f.write(f"Original_NLL\tOriginal_ECE\n")
        f.write(f"{original_nll:.8f}\t{original_ece:.8f}\n")
    print(f"Original NLL: {original_nll:.8f}")
    print(f"Original ECE: {original_ece:.8f}")

    calibrated_nll, calibrated_ece = calibrated_model.compute_ece_nll(logits_scaled, labels)
    with open(metric_calibrated_file, 'w') as f:
        f.write(f"Calibrated_NLL\tCalibrated_ECE\n")
        f.write(f"{calibrated_nll:.8f}\t{calibrated_ece:.8f}\n")
    print(f"Calibrated NLL: {calibrated_nll:.8f}")
    print(f"Calibrated ECE: {calibrated_ece:.8f}")

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    probs_scaled = F.softmax(logits_scaled, dim=1).detach().cpu().numpy()
    labels = labels.cpu().numpy()

    # Plotting, calibration curve computations, etc. remain the same
    for idx in [0, 1, 2]:
        plot_score_distribution(probs, probs_scaled, labels, plots_dir, idx)
    
    classes = ["Non-splice site", "Acceptor site", "Donor site"]
    calibration_data = []
    calibration_data_scaled = []
    for i in range(3):
        class_labels = (labels == i).astype(int)
        class_probs = probs[:, i]
        class_probs_scaled = probs_scaled[:, i]
        
        prob_true, prob_pred, bin_counts = compute_calibration_curve(
            class_labels, class_probs, n_bins=30, strategy='uniform')
        prob_true_scaled, prob_pred_scaled, _ = compute_calibration_curve(
            class_labels, class_probs_scaled, n_bins=30, strategy='uniform')
        
        calibration_data.append((prob_true, prob_pred, bin_counts))
        calibration_data_scaled.append((prob_true_scaled, prob_pred_scaled, bin_counts))
        save_calibration_data(calib_data_dir, classes[i], flanking_size, prob_true, prob_pred, bin_counts, 'original')
        save_calibration_data(calib_data_dir, classes[i], flanking_size, prob_true_scaled, prob_pred_scaled, bin_counts, 'calibrated')

    plot_calibration_curves(calibration_data, calibration_data_scaled, classes, plots_dir)
    brier_uncal, brier_cal = calculate_brier_scores(labels, probs, probs_scaled)
    plot_brier_scores(brier_uncal, brier_cal, classes, plots_dir)
    plot_calibration_map(calibrated_model, device, plots_dir)
    print(f"Results for {dataset_name} set saved to: {results_dir}")
    print("===============================================")


def calibrate(args):
    print("Running OpenSpliceAI with 'calibrate' mode")
    start_time = time.time()
    
    # Create the main output directory structure
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    calibration_output_dir = os.path.join(base_output_dir, "calibration")
    os.makedirs(calibration_output_dir, exist_ok=True)
    
    # Set up the device, datasets, and indices
    device = setup_environment(args)
    train_h5f, valid_h5f, test_h5f, _ = load_datasets(args)
    _, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)
    
    # Initialize the model
    model, model_params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)

    # -----------------------------
    # NEW: Provide num_classes here
    # (for example 3, if you know you have 3 classes)
    # If it's variable, you can determine dynamically from the model output dimension.
    num_classes = 3

    calibrated_model = ModelWithTemperature(model, num_classes=num_classes)
    print("Initialized calibrated model:", calibrated_model)    
    print("Validation indices count:", len(val_idxs))
    print("Test indices count:", len(test_idxs))
    
    # Create data loaders for the validation (calibration) and test sets
    validation_loader = get_validation_loader(valid_h5f, val_idxs, model_params["BATCH_SIZE"])
    test_loader = get_validation_loader(test_h5f, test_idxs, model_params["BATCH_SIZE"])
    
    # Load or determine the temperature vector
    if args.temperature_file:
        calibrated_model.load_temperature(args.temperature_file, validation_loader, model_params)
        # Because we have a vector, show the full array
        print(f"Loaded temperature from {args.temperature_file}: {calibrated_model.temperature.data.cpu().numpy()}")
    else:
        # Calibrate
        calibrated_model.set_temperature(validation_loader, model_params)
        temperature_save_path = os.path.join(base_output_dir, "temperature.pt")
        calibrated_model.save_temperature(temperature_save_path)
        print(f"Saved calibrated temperature to {temperature_save_path}")
    
    # Save the temperature vector in text form
    temperature_txt_save_path = os.path.join(base_output_dir, "temperature.txt")
    with open(temperature_txt_save_path, 'w') as f:
        f.write(str(calibrated_model.temperature.data.cpu().numpy()))
    
    # Save the full calibrated model (including the temperature vector)
    model_save_path = os.path.join(base_output_dir, "calibrated_model.pt")
    torch.save(calibrated_model, model_save_path)
    print(f"Calibrated model saved to: {model_save_path}")
    
    # Evaluate and visualize on the validation set
    evaluate_and_visualize(calibrated_model, validation_loader, device, calibration_output_dir, "validation", model_params, args.flanking_size)
    
    # Evaluate and visualize on the test set
    evaluate_and_visualize(calibrated_model, test_loader, device, calibration_output_dir, "test", model_params, args.flanking_size)
    
    end_time = time.time()
    print(f"\nTotal calibration and evaluation time: {end_time - start_time:.2f} seconds")
