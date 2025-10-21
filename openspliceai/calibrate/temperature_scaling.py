"""
Filename: temperature_scaling.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Temperature scaling functions and classes for model calibration.
"""

import inspect

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from openspliceai.train_base.utils import *
from torch.utils.data import TensorDataset, DataLoader


def load_data_from_shard(h5f, shard_idx):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    return X, Y


def get_validation_loader(h5f, idxs, batch_size):
    """
    Create a DataLoader for the validation data.
    """
    X_list, Y_list = [], []
    for shard_idx in idxs:
        X, Y = load_data_from_shard(h5f, shard_idx)
        X_list.append(X)
        Y_list.append(Y)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print("X:", X.shape)
    print("Y:", Y.shape)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


class ModelWithTemperature(nn.Module):
    """
    Wraps a model with class-wise temperature scaling for better calibration.
    """
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        # Each class gets its own temperature parameter
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.0)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Class-wise temperature scaling: each logit for class c is divided by
        self.temperature[c]. If logits is [N, C], we broadcast over the batch dimension.
        """
        temperature = torch.clamp(self.temperature, min=0.05, max=5.0)
        return logits / temperature

    def save_temperature(self, filepath):
        """
        Save the temperature vector to a file.
        """
        torch.save(self.temperature.detach().cpu(), filepath)
        print(f"Temperature vector saved to {filepath}")

    def load_temperature(self, filepath, valid_loader, params):
        """
        Load the temperature parameter (vector) from a file, then compute and store
        logits/labels on the validation loader.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temperature = torch.load(filepath, map_location=device)
        self.temperature = nn.Parameter(temperature.to(device))
        self.temperature.data = torch.clamp(self.temperature.data, min=0.05, max=5.0)
        print(f"Loaded temperature vector: {self.temperature.data.cpu().numpy()}")

        # Collect logits and labels
        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                input, label = input.to(device), label.to(device)
                input, label = clip_datapoints(input, label, params["CL"], CL_max, params["N_GPUS"])
                logits = self.model(input)
                logits_list.append(logits.detach().cpu())
                labels_list.append(label.detach().cpu())

        logits = torch.cat(logits_list).permute(0, 2, 1).contiguous()
        labels = torch.cat(labels_list).permute(0, 2, 1).contiguous().argmax(dim=-1)

        # Flatten
        N, L, C = logits.shape
        logits = logits.view(-1, C).to(device)
        labels = labels.view(-1).long().to(device)

        self.logits, self.labels = logits, labels

    def set_temperature(self, valid_loader, params):
        """
        Tune the vector of temperature parameters using the validation set.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.model.eval()

        # Collect logits and labels
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = clip_datapoints(inputs, labels, params["CL"], CL_max, params["N_GPUS"])
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list).permute(0, 2, 1).contiguous()
        labels = torch.cat(labels_list).permute(0, 2, 1).contiguous().argmax(dim=-1)

        # Flatten
        N, L, C = logits.shape
        logits = logits.view(-1, C)
        labels = labels.view(-1).long()

        self.logits, self.labels = logits.to(device), labels.to(device)

        # Define losses
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # Print metrics before calibration
        self._compute_and_log_metrics(nll_criterion, ece_criterion, 'Before temperature scaling')

        # Optimize the temperature vector
        optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        scheduler_kwargs = dict(mode='min', factor=0.1, patience=2)
        scheduler_signature = inspect.signature(
            torch.optim.lr_scheduler.ReduceLROnPlateau.__init__
        )
        if 'verbose' in scheduler_signature.parameters:
            scheduler_kwargs['verbose'] = True
        else:
            print(
                "ReduceLROnPlateau 'verbose' argument unsupported in this torch version; proceeding without it."
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_kwargs
        )

        best_loss = float('inf')
        best_temp = self.temperature.data.clone()
        patience_counter = 0
        max_epochs = 2000
        min_delta = 1e-6
        patience = 2

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(self.logits), self.labels)
            ece_loss = ece_criterion(self.temperature_scale(self.logits), self.labels)
            loss.backward()
            optimizer.step()
            self.temperature.data = torch.clamp(self.temperature.data, min=0.05, max=5.0)
            ece_loss = ece_loss.item()
            current_loss = loss.item()
            scheduler.step(current_loss)
            # Early stopping logic
            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                best_temp = self.temperature.data.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {current_loss:.6f}, ECE: {ece_loss:.6f}",             
                  f"Temperature: {self.temperature.data.cpu().numpy()}")
            
            if patience_counter >= patience:
                print("Early stopping due to no improvement in loss.")
                break

        # Restore best temperature
        self.temperature.data = best_temp.to(device)
        print(f"Optimized temperature vector: {self.temperature.data.cpu().numpy()}")

        # Print metrics after calibration
        self._compute_and_log_metrics(nll_criterion, ece_criterion, 'After temperature scaling')

    def _compute_and_log_metrics(self, nll_criterion, ece_criterion, phase):
        logits_scaled = self.temperature_scale(self.logits)
        nll = nll_criterion(logits_scaled, self.labels).item()
        ece = ece_criterion(logits_scaled, self.labels).item()
        print(f'{phase} - NLL: {nll:.4f}, ECE: {ece:.4f}')

    def compute_ece_nll(self, logits, labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)
        nll = nll_criterion(logits.to(device), labels.to(device)).item()
        ece = ece_criterion(logits.to(device), labels.to(device)).item()
        return nll, ece


class _ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) Loss.
    """
    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
