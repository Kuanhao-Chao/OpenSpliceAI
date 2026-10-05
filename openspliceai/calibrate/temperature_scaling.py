"""
Filename: temperature_scaling.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Temperature scaling functions and classes for model calibration.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from openspliceai.train_base.utils import *
from torch.utils.data import DataLoader


def load_data_from_shard(h5f, shard_idx):
    """Read one legacy calibration shard into channel-first NumPy arrays."""
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    return X, Y


def get_validation_loader(h5f, idxs, batch_size):
    """Iterate an open HDF5 split without concatenating its shards in memory."""
    from openspliceai.calibrate.streaming import ShardDataset
    if batch_size < 1:
        raise ValueError('Calibration batch size must be positive')
    return DataLoader(ShardDataset(h5f, idxs), batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=0)


class ModelWithTemperature(nn.Module):
    """Class-wise logits scaling; fitting uses a temporary disk-backed cache.

    ``logits``/``labels`` are bounded diagnostic previews, not the fitted split.
    ``observation_count`` and ``history`` describe the complete optimization.
    """
    def __init__(self, model, num_classes):
        """Initialize ModelWithTemperature with the supplied model, data or runtime settings."""
        super().__init__()
        if num_classes != 3:
            raise ValueError('OpenSpliceAI calibration requires three classes')
        self.model = model
        device = next(model.parameters(), torch.empty(0)).device
        self.temperature = nn.Parameter(torch.ones(num_classes, device=device))
        self.history = []

    def forward(self, inputs):
        """Apply the model to the supplied channel-first inputs."""
        return self.temperature_scale(self.model(inputs))

    def temperature_scale(self, logits):
        """Divide the class axis of (observations, 3) or (batch, 3, positions) logits."""
        if logits.ndim not in (2, 3) or logits.shape[1] != 3:
            raise ValueError('Logits must have three classes on axis 1')
        temperature = self.temperature.clamp(min=.05, max=5)
        return logits / (temperature.view(1, 3, 1) if logits.ndim == 3 else temperature)

    def save_temperature(self, filepath):
        """Validate and atomically save three CPU class temperatures."""
        from openspliceai.checkpoints import validate_temperature
        from openspliceai.checkpoints import atomic_torch_save
        atomic_torch_save(validate_temperature(self.temperature).cpu(), filepath)

    def load_temperature(self, filepath, valid_loader=None, params=None):
        """Restore a valid temperature; optionally collect a bounded diagnostic preview."""
        from openspliceai.checkpoints import validate_temperature
        from openspliceai.calibrate.streaming import LogitCache
        device = self.temperature.device
        value = validate_temperature(torch.load(filepath, map_location=device, weights_only=True))
        self.temperature = nn.Parameter(value.to(device))
        self.model.eval()
        if valid_loader is not None:
            with LogitCache(self.model, valid_loader, device, params) as cache:
                self.logits, self.labels = cache.preview()
                self.observation_count = len(cache)

    def _cache_metrics(self, cache):
        from openspliceai.calibrate.streaming import CalibrationStats
        stats = CalibrationStats()
        with torch.no_grad():
            for logits, labels in cache.batches(self.temperature.device):
                stats.update(self.temperature_scale(logits), labels)
        return stats.nll, stats.ece

    def fit_cache(self, cache, epochs=10, early_stopping=False, patience=2):
        """Optimize full-split NLL with one accumulated-gradient update per epoch.

        Each checkpoint is scored after its update. The initial temperature is
        included in best-state selection, so fitting cannot select a worse NLL.
        """
        if epochs < 1 or patience < 1:
            raise ValueError('Calibration epochs and patience must be positive')
        self.model.eval()
        self.observation_count = len(cache)
        self.logits, self.labels = cache.preview()
        optimizer = torch.optim.Adam([self.temperature], lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1,
                                                              patience=patience)
        best_loss, before_ece = self._cache_metrics(cache)
        best_temp = self.temperature.detach().clone()
        self.history = [{'epoch': 0, 'nll': best_loss, 'ece': before_ece,
                         'temperature': best_temp.cpu().tolist()}]
        no_improvement = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            for logits, labels in cache.batches(self.temperature.device):
                loss = F.cross_entropy(self.temperature_scale(logits), labels, reduction='sum') / len(cache)
                loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.temperature.clamp_(.05, 5)
            current_loss, current_ece = self._cache_metrics(cache)
            if not np.isfinite(current_loss):
                raise ValueError('Calibration produced a nonfinite objective')
            scheduler.step(current_loss)
            self.history.append({'epoch': epoch+1, 'nll': current_loss, 'ece': current_ece,
                                 'temperature': self.temperature.detach().cpu().tolist()})
            if best_loss-current_loss > 1e-6:
                best_loss, best_temp = current_loss, self.temperature.detach().clone()
                no_improvement = 0
            else:
                no_improvement += 1
            print(f'Calibration epoch {epoch+1}/{epochs}: NLL={current_loss:.8f}, ECE={current_ece:.8f}')
            if early_stopping and no_improvement >= patience:
                break
        with torch.no_grad():
            self.temperature.copy_(best_temp)
        return self

    def set_temperature(self, valid_loader, params, epochs=10, early_stopping=False, patience=2):
        """Cache logits once, fit temperatures, and remove temporary files on exit."""
        from openspliceai.calibrate.streaming import LogitCache
        with LogitCache(self.model, valid_loader, self.temperature.device, params) as cache:
            return self.fit_cache(cache, epochs, early_stopping, patience)

    def _compute_and_log_metrics(self, nll_criterion, ece_criterion, phase):
        """Print metrics of the bounded diagnostic preview (not full-split metrics)."""
        logits = self.temperature_scale(self.logits.to(self.temperature.device))
        labels = self.labels.to(self.temperature.device)
        print(f'{phase} (preview) - NLL: {nll_criterion(logits, labels).item():.4f}, '
              f'ECE: {ece_criterion(logits, labels).item():.4f}')

    def compute_ece_nll(self, logits, labels):
        """Compute NLL and ECE for an explicit small (observations, 3) tensor."""
        device = self.temperature.device
        return (F.cross_entropy(logits.to(device), labels.to(device)).item(),
                _ECELoss().to(device)(logits.to(device), labels.to(device)).item())


class _ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) Loss.
    """
    def __init__(self, n_bins=15):
        """Initialize _ECELoss with the supplied model, data or runtime settings."""
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        """Apply the model to the supplied channel-first inputs."""
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
