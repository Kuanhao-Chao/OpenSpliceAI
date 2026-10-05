"""Portable PyTorch checkpoint contracts shared by calibration and inference.

Raw state dictionaries remain supported. Version 1 calibrated checkpoints contain
only tensors and primitive metadata and load with ``weights_only=True``.
"""
from collections.abc import Mapping
from pathlib import Path
import os
import tempfile
import pickle

import torch
from torch import nn

CLASS_NAMES = ['non_splice', 'acceptor', 'donor']


class CheckpointError(ValueError):
    """A missing, corrupt or incompatible model artifact."""


def atomic_torch_save(value, destination):
    """Publish a complete checkpoint atomically; preserve existing output on failure."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+destination.name+'.', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def unpack_checkpoint(checkpoint, flanking_size):
    """Return ``(state_dict, temperature_or_none)`` after checking metadata.

    Raises ValueError for unsupported artifacts, empty weights, mismatched
    contexts, class orders, or nonfinite/out-of-range temperatures.
    """
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError('Checkpoint must contain a nonempty state dictionary')
    if 'format_version' not in checkpoint:
        if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                   for key, value in checkpoint.items()):
            raise ValueError('Expected a tensor state dictionary or a versioned checkpoint')
        if any(not torch.isfinite(value).all() for value in checkpoint.values()):
            raise ValueError('Checkpoint weights must be finite')
        return checkpoint, None
    if type(checkpoint['format_version']) is not int or checkpoint['format_version'] != 1:
        raise ValueError('Unsupported checkpoint format_version')
    if checkpoint.get('flanking_size') != int(flanking_size):
        raise ValueError('Checkpoint flanking_size does not match the requested context')
    if checkpoint.get('class_names') != CLASS_NAMES:
        raise ValueError('Checkpoint class order must be non_splice, acceptor, donor')
    state, unused = unpack_checkpoint(checkpoint.get('state_dict'), flanking_size)
    if unused is not None:
        raise ValueError('Nested calibrated checkpoints are not supported')
    temperature = validate_temperature(checkpoint.get('temperature'))
    return state, temperature


def load_checkpoint(path, device='cpu'):
    """Read a tensor-only checkpoint with a consistent library exception."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError, IndexError, TypeError) as exc:
        raise CheckpointError(f'Unable to load checkpoint at {path}: {exc}') from exc


def validate_temperature(temperature):
    """Validate the three class temperatures in the fitted range [0.05, 5]."""
    if not isinstance(temperature, torch.Tensor) or temperature.shape != (3,) or not temperature.is_floating_point():
        raise ValueError('Temperature must contain one value per model class (three values)')
    if not torch.isfinite(temperature).all() or (temperature < .05).any() or (temperature > 5).any():
        raise ValueError('Temperature values must be finite and between 0.05 and 5')
    return temperature.detach()


class CalibratedSpliceAI(nn.Module):
    """Return normalized (N, 3, L) probabilities from a logits model and temperatures."""

    def __init__(self, model, temperature):
        """Initialize CalibratedSpliceAI with the supplied model, data or runtime settings."""
        super().__init__()
        self.model = model
        self.model.apply_softmax = False
        self.register_buffer('temperature', validate_temperature(temperature).clone())

    @property
    def CL(self):
        """Return the base model context length."""
        return self.model.CL

    def forward(self, inputs):
        """Apply the model to the supplied channel-first inputs."""
        return torch.softmax(self.model(inputs) / self.temperature.view(1, 3, 1), dim=1)


def calibrated_checkpoint(model, temperature, flanking_size):
    """Build a CPU-portable version 1 artifact without pickling a model object."""
    from openspliceai.header import __version__
    if flanking_size not in (80, 400, 2000, 10000) or getattr(model, 'CL', None) != flanking_size or isinstance(model, CalibratedSpliceAI):
        raise ValueError('Calibrated artifacts require a context-matched unwrapped base model')
    return {
        'format_version': 1,
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'flanking_size': int(flanking_size),
        'temperature': validate_temperature(temperature).cpu(),
        'class_names': CLASS_NAMES.copy(),
        'openspliceai_version': __version__,
    }
