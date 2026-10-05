"""Bounded-memory calibration data, disk-backed logits, and exact metric sums."""
from bisect import bisect_right
from tempfile import TemporaryDirectory
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from openspliceai.data_schema import shard_indices, validate_shard, validate_encoding
from openspliceai.train_base.utils import clip_datapoints


class ShardDataset(Dataset):
    """Read one window at a time from an open legacy HDF5 split (no worker processes)."""

    def __init__(self, handle, indices):
        """Initialize ShardDataset with the supplied model, data or runtime settings."""
        self.handle = handle
        self.indices = list(indices)
        available = set(shard_indices(handle))
        self.ends = np.cumsum([validate_shard(handle, index) if index in available else
                               self._missing(index) for index in self.indices]).tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError('No calibration shards: the selected split is empty')

    @staticmethod
    def _missing(index):
        raise ValueError(f'Calibration shard {index} is missing')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, position):
        if position < 0 or position >= len(self):
            raise IndexError(position)
        shard = bisect_right(self.ends, position)
        offset = position - (self.ends[shard-1] if shard else 0)
        index = self.indices[shard]
        x = self.handle[f'X{index}'][offset].T.astype(np.float32)
        y = self.handle[f'Y{index}'][0, offset].T.astype(np.float32)
        validate_encoding(x.T, y.T)
        return torch.from_numpy(x), torch.from_numpy(y)


class LogitCache:
    """Temporary chunked logits/labels; closes and removes its files on every exit."""

    def __init__(self, model, loader, device, params, directory=None, chunk_rows=8192):
        """Initialize LogitCache with the supplied model, data or runtime settings."""
        if chunk_rows < 1:
            raise ValueError('Cache chunk size must be positive')
        self.model, self.loader, self.device, self.params = model, loader, device, params
        self.directory, self.chunk_rows = directory, chunk_rows
        self.folder = self.handle = None

    def __enter__(self):
        """Enter the resource scope and return this object."""
        self.folder = TemporaryDirectory(prefix='openspliceai-logits-', dir=self.directory)
        try:
            self.handle = h5py.File(Path(self.folder.name)/'logits.h5', 'w')
            logits = self.handle.create_dataset('logits', (0, 3), maxshape=(None, 3),
                                                chunks=(self.chunk_rows, 3), dtype='f4')
            labels = self.handle.create_dataset('labels', (0,), maxshape=(None,),
                                                chunks=(self.chunk_rows,), dtype='i8')
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in self.loader:
                    inputs, targets = clip_datapoints(inputs, targets, self.params['CL'],
                                                      10000, self.params.get('N_GPUS', 1))
                    validate_encoding(inputs.permute(0, 2, 1).cpu().numpy(),
                                      targets.permute(0, 2, 1).cpu().numpy())
                    values = self.model(inputs.to(self.device, dtype=torch.float32)).detach().cpu()
                    if values.shape != targets.shape or values.ndim != 3 or values.shape[1] != 3:
                        raise ValueError('Calibration logits and labels must match (batch, 3, positions)')
                    if not torch.isfinite(values).all() or not torch.isfinite(targets).all():
                        raise ValueError('Calibration logits and labels must be finite')
                    observed = targets.sum(dim=1).reshape(-1) > 0
                    values = values.permute(0, 2, 1).reshape(-1, 3)[observed].numpy()
                    truth = targets.argmax(dim=1).reshape(-1)[observed].numpy()
                    start, end = len(labels), len(labels)+len(truth)
                    logits.resize(end, axis=0)
                    labels.resize(end, axis=0)
                    logits[start:end], labels[start:end] = values, truth
            if len(labels) == 0:
                raise ValueError('No calibration observations: the selected split is empty')
            self.handle.flush()
            return self
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.handle['labels'])

    def batches(self, device='cpu'):
        """Yield contiguous (observations, 3) logits and integer class labels."""
        for start in range(0, len(self), self.chunk_rows):
            yield (torch.from_numpy(self.handle['logits'][start:start+self.chunk_rows]).to(device),
                   torch.from_numpy(self.handle['labels'][start:start+self.chunk_rows]).to(device))

    def preview(self, maximum=100000):
        """Bounded diagnostic attributes; fitting always uses the entire cache."""
        if maximum < 1:
            raise ValueError('Preview size must be positive')
        return (torch.from_numpy(self.handle['logits'][:maximum]),
                torch.from_numpy(self.handle['labels'][:maximum]))

    def close(self):
        """Close the owned HDF5 handle and remove temporary cache storage."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.folder is not None:
            self.folder.cleanup()
            self.folder = None

    def __exit__(self, *unused):
        """Close owned resources when leaving the scope, including on failure."""
        self.close()


class CalibrationStats:
    """Exact NLL, ECE, Brier and uniform reliability curves in constant memory."""

    def __init__(self, n_bins=30, ece_bins=15):
        """Initialize CalibrationStats with the supplied model, data or runtime settings."""
        if n_bins < 1 or ece_bins < 1:
            raise ValueError('Calibration bin counts must be positive')
        self.count = 0
        self.nll_sum = 0.
        self.brier_sum = np.zeros(3)
        self.bin_edges = np.linspace(0, 1, n_bins+1)
        self.bin_counts = np.zeros((3, n_bins), dtype=np.int64)
        self.bin_probs = np.zeros((3, n_bins))
        self.bin_true = np.zeros((3, n_bins))
        self.ece_edges = np.linspace(0, 1, ece_bins+1)
        self.ece_counts = np.zeros(ece_bins, dtype=np.int64)
        self.ece_confidence = np.zeros(ece_bins)
        self.ece_correct = np.zeros(ece_bins)

    def update(self, logits, labels):
        """Accumulate full-split metric sums and return probabilities and integer labels."""
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        truth = labels.detach().cpu().numpy()
        self.count += len(truth)
        self.nll_sum += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        for index in range(3):
            targets = (truth == index).astype(float)
            self.brier_sum[index] += np.square(probabilities[:, index]-targets).sum()
            bins = np.searchsorted(self.bin_edges[1:-1], probabilities[:, index], side='left')
            self.bin_counts[index] += np.bincount(bins, minlength=len(self.bin_edges)-1)
            self.bin_probs[index] += np.bincount(bins, weights=probabilities[:, index], minlength=len(self.bin_edges)-1)
            self.bin_true[index] += np.bincount(bins, weights=targets, minlength=len(self.bin_edges)-1)
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == truth
        bins = np.searchsorted(self.ece_edges[1:-1], confidence, side='left')
        self.ece_counts += np.bincount(bins, minlength=len(self.ece_counts))
        self.ece_confidence += np.bincount(bins, weights=confidence, minlength=len(self.ece_counts))
        self.ece_correct += np.bincount(bins, weights=correct.astype(float), minlength=len(self.ece_counts))
        return probabilities, truth

    @property
    def nll(self):
        """Return exact mean negative log likelihood over all cached observations."""
        return self.nll_sum/self.count

    @property
    def ece(self):
        """Return exact confidence-weighted expected calibration error."""
        return np.abs(self.ece_correct-self.ece_confidence).sum()/self.count

    @property
    def brier(self):
        """Return exact per-class mean squared probability error."""
        return self.brier_sum/self.count

    def curve(self, index):
        """Return fractions, probabilities and counts for occupied class bins."""
        occupied = self.bin_counts[index] > 0
        counts = self.bin_counts[index, occupied]
        return self.bin_true[index, occupied]/counts, self.bin_probs[index, occupied]/counts, counts


def evaluate_cache(cache, scale, device='cpu', maximum_plot_samples=100000, seed=42):
    """Compute exact metrics with an aligned, bounded random-priority plot reservoir."""
    if maximum_plot_samples < 1:
        raise ValueError('Plot sample size must be positive')
    original, calibrated = CalibrationStats(), CalibrationStats()
    rng = np.random.default_rng(seed)
    priorities = np.empty(0)
    before, after, labels = np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=int)
    with torch.no_grad():
        for logits, truth in cache.batches(device):
            probabilities, target = original.update(logits, truth)
            scaled, _ = calibrated.update(scale(logits), truth)
            priorities = np.concatenate((priorities, rng.random(len(target))))
            before = np.concatenate((before, probabilities))
            after = np.concatenate((after, scaled))
            labels = np.concatenate((labels, target))
            if len(labels) > maximum_plot_samples:
                selected = np.argpartition(priorities, maximum_plot_samples-1)[:maximum_plot_samples]
                priorities, before, after, labels = priorities[selected], before[selected], after[selected], labels[selected]
    return original, calibrated, before, after, labels
