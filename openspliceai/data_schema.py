"""Validation of the legacy X/Y HDF5 shard contract without materializing data."""
import re
import numpy as np


def shard_indices(handle):
    """Return sorted paired shard indices; metadata groups/attributes are ignored."""
    xs, ys = set(), set()
    for key in handle:
        match = re.fullmatch(r'([XY])(\d+)', key)
        if match:
            if str(int(match[2])) != match[2]:
                raise ValueError('Shard names must use canonical integer indices (X0, Y0, ...)')
            (xs if match[1] == 'X' else ys).add(int(match[2]))
    if xs != ys:
        raise ValueError('HDF5 must contain matching X/Y shard pairs')
    return sorted(xs)


def validate_shard(handle, index):
    """Validate X=(N,L,4), Y=(1,N,SL,3) and return the sample count.

    X includes symmetric context; Y remains at the output length. Zero-length
    shards are allowed so callers can report a selected split with no samples.
    """
    x, y = handle[f'X{index}'], handle[f'Y{index}']
    if not hasattr(x, 'shape') or len(x.shape) != 3 or x.shape[-1] != 4:
        raise ValueError(f'X{index} must have shape (samples, sequence, 4)')
    if not hasattr(y, 'shape') or len(y.shape) != 4 or y.shape[0] != 1 or y.shape[-1] != 3 or y.shape[1] != x.shape[0] or y.shape[2] < 1:
        raise ValueError(f'Y{index} must have shape (1, samples, output_sequence, 3)')
    context = x.shape[1] - y.shape[2]
    if context < 0 or context % 2:
        raise ValueError(f'X/Y shard {index} must differ by symmetric nonnegative context')
    return x.shape[0]


def validate_encoding(inputs, labels):
    """Check finite one-hot/zero rows in channel-last encoded arrays.

    Zero input rows denote unknown nucleotides; zero label rows are unobserved
    padding and are excluded from training/calibration metrics and losses.
    """
    for name, values in (('Input', inputs), ('Label', labels)):
        if not np.isfinite(values).all() or not np.logical_or(values == 0, values == 1).all():
            raise ValueError(f'{name} encoding must contain finite zero/one values')
        if np.any(values.sum(axis=-1) > 1):
            raise ValueError(f'{name} encoding must be one-hot or an all-zero padding row')
