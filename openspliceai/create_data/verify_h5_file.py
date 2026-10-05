"""Bounded integrity checks and one representative plot for each encoded split."""
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt

from openspliceai.data_schema import shard_indices, validate_shard, validate_encoding


def verify_h5(args):
    """Validate every X/Y pair, including validation, metadata and empty splits.

    Read at most 32 windows per batch. A split with no windows is reported
    explicitly; fitting commands reject it when selected for training/evaluation.
    """
    start = time.monotonic()
    if args.biotype not in ('all', 'protein-coding', 'non-coding'):
        raise ValueError('Unknown biotype')
    if args.chr_split not in ('test', 'train-test'):
        raise ValueError('Chromosome split must be test or train-test')
    suffix = '_ncRNA' if args.biotype == 'non-coding' else ''
    for split in (('test',) if args.chr_split == 'test' else ('test', 'train', 'validation')):
        filename = Path(args.output_dir)/f'dataset_{split}{suffix}.h5'
        print(f'Verifying {filename}...')
        representative = None
        count = 0
        with h5py.File(filename, 'r') as handle:
            for index in shard_indices(handle):
                samples = validate_shard(handle, index)
                for first in range(0, samples, 32):
                    inputs = handle[f'X{index}'][first:first+32]
                    labels = handle[f'Y{index}'][0, first:first+32]
                    validate_encoding(inputs, labels)
                    if representative is None:
                        representative = inputs[0].sum(axis=1)
                count += samples
        print(f'{split}: {count} windows')
        if representative is None:
            print(f'{split}: empty split; no verification plot')
            continue
        figure, axis = plt.subplots(figsize=(7, 3))
        try:
            axis.plot(representative)
            axis.set(xlabel='Input position', ylabel='Encoded nucleotide count')
            figure.savefig(Path(args.output_dir)/f'verify_{split}{suffix}.png', dpi=150, bbox_inches='tight')
        finally:
            plt.close(figure)
    print(f'--- {time.monotonic()-start:.2f} seconds ---')
