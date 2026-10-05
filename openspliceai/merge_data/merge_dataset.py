"""Validated, atomic merging of legacy HDF5 X/Y shards (Python API)."""
from pathlib import Path
import os
import tempfile

import h5py

from openspliceai.data_schema import shard_indices, validate_shard, validate_encoding


def merge_dataset(args):
    """Merge paired shards in numeric order and publish each split atomically.

    ``args`` supplies input_dir, output_dir and chr_split (test or train-test).
    Optional biotype selects non-coding suffixes. Validation is included when
    present in every input directory; mixed presence raises an error. Metadata
    remains in its original input files; sample arrays are copied unchanged.
    """
    if args.chr_split not in ('test', 'train-test') or not args.input_dir:
        raise ValueError('Merge requires input directories and test or train-test')
    biotype = getattr(args, 'biotype', 'all')
    if biotype not in ('all', 'protein-coding', 'non-coding'):
        raise ValueError('Unknown biotype')
    suffix = '_ncRNA' if biotype == 'non-coding' else ''
    inputs = [Path(directory) for directory in args.input_dir]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = ['test'] if args.chr_split == 'test' else ['test', 'train']
    if args.chr_split == 'train-test':
        validation = [(directory/f'dataset_validation{suffix}.h5').is_file() for directory in inputs]
        if any(validation) and not all(validation):
            raise ValueError('Validation split must be present in every input directory or none')
        if all(validation):
            splits.append('validation')
    for split in splits:
        filename = f'dataset_{split}{suffix}.h5'
        destination = output/filename
        if any((directory/filename).resolve() == destination.resolve() for directory in inputs):
            raise ValueError('Merge output must be distinct from all input datasets')
        descriptor, temporary = tempfile.mkstemp(prefix='.'+filename+'.', dir=output)
        os.close(descriptor)
        try:
            with h5py.File(temporary, 'w') as merged:
                count, schema = 0, None
                for directory in inputs:
                    with h5py.File(directory/filename, 'r') as source:
                        for index in shard_indices(source):
                            samples = validate_shard(source, index)
                            x, y = source[f'X{index}'], source[f'Y{index}']
                            dimensions = (x.shape[1:], y.shape[2:])
                            if schema is not None and dimensions != schema:
                                raise ValueError('All merged shards must share context and output lengths')
                            schema = dimensions
                            for first in range(0, samples, 32):
                                validate_encoding(x[first:first+32], y[0, first:first+32])
                            source.copy(x, merged, name=f'X{count}')
                            source.copy(y, merged, name=f'Y{count}')
                            count += 1
                merged.flush()
            with open(temporary, 'rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
