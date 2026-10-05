"""
Filename: paralogs.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Remove paralogous sequences between train and test datasets using mappy.
"""

import mappy as mp
import numpy as np
import h5py
import tempfile
import os

def remove_paralogous_sequences(train_data, test_data, min_identity, min_coverage, output_dir, exp):
    """
    Remove paralogous sequences between train and test datasets using mappy.
    
    :param train_data: List of lists containing train data (NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL)
    :param test_data: List of lists containing test data (NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL)
    :param min_identity: Minimum identity for sequences to be considered paralogous
    :param min_coverage: Minimum coverage for sequences to be considered paralogous
    :return: Tuple of (filtered_train_data, filtered_test_data)
    """
    if not 0 <= min_identity <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError('Paralogy identity and coverage thresholds must be between zero and one')
    for data in (train_data, test_data):
        if len(data) != 7 or len({len(field) for field in data}) != 1:
            raise ValueError('Expected seven equally sized gene data fields')
        if any(not sequence for sequence in data[5]):
            raise ValueError('Paralogy filtering requires nonempty sequences')
    if not train_data[0] or not test_data[0]:
        return train_data, test_data
    filtered_test_data = [[] for _ in range(len(test_data))]
    paralogous_count = 0
    total_count = len(test_data[5])
    # The temporary reference and diagnostic log close even if minimap2 fails.
    # A requested filter must never silently return unfiltered held-out data.
    with tempfile.TemporaryDirectory(prefix='openspliceai-paralogs-') as directory:
        reference = os.path.join(directory, 'train.fa')
        with open(reference, 'w') as output:
            for i, sequence in enumerate(train_data[5]):
                output.write(f'>seq{i}\n{sequence}\n')
        aligner = mp.Aligner(reference, preset='map-ont')
        if not aligner:
            raise ValueError('Unable to build minimap2 index for requested paralogy filtering')
        with open(os.path.join(output_dir, f'removed_paralogs_{exp}.txt'), 'w') as output:
            for i, sequence in enumerate(test_data[5]):
                is_paralogous = False
                for hit in aligner.map(sequence):
                    if hit.blen <= 0 or not 0 <= hit.q_st <= hit.q_en <= len(sequence) or not 0 <= hit.mlen <= hit.blen:
                        raise ValueError('minimap2 returned an invalid alignment span')
                    identity = hit.mlen / hit.blen
                    # Alignment block length includes reference deletions. Query
                    # coverage is the aligned query span, bounded by [0, 1].
                    coverage = (hit.q_en-hit.q_st) / len(sequence)
                    output.write(f'{test_data[0][i]}\t{identity}\t{coverage}\n')
                    if identity >= min_identity and coverage >= min_coverage:
                        is_paralogous = True
                        paralogous_count += 1
                        break
                if not is_paralogous:
                    for j in range(len(test_data)):
                        filtered_test_data[j].append(test_data[j][i])
    print("\tParalogy removal process completed.")
    print(f"\tNumber of paralogous sequences removed: {paralogous_count}")
    print(f"\tFinal {exp} set size: {len(filtered_test_data[0])}")
    print(f"\tPercentage of {exp} set removed: {(paralogous_count / total_count) * 100:.2f}%")
    return train_data, filtered_test_data


def write_h5_file(output_dir, data_type, data):
    """
    Write the data to an h5 file.
    """
    h5fname = os.path.join(output_dir, f'datafile_{data_type}.h5')
    dt = h5py.string_dtype(encoding='utf-8')
    
    dataset_names = ['NAME', 'CHROM', 'STRAND', 'TX_START', 'TX_END', 'SEQ', 'LABEL']
    if len(data) != 7 or len({len(field) for field in data}) != 1:
        raise ValueError('Expected seven equally sized gene data fields')
    with h5py.File(h5fname, 'w') as h5f:
        for i, name in enumerate(dataset_names):
            h5f.create_dataset(name, data=np.asarray(data[i], dtype=dt), dtype=dt)
