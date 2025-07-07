"""
Filename: paralogs.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Remove paralogous sequences between train and test datasets using mappy.
"""

import mappy as mp
import numpy as np
import logging
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
    print(f"\tStarting paralogy removal process...")
    print(f"\tInitial train set size: {len(train_data[0])}")
    print(f"\tInitial test set size: {len(test_data[0])}")
    train_seqs = train_data[5]  # SEQ is at index 5
    # Create a temporary file with training sequences
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for i, seq in enumerate(train_seqs):
            temp_file.write(f">seq{i}\n{seq}\n")
        temp_filename = temp_file.name
    # Create a mappy index from the training sequences
    print("\tCreating mappy index from training sequences...")
    try:
        aligner = mp.Aligner(temp_filename, preset="map-ont")
        if not aligner:
            raise Exception("\tFailed to load/build index")
    except Exception as e:
        logging.error(f"\tError creating mappy aligner: {str(e)}")
        os.unlink(temp_filename)
        return train_data, test_data
    filtered_test_data = [[] for _ in range(len(test_data))]
    paralogous_count = 0
    total_count = len(test_data[5])
    fw = open(f"{output_dir}removed_paralogs_{exp}.txt", "w")
    print("\tStarting to process test sequences...")
    for i in range(total_count):  # Iterate over test sequences
        test_seq = test_data[5][i]
        is_paralogous = False
        for hit in aligner.map(test_seq):
            identity = hit.mlen / hit.blen
            coverage = hit.blen / len(test_seq)
            fw.write(f"{test_data[0][i]}\t{identity}\t{coverage}\n")
            if identity >= min_identity and coverage >= min_coverage:
                # fw.write(f"{test_data[0][i]}\t{identity}\t{coverage}\n")
                print(f"\t\tParalogs detected: Identity: {identity}, Coverage: {coverage}")
                is_paralogous = True
                paralogous_count += 1
                break
        if not is_paralogous:
            for j in range(len(test_data)):
                filtered_test_data[j].append(test_data[j][i])
        # Log progress every 1000 sequences
        if (i + 1) % 1000 == 0:
            print(f"\tProcessed {i + 1}/{total_count} sequences...")
    print(f"\tParalogy removal process completed.")
    print(f"\tNumber of paralogous sequences removed: {paralogous_count}")
    print(f"\tFinal {exp} set size: {len(filtered_test_data[0])}")
    print(f"\tPercentage of {exp} set removed: {(paralogous_count / total_count) * 100:.2f}%")
    fw.close()
    return train_data, filtered_test_data


def write_h5_file(output_dir, data_type, data):
    """
    Write the data to an h5 file.
    """
    h5fname = output_dir + f'datafile_{data_type}.h5'
    h5f = h5py.File(h5fname, 'w')
    dt = h5py.string_dtype(encoding='utf-8')
    
    dataset_names = ['NAME', 'CHROM', 'STRAND', 'TX_START', 'TX_END', 'SEQ', 'LABEL']
    for i, name in enumerate(dataset_names):
        h5f.create_dataset(name, data=np.asarray(data[i], dtype=dt), dtype=dt)    
    h5f.close()