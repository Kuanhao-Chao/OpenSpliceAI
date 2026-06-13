###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

# PROCESSING BATCH SIZE
from math import ceil
def ceil_div(x, y):
    """
    Calculate the ceiling of a division between two numbers.

    Parameters:
    - x (int): Numerator
    - y (int): Denominator

    Returns:
    - int: The ceiling of the division result.
    """
    return int(ceil(float(x)/y))

# FOR TESTING PURPOSES
import os
import sys
import psutil
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB", file=sys.stderr)

# SETUP INITIALIZATION   
def initialize_constants(flanking_size, hdf_threshold_len=0, flush_predict_threshold=500, chunk_size=100, split_fasta_threshold=1500000):
    from openspliceai.constants import SL
    
    assert int(flanking_size) in [80, 400, 2000, 10000]
    
    CL_max = flanking_size                              # context length for sequence prediction (flanking size sum)
    HDF_THRESHOLD_LEN = hdf_threshold_len               # maximum size before reading sequence into an HDF file for storage
    FLUSH_PREDICT_THRESHOLD = flush_predict_threshold   # maximum number of predictions before flushing to file
    CHUNK_SIZE = chunk_size                             # chunk size for loading hdf5 dataset
    SPLIT_FASTA_THRESHOLD = split_fasta_threshold       # maximum length of fasta entry before splitting
    
    return {'SL': SL,
            'CL_max': CL_max, 
            'HDF_THRESHOLD_LEN': HDF_THRESHOLD_LEN, 
            'FLUSH_PREDICT_THRESHOLD': FLUSH_PREDICT_THRESHOLD, 
            'CHUNK_SIZE': CHUNK_SIZE, 
            'SPLIT_FASTA_THRESHOLD': SPLIT_FASTA_THRESHOLD}
    
def initialize_paths(output_dir, flanking_size, sequence_length, model_arch='SpliceAI'):
    """Initialize project directories and create them if they don't exist."""
    BASENAME = f"{model_arch}_{sequence_length}_{flanking_size}"
    model_pred_outdir = f"{output_dir}/{BASENAME}/"
    os.makedirs(model_pred_outdir, exist_ok=True)

    return model_pred_outdir