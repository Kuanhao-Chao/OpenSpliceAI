'''Full predict process with custom data processing, batch prediction, and file output'''

from openspliceai.predict import predict
from pathlib import Path


# Resolve the absolute path to parent dir
# pdir = Path(__file__).resolve().parents[2]
pdir = "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/"
# Define file paths and model
input_sequence_file = f'{pdir}/examples/predict/tp53.fa'
output_dir = f'{pdir}/examples/predict/results/'
flanking_size = 10000
model = f'{pdir}/models/openspliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs14.pt'

### SETUP ###
import os
results_dir = f'{output_dir}/predict_custom/'
os.makedirs(results_dir, exist_ok=True)

# define sequence length for batching (default: 5000)
SL = 5000

# setup device and load model
device = predict.setup_device()
model, params = predict.load_pytorch_models(model, device, flanking_size, 10000)

# optional argument definition
gff_file = f'{pdir}/examples/data/chr22.gff'
debug = False
get_raw_tensors = True
threshold = 0.5

### PREPROCESSING INPUT DATA ###
# extract pc genes from gff file 
if gff_file:
    input_sequence_file = predict.process_gff(input_sequence_file, gff_file, results_dir)

# write all the sequences to a compressed datafile 
datafile_path, NAME, SEQ = predict.get_sequences(input_sequence_file, results_dir, flanking_size, debug=debug)

# convert sequences to one-hot encoding and save to dataset
dataset_path, LEN = predict.convert_sequences(datafile_path, results_dir, 5000, flanking_size, SEQ=SEQ, debug=debug)

### EXTRACT PREDICTIONS ### 
if get_raw_tensors:
    # if you want the raw tensors for every base, they will be stored in a .h5 file following this operation
    outfile = predict.get_prediction(model, dataset_path, device, params['BATCH_SIZE'], results_dir, debug=debug)
    print("Raw tensors saved to: ", outfile)
    
else:
    # if you want to process the predictions and write the splice sites to a BED file, use this operation
    predict.predict_and_write(model, dataset_path, device, params['BATCH_SIZE'], NAME, LEN, results_dir, threshold=threshold, debug=debug)