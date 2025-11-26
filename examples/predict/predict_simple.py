'''Utilizing the simple predict function in python to get raw predictions for relatively smaller sequences'''

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

# Read input sequence 
input_sequence = open(input_sequence_file).read().strip()

# Get predictions
# NOTE: This command will run the predict function with default parameters (no memory management), returning the raw tensors
predictions = predict.predict(input_sequence, model, flanking_size)
print(predictions, predictions.shape)