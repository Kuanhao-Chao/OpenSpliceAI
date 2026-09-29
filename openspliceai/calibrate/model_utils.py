"""
Filename: model_utils.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Calibrate the OpenSpliceAI model.
"""

import torch
import numpy as np
from openspliceai.model_config import model_hyperparameters
from openspliceai.constants import *
from openspliceai.train_base.openspliceai import SpliceAI

def initialize_model_and_optim(device, flanking_size, pretrained_model):
    L, N_GPUS, W, AR, BATCH_SIZE = model_hyperparameters(flanking_size)
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR, apply_softmax=False).to(device)
    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Calibration must evaluate the complete trained model. Partial loading would
    # calibrate randomly initialized layers and invalidate the reported metrics.
    state_dict = torch.load(pretrained_model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, params

