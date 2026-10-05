"""
Filename: model_utils.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Calibrate the OpenSpliceAI model.
"""

from openspliceai.checkpoints import unpack_checkpoint, CheckpointError, load_checkpoint
import numpy as np
from openspliceai.model_config import model_hyperparameters
from openspliceai.constants import *
from openspliceai.train_base.openspliceai import SpliceAI

def initialize_model_and_optim(device, flanking_size, pretrained_model):
    """Build a context-matched model, AdamW optimizer and scheduler."""
    L, N_GPUS, W, AR, BATCH_SIZE = model_hyperparameters(flanking_size)
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR, apply_softmax=False).to(device)
    # Calibration must evaluate the complete trained model. Partial loading would
    # calibrate randomly initialized layers and invalidate the reported metrics.
    state_dict = load_checkpoint(pretrained_model, device)
    state_dict, temperature = unpack_checkpoint(state_dict, flanking_size)
    if temperature is not None:
        raise ValueError('Calibration requires an uncalibrated base checkpoint')
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointError(f'Calibration requires a complete {flanking_size}nt checkpoint: {error}') from error

    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, params
