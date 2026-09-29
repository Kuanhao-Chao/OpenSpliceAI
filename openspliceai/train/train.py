"""
Filename: train.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Train the OpenSpliceAI model.
"""

import sys
import numpy as np
from openspliceai.model_config import model_hyperparameters
import torch
from contextlib import ExitStack
import torch.optim as optim
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *

def initialize_model_and_optim(device, flanking_size, epochs, scheduler):
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L, N_GPUS, W, AR, BATCH_SIZE = model_hyperparameters(flanking_size)
    CL = 2 * np.sum(AR*(W-1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    model = SpliceAI(L, W, AR).to(device)
    print(model, file=sys.stderr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-5, epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler_obj, params


def train(args):
    """Train a SpliceAI model from scratch (entry point for the ``train`` subcommand).

    Sets up the device, resolves the output/log directory layout, loads the
    train/validation/test HDF5 datasets, builds the model + AdamW optimizer +
    LR scheduler sized for ``args.flanking_size``, then runs the shared
    ``train_model`` loop. Side effects: writes per-epoch checkpoints
    (``model_{epoch}.pt``, ``model_best.pt``) and appends metrics to the
    ``LOG/{TRAIN,VAL,TEST}`` ``.txt`` files; returns nothing.
    """
    print("Running OpenSpliceAI with 'train' mode")
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, valid_h5f, test_h5f, batch_num = load_datasets(args)
    with ExitStack() as stack:
        for handle in (train_h5f, valid_h5f, test_h5f):
            stack.enter_context(handle)
        train_idxs, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)

        model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size, args.epochs, args.scheduler)
        params["RANDOM_SEED"] = args.random_seed
        train_metric_files = create_metric_files(log_output_train_base)
        valid_metric_files = create_metric_files(log_output_val_base)
        test_metric_files = create_metric_files(log_output_test_base)
        train_model(model, optimizer, scheduler, train_h5f, valid_h5f, test_h5f,
                    train_idxs, val_idxs, test_idxs, model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
