"""
Filename: transfer.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Transfer-learning for OpenSpliceAI model.
"""

import numpy as np
import torch
import torch.optim as optim
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def initialize_model_and_optim_transfer(device, flanking_size, epochs, scheduler,
                               pretrained_model, unfreeze, unfreeze_all):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR).to(device)
    # # Print the shapes of the parameters in the initialized model
    # print("\nInitialized model parameter shapes:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print("\n unfreeze_all:", unfreeze_all)
    if not unfreeze_all:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last `unfreeze` layers
        if unfreeze > 0:
            # Unfreeze the last few layers (example: last residual unit)
            for param in model.residual_units[-unfreeze].parameters():
                param.requires_grad = True
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler_obj, params


def transfer(args):
    print('Running OpenSpliceAI with transfer mode.')
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, valid_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim_transfer(device, args.flanking_size, args.epochs, args.scheduler, args.pretrained_model, args.unfreeze, args.unfreeze_all)
    
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, valid_h5f, test_h5f, train_idxs, 
                val_idxs, test_idxs, model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    valid_h5f.close()
    test_h5f.close()