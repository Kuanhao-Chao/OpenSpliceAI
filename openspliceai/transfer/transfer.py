"""
Filename: transfer.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Transfer-learning for OpenSpliceAI model.
"""

import os
import h5py
import numpy as np
import torch
import torch.optim as optim
from openspliceai.train_base.openspliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *


def initialize_model_and_optim_transfer(device, flanking_size, epochs, scheduler,
                               pretrained_model, unfreeze, unfreeze_all, weight_decay=0.01):
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
        # Unfreeze the last `unfreeze` residual units. The residual_units ModuleList
        # interleaves Skip layers (one after every 4 ResidualUnits), so select the
        # ResidualUnit instances explicitly rather than indexing the raw list.
        if unfreeze > 0:
            res_units = [m for m in model.residual_units if isinstance(m, ResidualUnit)]
            unfreeze = min(unfreeze, len(res_units))
            for layer in res_units[-unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=weight_decay)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler_obj, params


def build_frozen_teacher(device, params, teacher_path):
    """Build a frozen copy of the SpliceAI model to act as a knowledge-distillation teacher.

    Reuses the ``(L, W, AR)`` architecture already derived in ``params`` (no new
    hyperparameter-table duplication), loads ``teacher_path`` with the same
    size-mismatch key filtering as the student, and returns it in ``eval`` mode
    with every parameter frozen (``requires_grad=False``).
    """
    teacher = SpliceAI(params['L'], params['W'], params['AR']).to(device)
    t_state = torch.load(teacher_path, map_location=device)
    t_dict = teacher.state_dict()
    t_state = {k: v for k, v in t_state.items() if k in t_dict and v.size() == t_dict[k].size()}
    teacher.load_state_dict(t_state, strict=False)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def cycle_anchor_batches(anchor_h5f, anchor_idxs, device, batch_size, params):
    """Yield genomic anchor batches forever, cycling over the shards of ``anchor_h5f``.

    Used to feed the distillation teacher a fresh genomic window every training
    step. Infinite by design so ``next(...)`` never raises ``StopIteration``
    mid-epoch.
    """
    while True:
        yielded = False
        for idx in anchor_idxs:
            loader = load_data_from_shard(anchor_h5f, idx, device, batch_size, params, shuffle=True)
            for batch in loader:
                yielded = True
                yield batch
        if not yielded:
            raise ValueError(
                "Anchor dataset yielded no batches: every shard is smaller than the "
                "(distillation) batch size. Lower --distill-batch-size or supply larger anchor shards."
            )


def setup_forgetting_mitigation(args, params, device, train_idxs, log_output_train_base):
    """Wire up the optional catastrophic-forgetting mitigations onto ``params``.

    Activates only the features the user requested (all default-off): genomic
    forgetting-tracking eval, rehearsal/data-mixing shard table, and the
    knowledge-distillation (LwF) teacher + anchor iterator. Returns
    ``(train_idxs, open_handles)`` where ``train_idxs`` may have been replaced by
    rehearsal shard-table positions and ``open_handles`` lists HDF5 files to
    close when training finishes.
    """
    open_handles = []
    # Read all new options defensively so tests/callers that build a bare args namespace
    # (without these fields) keep the legacy behaviour.
    genomic_eval_dataset = getattr(args, "genomic_eval_dataset", None)
    rehearsal_dataset = getattr(args, "rehearsal_dataset", None)
    rehearsal_shards = getattr(args, "rehearsal_shards", -1)
    distill_weight = getattr(args, "distill_weight", 0.0)
    distill_shards = getattr(args, "distill_shards", None)
    distill_teacher = getattr(args, "distill_teacher", None)
    distill_batch_size = getattr(args, "distill_batch_size", -1)
    l2sp = getattr(args, "l2sp", 0.0)

    # --- Genomic forgetting-tracking eval (D3) ---
    if genomic_eval_dataset:
        genomic_h5f = h5py.File(genomic_eval_dataset, 'r')
        open_handles.append(genomic_h5f)
        genomic_log_base = os.path.join(os.path.dirname(log_output_train_base.rstrip('/')), "GENOMIC") + "/"
        os.makedirs(genomic_log_base, exist_ok=True)
        params["GENOMIC_H5F"] = genomic_h5f
        params["GENOMIC_IDXS"] = np.arange(len(genomic_h5f.keys()) // 2)
        params["GENOMIC_METRIC_FILES"] = create_metric_files(genomic_log_base)
        print(f"[forgetting] Genomic eval each epoch -> {genomic_log_base}")

    # --- Rehearsal / data-mixing (D2) ---
    if rehearsal_dataset:
        rehearsal_h5f = h5py.File(rehearsal_dataset, 'r')
        open_handles.append(rehearsal_h5f)
        n_avail = len(rehearsal_h5f.keys()) // 2
        n_use = n_avail if rehearsal_shards < 0 else min(rehearsal_shards, n_avail)
        shard_table = [("train", int(i)) for i in train_idxs]
        shard_table += [("rehearsal", int(i)) for i in range(n_use)]
        params["REHEARSAL_H5F"] = rehearsal_h5f
        params["SHARD_TABLE"] = shard_table
        train_idxs = np.arange(len(shard_table))
        print(f"[forgetting] Rehearsal: interleaving {n_use} genomic shard(s) with "
              f"{len(shard_table) - n_use} training shard(s).")

    # --- Knowledge distillation / Learning-without-Forgetting (D1) ---
    if distill_weight and distill_weight > 0:
        if distill_shards is None:
            raise ValueError("--distill-shards is required when --distill-weight > 0")
        teacher_path = distill_teacher or args.pretrained_model
        teacher = build_frozen_teacher(device, params, teacher_path)
        anchor_h5f = h5py.File(distill_shards, 'r')
        open_handles.append(anchor_h5f)
        anchor_idxs = np.arange(len(anchor_h5f.keys()) // 2)
        distill_bs = params["BATCH_SIZE"] if distill_batch_size < 0 else distill_batch_size
        params["TEACHER"] = teacher
        params["ANCHOR_ITER"] = cycle_anchor_batches(anchor_h5f, anchor_idxs, device, distill_bs, params)
        params["DISTILL_WEIGHT"] = distill_weight
        print(f"[forgetting] Distillation: lambda={distill_weight}, teacher={teacher_path}, "
              f"anchors={distill_shards}")
        if l2sp and l2sp > 0:
            params["L2SP"] = l2sp
            params["L2SP_REF"] = {n: p.detach().clone() for n, p in teacher.named_parameters()}
            print(f"[forgetting] L2-SP toward pretrained weights: mu={l2sp}")

    return train_idxs, open_handles


def transfer(args):
    """Fine-tune a pretrained SpliceAI model on a new dataset (entry point for the ``transfer`` subcommand).

    Like :func:`train`, but ``initialize_model_and_optim_transfer`` first loads
    ``args.pretrained_model`` (filtering size-mismatched keys) and, unless
    ``--unfreeze-all``, freezes everything except the last ``args.unfreeze``
    residual units; it uses a smaller learning rate (1e-4). Side effects: writes
    checkpoints and metric logs under the experiment output directory; returns
    nothing.
    """
    print('Running OpenSpliceAI with transfer mode.')
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, valid_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(train_h5f, valid_h5f, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim_transfer(device, args.flanking_size, args.epochs, args.scheduler, args.pretrained_model, args.unfreeze, args.unfreeze_all, getattr(args, 'weight_decay', 0.01))

    params["RANDOM_SEED"] = args.random_seed
    # Optional catastrophic-forgetting mitigations (genomic eval / rehearsal / distillation); all default-off.
    train_idxs, open_handles = setup_forgetting_mitigation(args, params, device, train_idxs, log_output_train_base)
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, valid_h5f, test_h5f, train_idxs,
                val_idxs, test_idxs, model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    valid_h5f.close()
    test_h5f.close()
    for handle in open_handles:
        handle.close()