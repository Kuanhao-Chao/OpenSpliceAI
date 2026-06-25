"""Unit tests for the catastrophic-forgetting mitigations added to the ``transfer``
subcommand:

- ``train_base.utils.distillation_loss``     -- the LwF/knowledge-distillation auxiliary term
- ``train_base.utils.resolve_shard_loader``  -- rehearsal/data-mixing shard-table routing
- ``transfer.setup_forgetting_mitigation``   -- wires teacher / anchor iterator / genomic eval

These exercise the production code paths on the tiny ``model_80nt`` fixture and synthetic
HDF5 shards, with everything default-off proven to be a no-op.
"""
import copy
import os
import types

import h5py
import numpy as np
import pytest
import torch

import openspliceai.train_base.utils as tbu
from openspliceai.transfer import transfer as tr
from tests.fixtures.synthetic import write_dataset_h5

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# distillation_loss
# --------------------------------------------------------------------------- #
def test_distillation_loss_off_is_noop():
    """Without DISTILL_WEIGHT (or <= 0) the term is a plain 0.0 -- adds nothing, no teacher run."""
    assert tbu.distillation_loss(object(), {}, CPU) == 0.0
    assert tbu.distillation_loss(object(), {"DISTILL_WEIGHT": 0.0}, CPU) == 0.0


def _anchor_iter(tmp_path, batch_size=2, seed=7):
    p = str(tmp_path / "anchor.h5")
    write_dataset_h5(p, n_windows=2, seed=seed)
    h5f = h5py.File(p, "r")
    it = tr.cycle_anchor_batches(h5f, np.array([0]), CPU, batch_size, {})
    return it, h5f


def test_distillation_loss_on_is_grad_tensor(tmp_path, model_80nt):
    """With a frozen teacher and a genomic anchor batch, the term is a positive, finite,
    differentiable tensor whose backward populates student gradients."""
    teacher = copy.deepcopy(model_80nt).eval()
    teacher.requires_grad_(False)
    it, h5f = _anchor_iter(tmp_path)
    params = {"CL": 80, "N_GPUS": 2, "BATCH_SIZE": 2,
              "TEACHER": teacher, "ANCHOR_ITER": it, "DISTILL_WEIGHT": 0.7}
    try:
        model_80nt.zero_grad(set_to_none=True)
        loss = tbu.distillation_loss(model_80nt, params, CPU)
        assert torch.is_tensor(loss) and loss.requires_grad and torch.isfinite(loss)
        assert loss.item() > 0  # CE(teacher_probs, student_probs) of a softmax dist is > 0
        loss.backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in model_80nt.parameters())
    finally:
        h5f.close()


def test_distillation_loss_scales_with_weight(tmp_path, model_80nt):
    """The term is linear in DISTILL_WEIGHT for the same anchor batch + weights (deterministic)."""
    teacher = copy.deepcopy(model_80nt).eval()
    teacher.requires_grad_(False)

    def one(weight):
        it, h5f = _anchor_iter(tmp_path, seed=11)  # same seed -> same anchor batch
        try:
            torch.manual_seed(0)
            return tbu.distillation_loss(model_80nt, {
                "CL": 80, "N_GPUS": 2, "BATCH_SIZE": 2,
                "TEACHER": teacher, "ANCHOR_ITER": it, "DISTILL_WEIGHT": weight}, CPU).item()
        finally:
            h5f.close()

    assert one(1.0) == pytest.approx(2 * one(0.5), rel=1e-5)


def test_distillation_loss_l2sp_adds_positive_drift_penalty(tmp_path, model_80nt):
    """L2-SP penalises drift from the reference weights; when the student already differs
    from the reference, enabling L2-SP strictly increases the loss."""
    teacher = copy.deepcopy(model_80nt).eval()
    teacher.requires_grad_(False)
    # Make the student differ from the reference so the penalty is non-zero.
    ref = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    with torch.no_grad():
        for p in model_80nt.parameters():
            p.add_(0.05)

    def one(with_l2sp):
        it, h5f = _anchor_iter(tmp_path, seed=13)
        params = {"CL": 80, "N_GPUS": 2, "BATCH_SIZE": 2,
                  "TEACHER": teacher, "ANCHOR_ITER": it, "DISTILL_WEIGHT": 0.5}
        if with_l2sp:
            params["L2SP"] = 0.1
            params["L2SP_REF"] = ref
        try:
            return tbu.distillation_loss(model_80nt, params, CPU).item()
        finally:
            h5f.close()

    assert one(True) > one(False)


# --------------------------------------------------------------------------- #
# resolve_shard_loader (rehearsal shard-table routing)
# --------------------------------------------------------------------------- #
def _tiny_shard(path, fill):
    """A minimal X0/Y0 shard with a constant X fill value so its source is identifiable."""
    X = np.full((4, 8, 4), fill, dtype=np.int8)
    Y = np.zeros((1, 4, 6, 3), dtype=np.int8)
    Y[..., 0] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("X0", data=X)
        f.create_dataset("Y0", data=Y)


def test_resolve_shard_loader_legacy_without_table(tmp_path):
    """No SHARD_TABLE -> behaves exactly like load_data_from_shard on the given handle."""
    p = str(tmp_path / "train.h5")
    _tiny_shard(p, fill=0)
    h5f = h5py.File(p, "r")
    try:
        loader = tbu.resolve_shard_loader(h5f, 0, CPU, 2, {}, shuffle=False)
        xb = next(iter(loader))[0]
        assert xb.shape == (2, 4, 8)
        assert torch.all(xb == 0)
    finally:
        h5f.close()


def test_resolve_shard_loader_routes_train_vs_rehearsal(tmp_path):
    """With a SHARD_TABLE, position maps to (source, real_idx); 'rehearsal' reads the genomic
    handle, 'train' the training handle."""
    tr_path, re_path = str(tmp_path / "train.h5"), str(tmp_path / "reh.h5")
    _tiny_shard(tr_path, fill=0)
    _tiny_shard(re_path, fill=1)
    train_h5f, reh_h5f = h5py.File(tr_path, "r"), h5py.File(re_path, "r")
    params = {"SHARD_TABLE": [("train", 0), ("rehearsal", 0)], "REHEARSAL_H5F": reh_h5f}
    try:
        xb_train = next(iter(tbu.resolve_shard_loader(train_h5f, 0, CPU, 2, params, shuffle=False)))[0]
        xb_reh = next(iter(tbu.resolve_shard_loader(train_h5f, 1, CPU, 2, params, shuffle=False)))[0]
        assert torch.all(xb_train == 0)   # position 0 -> ("train", 0)
        assert torch.all(xb_reh == 1)     # position 1 -> ("rehearsal", 0) -> genomic handle
    finally:
        train_h5f.close()
        reh_h5f.close()


# --------------------------------------------------------------------------- #
# setup_forgetting_mitigation (wiring)
# --------------------------------------------------------------------------- #
def test_setup_forgetting_mitigation_wires_all_three(tmp_path, packaged_80nt_state):
    gpath, rpath, apath = (str(tmp_path / f"{n}.h5") for n in ("gen", "reh", "anc"))
    for i, p in enumerate((gpath, rpath, apath)):
        write_dataset_h5(p, n_windows=2, seed=i + 1)

    model, _opt, _sched, params = tr.initialize_model_and_optim_transfer(
        CPU, 80, 10, "MultiStepLR", packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    params["RANDOM_SEED"] = 42

    args = types.SimpleNamespace(
        pretrained_model=packaged_80nt_state,
        genomic_eval_dataset=gpath,
        rehearsal_dataset=rpath, rehearsal_shards=1,
        distill_weight=0.5, distill_shards=apath, distill_teacher=None,
        distill_batch_size=2, l2sp=0.1,
    )
    log_base = str(tmp_path / "LOG" / "TRAIN") + "/"
    os.makedirs(log_base, exist_ok=True)

    new_idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), log_base)
    try:
        # D3 genomic eval
        assert params["GENOMIC_H5F"] is not None
        assert "donor_auprc" in params["GENOMIC_METRIC_FILES"]
        assert os.path.isdir(os.path.join(os.path.dirname(log_base.rstrip("/")), "GENOMIC"))
        # D2 rehearsal: shard table mixes train + 1 rehearsal shard; train_idxs become positions
        assert params["SHARD_TABLE"][0] == ("train", 0)
        assert ("rehearsal", 0) in params["SHARD_TABLE"]
        assert len(new_idxs) == len(params["SHARD_TABLE"])
        # D1 distillation: frozen teacher + infinite anchor iterator + L2-SP reference
        assert all(not p.requires_grad for p in params["TEACHER"].parameters())
        assert params["DISTILL_WEIGHT"] == 0.5
        assert params["L2SP"] == 0.1 and "L2SP_REF" in params
        batch = next(params["ANCHOR_ITER"])
        assert batch[0].shape[0] >= 1
    finally:
        for h in handles:
            h.close()


def test_setup_distill_requires_shards(tmp_path, packaged_80nt_state):
    """--distill-weight>0 without --distill-shards is a clear error."""
    _model, _o, _s, params = tr.initialize_model_and_optim_transfer(
        CPU, 80, 10, "MultiStepLR", packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state,
                                 distill_weight=0.5, distill_shards=None)
    with pytest.raises(ValueError, match="distill-shards"):
        tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), str(tmp_path) + "/")


# --------------------------------------------------------------------------- #
# cycle_anchor_batches edge cases
# --------------------------------------------------------------------------- #
def test_cycle_anchor_batches_raises_on_empty_shards(tmp_path):
    """If every shard is smaller than the batch size, drop_last yields nothing -> clear error
    instead of an infinite hang."""
    p = str(tmp_path / "small.h5")
    write_dataset_h5(p, n_windows=1, seed=1)  # 1 window, batch_size 2 -> 0 batches (drop_last)
    h5f = h5py.File(p, "r")
    it = tr.cycle_anchor_batches(h5f, np.array([0]), CPU, 2, {})
    try:
        with pytest.raises(ValueError, match="yielded no batches"):
            next(it)
    finally:
        h5f.close()


def test_cycle_anchor_batches_cycles_indefinitely(tmp_path):
    """The iterator is infinite: it keeps yielding past one full pass over the shards."""
    p = str(tmp_path / "anc.h5")
    write_dataset_h5(p, n_windows=2, seed=2)  # 1 batch of 2 per pass
    h5f = h5py.File(p, "r")
    it = tr.cycle_anchor_batches(h5f, np.array([0]), CPU, 2, {})
    try:
        batches = [next(it) for _ in range(3)]  # 3 > 1 batch/pass -> must cycle
        assert len(batches) == 3
        assert all(b[0].shape[0] == 2 for b in batches)
    finally:
        h5f.close()


# --------------------------------------------------------------------------- #
# initialize_model_and_optim_transfer: weight_decay, flanking sizes, scheduler
# --------------------------------------------------------------------------- #
def test_initialize_weight_decay_reaches_optimizer(packaged_80nt_state):
    _m, opt, _s, _p = tr.initialize_model_and_optim_transfer(
        CPU, 80, 10, "MultiStepLR", packaged_80nt_state, unfreeze=1, unfreeze_all=True, weight_decay=0.05)
    assert opt.param_groups and all(g["weight_decay"] == 0.05 for g in opt.param_groups)


@pytest.mark.parametrize("flank,n_units,batch_size", [(80, 4, 36), (400, 8, 36), (2000, 12, 24), (10000, 16, 12)])
def test_initialize_flanking_architecture(packaged_80nt_state, flank, n_units, batch_size):
    """Each flanking size selects the right W/AR length and BATCH_SIZE (the table duplicated
    across subcommands)."""
    _m, _o, _s, params = tr.initialize_model_and_optim_transfer(
        CPU, flank, 10, "MultiStepLR", packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    assert len(params["W"]) == n_units and len(params["AR"]) == n_units
    assert params["BATCH_SIZE"] == batch_size


def test_initialize_cosine_scheduler(packaged_80nt_state):
    _m, _o, sched, _p = tr.initialize_model_and_optim_transfer(
        CPU, 80, 10, "CosineAnnealingWarmRestarts", packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


# --------------------------------------------------------------------------- #
# setup_forgetting_mitigation: each feature in isolation
# --------------------------------------------------------------------------- #
def _params_for_setup(packaged_80nt_state):
    _m, _o, _s, params = tr.initialize_model_and_optim_transfer(
        CPU, 80, 10, "MultiStepLR", packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    params["RANDOM_SEED"] = 42
    return params


def test_setup_genomic_eval_only(tmp_path, packaged_80nt_state):
    params = _params_for_setup(packaged_80nt_state)
    g = str(tmp_path / "g.h5")
    write_dataset_h5(g, n_windows=2, seed=1)
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state, genomic_eval_dataset=g)
    log_base = str(tmp_path / "TRAIN") + "/"
    os.makedirs(log_base, exist_ok=True)
    idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0, 1]), log_base)
    try:
        assert "GENOMIC_H5F" in params and "GENOMIC_METRIC_FILES" in params
        assert "SHARD_TABLE" not in params and "TEACHER" not in params
        assert list(idxs) == [0, 1]  # train_idxs unchanged without rehearsal
    finally:
        for h in handles:
            h.close()


def test_setup_rehearsal_only(tmp_path, packaged_80nt_state):
    params = _params_for_setup(packaged_80nt_state)
    r = str(tmp_path / "r.h5")
    write_dataset_h5(r, n_windows=2, seed=1)
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state, rehearsal_dataset=r, rehearsal_shards=-1)
    idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), str(tmp_path) + "/")
    try:
        assert params["SHARD_TABLE"][0] == ("train", 0)
        assert ("rehearsal", 0) in params["SHARD_TABLE"]
        assert "TEACHER" not in params and "GENOMIC_H5F" not in params
        assert len(idxs) == len(params["SHARD_TABLE"])
    finally:
        for h in handles:
            h.close()


def test_setup_distill_only(tmp_path, packaged_80nt_state):
    params = _params_for_setup(packaged_80nt_state)
    a = str(tmp_path / "a.h5")
    write_dataset_h5(a, n_windows=2, seed=1)
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state, distill_weight=0.5,
                                 distill_shards=a, distill_teacher=None, distill_batch_size=2)
    idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), str(tmp_path) + "/")
    try:
        assert "TEACHER" in params and params["DISTILL_WEIGHT"] == 0.5
        assert "L2SP" not in params                       # l2sp defaults to 0 -> inactive
        assert "SHARD_TABLE" not in params and "GENOMIC_H5F" not in params
        assert list(idxs) == [0]
    finally:
        for h in handles:
            h.close()


def test_setup_l2sp_without_distill_is_inert(tmp_path, packaged_80nt_state):
    """L2-SP only activates alongside a distillation teacher; on its own it is a no-op."""
    params = _params_for_setup(packaged_80nt_state)
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state, l2sp=0.1)  # distill_weight defaults 0
    _idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), str(tmp_path) + "/")
    try:
        assert "L2SP" not in params and "TEACHER" not in params
    finally:
        for h in handles:
            h.close()


def test_setup_rehearsal_shards_capped_to_available(tmp_path, packaged_80nt_state):
    params = _params_for_setup(packaged_80nt_state)
    r = str(tmp_path / "r.h5")
    write_dataset_h5(r, n_windows=2, seed=1)             # only 1 shard available
    args = types.SimpleNamespace(pretrained_model=packaged_80nt_state, rehearsal_dataset=r, rehearsal_shards=10)
    _idxs, handles = tr.setup_forgetting_mitigation(args, params, CPU, np.array([0]), str(tmp_path) + "/")
    try:
        n_rehearsal = sum(1 for s in params["SHARD_TABLE"] if s[0] == "rehearsal")
        assert n_rehearsal == 1                          # min(10, available=1)
    finally:
        for h in handles:
            h.close()


def test_resolve_shard_loader_casts_position_to_int(tmp_path):
    tr_path, re_path = str(tmp_path / "train.h5"), str(tmp_path / "reh.h5")
    _tiny_shard(tr_path, fill=0)
    _tiny_shard(re_path, fill=1)
    train_h5f, reh_h5f = h5py.File(tr_path, "r"), h5py.File(re_path, "r")
    params = {"SHARD_TABLE": [("train", 0), ("rehearsal", 0)], "REHEARSAL_H5F": reh_h5f}
    try:
        xb = next(iter(tbu.resolve_shard_loader(train_h5f, 1.0, CPU, 2, params, shuffle=False)))[0]
        assert torch.all(xb == 1)                        # float 1.0 -> table[1] -> rehearsal handle
    finally:
        train_h5f.close()
        reh_h5f.close()


# --------------------------------------------------------------------------- #
# train_epoch / train_model integration of the new terms
# --------------------------------------------------------------------------- #
def test_train_epoch_focal_loss_with_distillation(tmp_path, model_80nt):
    """The focal-loss primary criterion combines with the distillation term and steps cleanly."""
    files = tbu.create_metric_files(str(tmp_path))
    teacher = copy.deepcopy(model_80nt).eval()
    teacher.requires_grad_(False)
    it, anchor_h5f = _anchor_iter(tmp_path, seed=21)
    p = str(tmp_path / "train.h5")
    write_dataset_h5(p, n_windows=2, seed=22)
    train_h5f = h5py.File(p, "r")
    opt = torch.optim.AdamW(model_80nt.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1], gamma=0.5)
    params = {"CL": 80, "BATCH_SIZE": 2, "N_GPUS": 2, "RANDOM_SEED": 42,
              "TEACHER": teacher, "ANCHOR_ITER": it, "DISTILL_WEIGHT": 0.5}
    try:
        loss, gbi = tbu.train_epoch(model_80nt, train_h5f, np.array([0]), 2, "focal_loss",
                                    opt, sched, CPU, params, files, 80, "train", global_batch_idx=0)
        assert torch.isfinite(loss) and gbi >= 1
    finally:
        train_h5f.close()
        anchor_h5f.close()


def test_train_model_writes_genomic_forgetting_curve(tmp_path, model_80nt):
    """With GENOMIC_* in params, train_model runs a per-epoch genomic eval and logs the
    forgetting curve to its dedicated metric files."""
    def shard(name, seed):
        p = str(tmp_path / f"dataset_{name}.h5")
        write_dataset_h5(p, n_windows=2, seed=seed)
        return h5py.File(p, "r")

    train_h5f, valid_h5f, test_h5f, gen_h5f = (shard(n, s) for n, s in
                                               [("train", 1), ("validation", 2), ("test", 3), ("genomic", 4)])
    model_out = str(tmp_path / "models")
    os.makedirs(model_out, exist_ok=True)
    tmf = tbu.create_metric_files(_mkdir(tmp_path / "TRAIN"))
    vmf = tbu.create_metric_files(_mkdir(tmp_path / "VAL"))
    temf = tbu.create_metric_files(_mkdir(tmp_path / "TEST"))
    gmf = tbu.create_metric_files(_mkdir(tmp_path / "GENOMIC"))
    opt = torch.optim.AdamW(model_80nt.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1], gamma=0.5)
    params = {"CL": 80, "BATCH_SIZE": 2, "N_GPUS": 2, "RANDOM_SEED": 42,
              "GENOMIC_H5F": gen_h5f, "GENOMIC_IDXS": np.array([0]), "GENOMIC_METRIC_FILES": gmf}
    args = types.SimpleNamespace(epochs=1, loss="cross_entropy_loss", flanking_size=80,
                                 early_stopping=False, patience=2)
    try:
        tbu.train_model(model_80nt, opt, sched, train_h5f, valid_h5f, test_h5f,
                        np.array([0]), np.array([0]), np.array([0]), model_out, args, CPU,
                        params, tmf, vmf, temf)
        assert os.path.exists(gmf["donor_auprc"]) and open(gmf["donor_auprc"]).read().strip() != ""
    finally:
        for h in (train_h5f, valid_h5f, test_h5f, gen_h5f):
            h.close()


def _mkdir(path):
    os.makedirs(str(path), exist_ok=True)
    return str(path)
