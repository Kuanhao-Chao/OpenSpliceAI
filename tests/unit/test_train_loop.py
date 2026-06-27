"""Direct coverage for the training/validation loop in train_base/utils.py.

The integration smoke test runs train.train end-to-end with the default cross-entropy loss
and ``early_stopping=False``. These tests drive the loop functions directly on the tiny
``model_80nt`` fixture + a 4-window shard to cover the complementary branches: the
``focal_loss`` path in train_epoch/valid_epoch and the ``early_stopping=True`` bookkeeping
in train_model, plus the thin ``test_model`` wrapper (reachable only via the disabled
``test`` subcommand but pure enough to pin)."""
import types

import numpy as np
import pytest
import torch

import openspliceai.train_base.utils as tbu
from tests.fixtures.synthetic import write_dataset_h5

BATCH = 2
PARAMS = {"CL": 80, "BATCH_SIZE": BATCH, "N_GPUS": 2, "RANDOM_SEED": 42}


def _shard(tmp_path, name, n=4):
    p = str(tmp_path / f"dataset_{name}.h5")
    write_dataset_h5(p, n_windows=n, seed={"train": 1, "validation": 2, "test": 3}[name])
    import h5py
    return h5py.File(p, "r")


def _scheduler(optimizer, epochs=2):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs - 1], gamma=0.5)


@pytest.mark.parametrize("criterion", ["cross_entropy_loss", "focal_loss"])
def test_valid_epoch_returns_scalar(tmp_path, model_80nt, criterion):
    files = tbu.create_metric_files(str(tmp_path))
    h5f = _shard(tmp_path, "validation")
    try:
        loss = tbu.valid_epoch(model_80nt, h5f, np.array([0]), BATCH, criterion,
                               torch.device("cpu"), PARAMS, files, 80, "validation")
    finally:
        h5f.close()
    assert torch.is_tensor(loss) and torch.isfinite(loss)


@pytest.mark.parametrize("criterion", ["cross_entropy_loss", "focal_loss"])
def test_train_epoch_steps_optimizer_and_scheduler(tmp_path, model_80nt, criterion):
    files = tbu.create_metric_files(str(tmp_path))
    opt = torch.optim.AdamW(model_80nt.parameters(), lr=1e-3)
    sched = _scheduler(opt)
    h5f = _shard(tmp_path, "train")
    try:
        loss, gbi = tbu.train_epoch(model_80nt, h5f, np.array([0]), BATCH, criterion,
                                    opt, sched, torch.device("cpu"), PARAMS, files, 80,
                                    "train", global_batch_idx=0)
    finally:
        h5f.close()
    assert torch.isfinite(loss)
    assert gbi >= 1   # at least one batch processed -> global_batch_idx advanced


def test_test_model_wrapper_runs_valid_epoch(tmp_path, model_80nt):
    files = tbu.create_metric_files(str(tmp_path))
    opt = torch.optim.AdamW(model_80nt.parameters(), lr=1e-3)
    h5f = _shard(tmp_path, "test")
    args = types.SimpleNamespace(loss="cross_entropy_loss", flanking_size=80)
    try:
        tbu.test_model(model_80nt, opt, h5f, np.array([0]), args, torch.device("cpu"),
                       PARAMS, files)   # returns None, asserts it runs without error
    finally:
        h5f.close()
    assert (tmp_path).exists()


@pytest.mark.parametrize("early_stopping", [True, False])
def test_train_model_checkpointing_and_early_stop_bookkeeping(tmp_path, model_80nt, monkeypatch, early_stopping):
    """Drive train_model's per-epoch loop deterministically: train_epoch/valid_epoch are
    stubbed so the validation loss strictly *increases*, so only epoch 0 improves. With
    early_stopping=True this triggers the patience break; with False it runs all epochs.
    Pins checkpointing (model_{epoch}.pt + model_best.pt) and the early-stop counter."""
    out = tmp_path / "m"
    out.mkdir()
    (tmp_path / "TRAIN").mkdir()
    tf = tbu.create_metric_files(str(tmp_path / "TRAIN"))

    calls = {"n": 0}

    def fake_valid(*a, **k):
        calls["n"] += 1
        return torch.tensor(float(calls["n"]))   # strictly increasing -> only epoch 0 is "best"

    def fake_train(*a, **k):
        return torch.tensor(0.5), k.get("global_batch_idx", 0) + 1

    monkeypatch.setattr(tbu, "valid_epoch", fake_valid)
    monkeypatch.setattr(tbu, "train_epoch", fake_train)

    opt = torch.optim.AdamW(model_80nt.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5], gamma=0.5)
    args = types.SimpleNamespace(epochs=3, loss="cross_entropy_loss", flanking_size=80,
                                 early_stopping=early_stopping, patience=1)

    tbu.train_model(model_80nt, opt, sched, None, None, None, np.array([0]), np.array([0]),
                    np.array([0]), str(out), args, torch.device("cpu"), PARAMS, tf, tf, tf)

    assert (out / "model_0.pt").exists() and (out / "model_best.pt").exists()
    if early_stopping:
        # patience=1 + non-improving epoch 1 -> stop before writing model_2.pt
        assert not (out / "model_2.pt").exists()
    else:
        assert (out / "model_2.pt").exists()
