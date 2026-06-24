"""Additional unit coverage for openspliceai/train_base/utils.py.

Covers the pure/IO helpers not exercised by the train smoke test: both clip helpers'
full branch matrix, the metric accumulators and writers, model_evaluation's
cross-entropy/focal branches and >1000-window subsampling, print_topl_statistics edge
warnings, and the path/dataset helpers (including the ones reachable only via the disabled
``test`` subcommand, which are still pure and worth pinning)."""
import os
import types

import h5py
import numpy as np
import pytest
import torch

import openspliceai.train_base.utils as tbu
from tests.fixtures.synthetic import write_calibrate_datasets, write_dataset_h5


# --- clip_datapoints: the two clip==0 branches (CL == CL_max) ------------------------

def test_clip_datapoints_no_clip_even_batch():
    X = torch.zeros(4, 4, 5000)
    Y = torch.zeros(4, 3, 5000)
    Xc, Yc = tbu.clip_datapoints(X, Y, CL=10000, CL_max=10000, N_GPUS=2)  # clip=0, rem=0
    assert Xc.shape == (4, 4, 5000) and Yc.shape == (4, 3, 5000)


def test_clip_datapoints_no_clip_drops_remainder():
    X = torch.zeros(5, 4, 5000)
    Y = torch.zeros(5, 3, 5000)
    Xc, Yc = tbu.clip_datapoints(X, Y, CL=10000, CL_max=10000, N_GPUS=2)  # clip=0, rem=1
    assert Xc.shape == (4, 4, 5000) and Yc.shape == (4, 3, 5000)


# --- clip_datapoints_spliceai27 (Keras-shaped: X is (N, L), Y is a 1-list) -----------

@pytest.mark.parametrize("n,cl,exp_n,exp_l", [
    (4, 80, 4, 5080),     # rem==0, clip!=0
    (5, 80, 4, 5080),     # rem!=0, clip!=0
    (4, 10000, 4, 15000),  # rem==0, clip==0
    (5, 10000, 4, 15000),  # rem!=0, clip==0
])
def test_clip_datapoints_spliceai27_branches(n, cl, exp_n, exp_l):
    X = np.zeros((n, 15000))
    Y = [np.zeros((n, 5000))]
    Xc, Yc = tbu.clip_datapoints_spliceai27(X, Y, cl, N_GPUS=2)
    assert Xc.shape == (exp_n, exp_l)
    assert Yc[0].shape[0] == exp_n


# --- losses / metric helpers ---------------------------------------------------------

def test_weighted_binary_cross_entropy_weighted_and_unweighted():
    out = torch.tensor([0.7, 0.2, 0.9])
    tgt = torch.tensor([1.0, 0.0, 1.0])
    unweighted = tbu.weighted_binary_cross_entropy(out, tgt)
    weighted = tbu.weighted_binary_cross_entropy(out, tgt, weights=[1.0, 2.0])
    assert torch.isfinite(unweighted) and torch.isfinite(weighted)
    assert unweighted.item() > 0


def test_weighted_binary_cross_entropy_rejects_bad_weights():
    with pytest.raises(AssertionError):
        tbu.weighted_binary_cross_entropy(torch.tensor([0.5]), torch.tensor([1.0]), weights=[1.0])


def test_calculate_metrics_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, f1, acc = tbu.calculate_metrics(y_true, y_pred)
    assert acc == pytest.approx(0.8)
    assert 0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0


def test_metrics_accumulator_overall_metrics():
    acc = tbu.MetricsAccumulator(3)
    acc.update(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    acc.update(np.array([2, 2]), np.array([2, 0]))
    overall, precision, recall, f1, class_acc = acc.calculate_overall_metrics()
    assert len(class_acc) == 3 and len(precision) == 3
    assert 0.0 <= overall <= 1.0


def test_calculate_batch_metrics_feeds_accumulator():
    acc = tbu.MetricsAccumulator(3)
    tbu.calculate_batch_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), {}, acc)
    assert len(acc.true_classes) == 3 and len(acc.predicted_classes) == 3


def test_metrics_writes_accuracy_file(tmp_path):
    files = tbu.create_metric_files(str(tmp_path))
    yl = torch.zeros(4, 3, 8)
    yl[:, 0, :] = 1.0
    yp = torch.softmax(torch.randn(4, 3, 8), dim=1)
    tbu.metrics(yp, yl, files, "validation")
    assert os.path.getsize(files["accuracy"]) > 0


def _expressed_label_pred(n, length, rng):
    """Build (label, pred) tensors where every window has an acceptor & donor site."""
    yl = torch.zeros(n, 3, length)
    yl[:, 0, :] = 1.0
    yl[:, 0, 1], yl[:, 1, 1] = 0.0, 1.0   # acceptor at index 1
    yl[:, 0, 2], yl[:, 2, 2] = 0.0, 1.0   # donor at index 2
    yp = torch.softmax(torch.tensor(rng.standard_normal((n, 3, length)), dtype=torch.float32), dim=1)
    return yl, yp


@pytest.mark.parametrize("criterion", ["cross_entropy_loss", "focal_loss"])
def test_model_evaluation_returns_loss_and_writes(tmp_path, criterion):
    files = tbu.create_metric_files(str(tmp_path))
    rng = np.random.default_rng(0)
    yl, yp = _expressed_label_pred(6, 16, rng)
    loss = tbu.model_evaluation([yl], [yp], files, "validation", criterion)
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    assert os.path.getsize(files["loss_batch"]) > 0


def test_model_evaluation_subsamples_above_1000_windows(tmp_path):
    files = tbu.create_metric_files(str(tmp_path))
    rng = np.random.default_rng(1)
    yl, yp = _expressed_label_pred(1100, 4, rng)   # > 1000 expressed -> hits the subset cap
    loss = tbu.model_evaluation([yl], [yp], files, "test", "cross_entropy_loss")
    assert torch.isfinite(loss)


# --- print_topl_statistics edge branches ---------------------------------------------

def test_print_topl_statistics_warns_when_requested_exceeds_size(tmp_path, capsys):
    out = tmp_path / "topk.txt"
    y_true = np.array([1, 1, 1, 0], dtype=float)   # 3 trues; top_length=4 -> 12 > 4 -> warn
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])
    topk, auprc = tbu.print_topl_statistics(y_true, y_pred, str(out), ss_type="donor", print_top_k=True)
    assert 0.0 <= topk <= 1.0 and 0.0 <= auprc <= 1.0
    assert "exceeds y_pred size" in capsys.readouterr().out
    assert out.exists()


# --- device / environment ------------------------------------------------------------

def test_setup_device_returns_torch_device():
    dev = tbu.setup_device()
    assert dev.type in ("cpu", "cuda", "mps")


def test_setup_environment_accepts_valid_flank_and_rejects_invalid():
    assert tbu.setup_environment(types.SimpleNamespace(flanking_size=80)).type in ("cpu", "cuda", "mps")
    with pytest.raises(AssertionError):
        tbu.setup_environment(types.SimpleNamespace(flanking_size=123))


# --- path + dataset helpers ----------------------------------------------------------

def test_initialize_paths_inner_creates_tree(tmp_path):
    model_base, tr, val, te = tbu.initialize_paths_inner(
        str(tmp_path), "proj", 80, "0", 5000, "cross_entropy_loss", 42)
    for p in (model_base, tr, val, te):
        assert os.path.isdir(p)
    assert "SpliceAI_proj_80_0_rs42" in model_base


def test_initialize_paths_wrapper_from_args(tmp_path):
    args = types.SimpleNamespace(output_dir=str(tmp_path), project_name="proj", flanking_size=80,
                                 exp_num="0", loss="cross_entropy_loss", random_seed=42)
    model_base, tr, val, te = tbu.initialize_paths(args)
    for p in (model_base, tr, val, te):
        assert os.path.isdir(p)


def test_initialize_test_paths_creates_test_dir(tmp_path):
    args = types.SimpleNamespace(output_dir=str(tmp_path), project_name="proj", flanking_size=80,
                                 exp_num="0", random_seed=42, test_target="MANE", log_dir="LOG")
    test_base = tbu.initialize_test_paths(args)
    assert os.path.isdir(test_base) and test_base.endswith("TEST/")


def test_create_metric_files_returns_all_named_paths(tmp_path):
    files = tbu.create_metric_files(str(tmp_path))
    for key in ("loss_batch", "accuracy", "acceptor_topk_all", "donor_auprc"):
        assert key in files and files[key].endswith(f"{key}.txt")


def test_generate_indices_and_test_indices(tmp_path):
    paths = write_calibrate_datasets(tmp_path, n_windows=2)
    with h5py.File(paths["train"], "r") as tr, h5py.File(paths["validation"], "r") as va, \
            h5py.File(paths["test"], "r") as te:
        train_idxs, val_idxs, test_idxs = tbu.generate_indices(tr, va, te)
        assert list(train_idxs) == [0] and list(val_idxs) == [0] and list(test_idxs) == [0]
        ti = tbu.generate_test_indices(42, te)
        assert list(ti) == [0]


def test_load_datasets_derives_validation_path(tmp_path):
    paths = write_calibrate_datasets(tmp_path, n_windows=2)
    args = types.SimpleNamespace(train_dataset=paths["train"], test_dataset=paths["test"])
    train_h5f, valid_h5f, test_h5f, batch_num = tbu.load_datasets(args)
    try:
        assert batch_num == 1
        assert "X0" in train_h5f and "X0" in valid_h5f and "X0" in test_h5f
    finally:
        train_h5f.close()
        valid_h5f.close()
        test_h5f.close()


def test_load_test_datasets_opens_file(tmp_path):
    p = str(tmp_path / "dataset_test.h5")
    write_dataset_h5(p, n_windows=2)
    args = types.SimpleNamespace(test_dataset=p)
    h5f = tbu.load_test_datasets(args)
    try:
        assert "X0" in h5f
    finally:
        h5f.close()
