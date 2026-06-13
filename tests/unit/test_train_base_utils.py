"""Unit tests for pure functions in openspliceai/train_base/utils.py."""
import numpy as np
import torch

import openspliceai.train_base.utils as tbu


def test_categorical_crossentropy_2d_known_value():
    y_true = torch.tensor([[[1.0], [0.0], [0.0]]])     # (1, 3, 1): no-splice
    y_pred = torch.tensor([[[0.5], [0.3], [0.2]]])
    loss = tbu.categorical_crossentropy_2d(y_true, y_pred)
    assert torch.isclose(loss, torch.tensor(-np.log(0.5)).float(), atol=1e-5)


def test_categorical_crossentropy_perfect_prediction_near_zero():
    y = torch.zeros(1, 3, 4)
    y[:, 0, :] = 1.0
    loss = tbu.categorical_crossentropy_2d(y, y.clone())
    assert loss.item() < 1e-3


def test_focal_loss_known_issue_gamma_ignored():
    """KNOWN ISSUE (documented, not fixed): focal_loss hardcodes gamma=2 and ignores the
    gamma argument. This characterization test pins the current behavior; flip it once the
    behavior-changing fix is applied."""
    torch.manual_seed(0)
    y_true = torch.zeros(2, 3, 5)
    y_true[:, 0, :] = 1.0
    y_pred = torch.softmax(torch.randn(2, 3, 5), dim=1)
    assert torch.isclose(
        tbu.focal_loss(y_true, y_pred, gamma=0.0),
        tbu.focal_loss(y_true, y_pred, gamma=2.0),
    )


def test_classwise_accuracy():
    true = np.array([0, 0, 1, 2, 2])
    pred = np.array([0, 1, 1, 2, 0])
    assert tbu.classwise_accuracy(true, pred, 3) == [0.5, 1.0, 0.5]


def test_classwise_accuracy_absent_class_is_zero():
    true = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    # classes 1 and 2 have no samples -> accuracy 0.0
    assert tbu.classwise_accuracy(true, pred, 3) == [1.0, 0.0, 0.0]


def test_threshold_predictions_is_strict():
    out = tbu.threshold_predictions(np.array([0.4, 0.6, 0.5]), threshold=0.5)
    np.testing.assert_array_equal(out, [0, 1, 0])   # 0.5 is NOT > 0.5


def test_clip_datapoints_even_batch():
    X = torch.zeros(4, 4, 15000)
    Y = torch.zeros(4, 3, 5000)
    Xc, Yc = tbu.clip_datapoints(X, Y, CL=80, CL_max=10000, N_GPUS=2)
    # clip = (10000-80)//2 = 4960 -> X spatial 15000-9920 = 5080 ; Y untouched
    assert Xc.shape == (4, 4, 5080)
    assert Yc.shape == (4, 3, 5000)


def test_clip_datapoints_drops_remainder_for_gpu_alignment():
    X = torch.zeros(5, 4, 15000)
    Y = torch.zeros(5, 3, 5000)
    Xc, Yc = tbu.clip_datapoints(X, Y, CL=80, CL_max=10000, N_GPUS=2)
    # 5 % 2 == 1 -> drop one sample from the batch dim
    assert Xc.shape == (4, 4, 5080)
    assert Yc.shape == (4, 3, 5000)
