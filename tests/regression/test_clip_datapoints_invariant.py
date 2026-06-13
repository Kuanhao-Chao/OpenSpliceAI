"""Regression lock for a FALSE-POSITIVE audit finding.

An audit flagged clip_datapoints as buggy for cropping X spatially but not Y. It is actually
CORRECT: Y is stored at length SL while X is stored at length SL+CL_max. clip_datapoints
trims X to SL+CL, and the model's internal Cropping1D then removes the remaining CL, yielding
SL == the Y length. This test pins that invariant end-to-end.

Uses a scaled-down SL=200, CL_max=280 (mirrors the real SL=5000, CL_max=10000 relationship)
so the model forward pass is cheap.
"""
import numpy as np
import torch

from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.train_base.utils import clip_datapoints


def test_clip_then_model_output_matches_label_length():
    SL, CL, CL_max = 200, 80, 280
    X = torch.randn(2, 4, SL + CL_max)     # 480 wide, like the stored datapoints
    Y = torch.zeros(2, 3, SL)              # 200 wide, like the stored labels

    Xc, Yc = clip_datapoints(X, Y, CL=CL, CL_max=CL_max, N_GPUS=2)
    assert Xc.shape == (2, 4, SL + CL)     # X trimmed to SL+CL = 280
    assert Yc.shape == (2, 3, SL)          # Y is left untouched at SL = 200

    model = SpliceAI(32, np.asarray([11, 11, 11, 11]), np.asarray([1, 1, 1, 1])).to("cpu").eval()
    with torch.no_grad():
        out = model(Xc)
    # model's Cropping1D removes the remaining CL, so output length == Y length
    assert out.shape == Yc.shape == (2, 3, SL)
