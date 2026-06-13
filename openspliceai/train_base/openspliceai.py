"""
Filename: train.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Train the OpenSpliceAI model.
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    """A single pre-activation residual block of the SpliceAI dilated CNN.

    Applies (BatchNorm -> LeakyReLU(0.1) -> Conv1d) twice and adds the result
    back to the block's input (identity residual connection). All convolutions
    keep the channel count and temporal length fixed: ``l`` input/output
    channels, kernel width ``w``, and dilation (atrous rate) ``ar`` with
    symmetric ``(w-1)*ar//2`` padding so the output length equals the input
    length. ``forward`` threads through an auxiliary ``y`` (the accumulated skip
    tensor) unchanged, returning ``(x + residual, y)`` so blocks can be chained
    in an ``nn.ModuleList`` alongside :class:`Skip` units.
    """

    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)

    def forward(self, x, y):
        """Return ``(x + residual, y)``, leaving the skip tensor ``y`` untouched.

        ``x`` has shape ``(N, l, length)``; the returned residual has the same
        shape. ``y`` is passed through unchanged for chaining.
        """
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, y


class Cropping1D(nn.Module):
    """Trim a fixed number of timesteps from each end of a ``(N, C, length)`` tensor.

    ``cropping`` is a ``(left, right)`` pair; in SpliceAI both equal ``CL // 2``
    so the flanking context (``CL`` total) is removed and only the central
    ``SL`` prediction window remains. ``forward`` slices the temporal (last)
    dimension, handling ``right == 0`` so no positions are dropped.
    """

    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        """Slice off ``cropping[0]`` timesteps from the left and ``cropping[1]`` from the right."""
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    """A skip connection that accumulates a 1x1-convolved copy of the activations.

    SpliceAI inserts a ``Skip`` after the initial convolution and after every
    4th :class:`ResidualUnit`. Each ``Skip`` adds ``Conv1d(l, l, 1)(x)`` (a
    learned 1x1 projection of the current feature map) into the running skip
    accumulator ``y``, while passing ``x`` through unchanged so the main
    residual trunk continues. The final accumulated ``y`` is what gets cropped
    and projected to the 3 output classes.
    """

    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        """Return ``(x, conv1x1(x) + y)``: trunk unchanged, skip accumulator updated."""
        return x, self.conv(x) + y


class SpliceAI(nn.Module):
    """The SpliceAI deep residual dilated 1-D CNN for splice-site prediction.

    Architecture (Jaganathan et al., 2019), reused by every subcommand:
    an initial ``Conv1d(4, L, 1)`` lifts the 4-channel one-hot DNA input
    (A, C, G, T) to ``L`` channels, followed by a stack of
    :class:`ResidualUnit` blocks parameterised by per-block window sizes ``W``
    and atrous/dilation rates ``AR`` (numpy arrays of equal length: 4/8/12/16
    units for flanking size 80/400/2000/10000). A :class:`Skip` connection is
    inserted before the residual stack and after every 4th residual unit,
    accumulating learned 1x1 projections of the activations. The accumulated
    skip tensor is trimmed by :class:`Cropping1D` (``CL // 2`` timesteps off
    each end, where ``CL = 2 * sum(AR * (W - 1))`` equals the flanking size) so
    only the central prediction window survives, then a final ``Conv1d(L, 3, 1)``
    projects to 3 channels (non-splice, acceptor, donor).

    Args:
        L: number of convolution kernels / hidden channels (32 in practice).
        W: per-residual-unit convolution window sizes (numpy array).
        AR: per-residual-unit atrous/dilation rates (numpy array, same length as W).
        apply_softmax: if True, apply channel-wise softmax to the output; if
            False, return raw logits (used e.g. for temperature calibration).
    """

    def __init__(self, L, W, AR, apply_softmax=True):
        super(SpliceAI, self).__init__()
        self.apply_softmax = apply_softmax  # new parameter to control softmax usage
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r))
            if (i+1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))

    def forward(self, x):
        """Run a one-hot DNA batch through the network.

        Args:
            x: input tensor of shape ``(N, 4, SL + CL)`` -- 4-channel one-hot DNA.

        Returns:
            Tensor of shape ``(N, 3, SL)`` giving per-position class scores
            (non-splice, acceptor, donor); softmax-normalised across the 3
            channels when ``apply_softmax`` is True, otherwise raw logits.
        """
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        final_x = self.crop(skip)
        out = self.final_conv(final_x)
        if self.apply_softmax:
            return F.softmax(out, dim=1)
        else:
            return out
