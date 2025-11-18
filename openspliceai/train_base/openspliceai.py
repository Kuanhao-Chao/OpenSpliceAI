import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openspliceai.rbp.metadata import (
    decode_rbp_metadata,
    encode_rbp_metadata,
)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on RBP vectors."""

    def __init__(self, channels: int, rbp_dim: int):
        super().__init__()
        self.affine = nn.Linear(rbp_dim, channels * 2)
        self.channels = channels

    def forward(self, rbp_batch):
        gamma_beta = self.affine(rbp_batch)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        # reshape for broadcasting across sequence length
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma, beta


class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar, film_dim=None):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2)
        self.film = FiLMLayer(l, film_dim) if film_dim else None

    def forward(self, x, y, rbp_batch=None):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        if self.film is not None and rbp_batch is not None:
            gamma, beta = self.film(rbp_batch)
            out = gamma * out + beta
        return x + out, y


class Cropping1D(nn.Module):
    def __init__(self, cropping):
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        return x[:, :, self.cropping[0]:-self.cropping[1]] if self.cropping[1] > 0 else x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR, apply_softmax=True, film_config=None):
        super(SpliceAI, self).__init__()
        self.apply_softmax = apply_softmax
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        self.film_config = self._normalize_film_config(film_config, len(W))
        self.film_enabled = bool(self.film_config)
        self._warned_missing_rbp = False
        self.residual_units = nn.ModuleList()
        residual_idx = 0
        for i, (w, r) in enumerate(zip(W, AR)):
            film_dim = None
            if self.film_enabled and residual_idx >= self.film_config["film_start"]:
                film_dim = self.film_config["rbp_dim"]
            self.residual_units.append(ResidualUnit(L, w, r, film_dim=film_dim))
            residual_idx += 1
            if (i+1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)
        self.CL = 2 * np.sum(AR * (W - 1))
        self.crop = Cropping1D((self.CL//2, self.CL//2))
        metadata = None
        if self.film_enabled:
            metadata = {
                "rbp_dim": self.film_config["rbp_dim"],
                "rbp_names": self.film_config.get("rbp_names"),
                "film_start": self.film_config["film_start"],
            }
        self.register_buffer("_rbp_metadata_blob", encode_rbp_metadata(metadata))

    def _normalize_film_config(self, film_config, num_residual_units):
        if not film_config:
            return {}
        if "rbp_dim" not in film_config:
            raise ValueError("film_config requires 'rbp_dim'.")
        start = film_config.get("film_start")
        if start is None:
            start = num_residual_units // 2
        if start < 0 or start >= num_residual_units:
            raise ValueError(f"film_start {start} outside valid range (0-{num_residual_units-1}).")
        normalized = dict(film_config)
        normalized["film_start"] = int(start)
        normalized["rbp_dim"] = int(film_config["rbp_dim"])
        return normalized

    def rbp_metadata(self):
        return decode_rbp_metadata(self._rbp_metadata_blob)

    def _prepare_rbp_batch(self, rbp_embedding, batch_size, device):
        if not self.film_enabled:
            return None
        if rbp_embedding is None:
            if not self._warned_missing_rbp:
                warnings.warn(
                    "FiLM-conditioned model received no RBP expression; falling back to unconditioned mode.",
                    RuntimeWarning,
                )
                self._warned_missing_rbp = True
            return None
        if not torch.is_tensor(rbp_embedding):
            rbp_embedding = torch.tensor(rbp_embedding, dtype=torch.float32, device=device)
        else:
            rbp_embedding = rbp_embedding.to(device=device, dtype=torch.float32)
        if rbp_embedding.ndim == 1:
            rbp_embedding = rbp_embedding.unsqueeze(0)
        if rbp_embedding.shape[-1] != self.film_config["rbp_dim"]:
            raise ValueError(
                f"RBP vector dim {rbp_embedding.shape[-1]} does not match model requirement "
                f"{self.film_config['rbp_dim']}."
            )
        if rbp_embedding.size(0) == 1 and batch_size > 1:
            rbp_embedding = rbp_embedding.expand(batch_size, -1)
        elif rbp_embedding.size(0) not in (1, batch_size):
            raise ValueError(
                f"RBP batch dimension {rbp_embedding.size(0)} incompatible with batch size {batch_size}."
            )
        return rbp_embedding

    def forward(self, x, rbp_embedding=None):
        rbp_batch = self._prepare_rbp_batch(rbp_embedding, x.size(0), x.device)
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            if isinstance(m, ResidualUnit):
                x, skip = m(x, skip, rbp_batch)
            else:
                x, skip = m(x, skip)
        final_x = self.crop(skip)
        out = self.final_conv(final_x)
        if self.apply_softmax:
            return F.softmax(out, dim=1)
        else:
            return out
