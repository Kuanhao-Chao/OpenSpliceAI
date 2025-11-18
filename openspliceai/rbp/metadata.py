"""
Helpers for persisting FiLM/RBP configuration inside SpliceAI checkpoints.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import torch


def _ensure_tensor(blob: bytes) -> torch.Tensor:
    """Convert a bytes payload to a uint8 tensor."""
    if not blob:
        return torch.zeros(0, dtype=torch.uint8)
    data = list(blob)
    return torch.tensor(data, dtype=torch.uint8)


def encode_rbp_metadata(metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
    """
    Serialize a metadata dictionary into a tensor that can be stored as a buffer.
    """
    metadata = metadata or {}
    payload = json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return _ensure_tensor(payload)


def decode_rbp_metadata(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    """
    Deserialize metadata stored in a buffer tensor.
    """
    if tensor is None or tensor.numel() == 0:
        return {}
    if tensor.dtype != torch.uint8:
        raise ValueError("RBP metadata buffer must use dtype uint8.")
    payload = bytes(tensor.tolist())
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))


def extract_film_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Inspect a serialized model state_dict and return FiLM configuration hints.
    """
    blob = state_dict.get("_rbp_metadata_blob")
    metadata = decode_rbp_metadata(blob)
    rbp_dim = metadata.get("rbp_dim")
    if not rbp_dim:
        return {}
    config = {
        "rbp_dim": rbp_dim,
        "rbp_names": metadata.get("rbp_names"),
        "film_start": metadata.get("film_start"),
    }
    return {k: v for k, v in config.items() if v is not None}
