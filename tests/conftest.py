"""Shared pytest fixtures and collection hooks for the OpenSpliceAI test suite.

Unit + regression tests are CPU-only and free of TensorFlow/Keras. Tests that need a
GPU or Keras are marked accordingly and auto-skipped when those are unavailable.
"""
import os

# Force a non-interactive matplotlib backend and CPU-only torch before anything imports them.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import random
from pathlib import Path

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip keras/gpu-marked tests when those backends are unavailable."""
    import importlib.util
    have_tf = importlib.util.find_spec("tensorflow") is not None   # avoid the slow TF import
    try:
        import torch
        have_cuda = torch.cuda.is_available()
    except Exception:
        have_cuda = False

    skip_keras = pytest.mark.skip(reason="tensorflow/keras not installed")
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "keras" in item.keywords and not have_tf:
            item.add_marker(skip_keras)
        if "gpu" in item.keywords and not have_cuda:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed_everything():
    """Make every test deterministic."""
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except Exception:
        pass
    yield


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def model_80nt():
    """A fresh, untrained 80nt SpliceAI model on CPU (eval mode)."""
    import torch  # noqa: F401
    from openspliceai.train_base.openspliceai import SpliceAI
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    return SpliceAI(32, W, AR).to("cpu").eval()


@pytest.fixture(scope="session")
def packaged_80nt_state(repo_root):
    """Path to a real packaged 80nt checkpoint (state_dict), for integration tests."""
    p = repo_root / "models" / "openspliceai-honeybee" / "80nt" / "model_80nt_rs10.pt"
    if not p.exists():
        pytest.skip(f"packaged 80nt checkpoint not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def packaged_80nt_dir(repo_root):
    """Path to a directory of packaged 80nt checkpoints (for ensemble tests)."""
    p = repo_root / "models" / "openspliceai-honeybee" / "80nt"
    if not p.exists():
        pytest.skip(f"packaged 80nt model dir not found: {p}")
    return str(p)


@pytest.fixture
def calibrate_datasets(tmp_path):
    """Write tiny dataset_{train,validation,test}.h5 (2 windows each) and return paths."""
    from tests.fixtures.synthetic import write_calibrate_datasets
    return write_calibrate_datasets(tmp_path)


@pytest.fixture
def train_datasets(tmp_path):
    """dataset_{train,validation,test}.h5 with 40 windows each (>= batch size) for 1-epoch e2e."""
    from tests.fixtures.synthetic import write_train_datasets
    return write_train_datasets(tmp_path, n_windows=40)


@pytest.fixture
def mini_genome_gff(tmp_path):
    """(fasta_path, gff_path) for a small 2-chromosome genome that drives the create-data CLI."""
    from tests.fixtures.synthetic import write_mini_genome_and_gff
    return write_mini_genome_and_gff(tmp_path)


@pytest.fixture
def variant_inputs(tmp_path):
    """(ref_path, ann_path, vcf_path) for variant e2e: ~12kb ref + custom TSV + SNV/indel/multiallelic VCF."""
    from tests.fixtures.synthetic import write_variant_inputs
    return write_variant_inputs(tmp_path)
