"""Unit tests for pure helpers in openspliceai/variant/utils.py (TensorFlow-free)."""
import inspect
import os

import numpy as np

import openspliceai.variant.utils as vu


def test_normalise_chrom_strip_prefix():
    assert vu.normalise_chrom("chr1", "1") == "1"


def test_normalise_chrom_add_prefix():
    assert vu.normalise_chrom("1", "chr1") == "chr1"


def test_normalise_chrom_noop_when_consistent():
    assert vu.normalise_chrom("chr1", "chr1") == "chr1"
    assert vu.normalise_chrom("1", "2") == "1"


def test_one_hot_encode_acgtn():
    oh = vu.one_hot_encode("ACGTN")
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])
    np.testing.assert_array_equal(oh, expected)


def test_one_hot_encode_lowercase_upcased():
    np.testing.assert_array_equal(vu.one_hot_encode("acgt"), vu.one_hot_encode("ACGT"))


def test_builtin_annotation_paths_resolve_to_existing_files():
    for name in ("grch37", "grch38"):
        path = vu._resolve_builtin_annotation(name)
        assert os.path.exists(path), f"packaged annotation missing: {path}"


def test_load_keras_models_error_message_says_keras():
    src = inspect.getsource(vu.load_keras_models)
    assert "No valid Keras models found" in src
    assert "No valid PyTorch models found" not in src
