"""Unit tests for pure functions in openspliceai/create_data/utils.py."""
import warnings
from collections import namedtuple

import numpy as np
import pytest

import openspliceai.create_data.utils as cdu

Rec = namedtuple("Rec", ["seq"])


def test_ceil_div():
    assert cdu.ceil_div(10, 3) == 4
    assert cdu.ceil_div(9, 3) == 3
    assert cdu.ceil_div(0, 5) == 0
    assert cdu.ceil_div(1, 1) == 1


def test_replace_non_acgt_to_n():
    assert cdu.replace_non_acgt_to_n("ACGT") == "ACGT"
    assert cdu.replace_non_acgt_to_n("acgt") == "NNNN"          # lowercase -> N
    assert cdu.replace_non_acgt_to_n("acgtN") == "NNNNN"        # N is not in the allowed set
    assert cdu.replace_non_acgt_to_n("ACGTXY") == "ACGTNN"
    assert cdu.replace_non_acgt_to_n("") == ""


def test_one_hot_encode_in_out_maps():
    Xd = np.array([[1, 2, 3, 4, 0]])               # A,C,G,T,pad
    Yd = [np.array([[0, 1, 2, 3]])]                 # no-splice, acceptor, donor, pad
    X, Y = cdu.one_hot_encode(Xd, Yd)
    assert X.shape == (1, 5, 4)
    np.testing.assert_array_equal(X[0, 0], [1, 0, 0, 0])   # A
    np.testing.assert_array_equal(X[0, 1], [0, 1, 0, 0])   # C
    np.testing.assert_array_equal(X[0, 4], [0, 0, 0, 0])   # padding
    np.testing.assert_array_equal(
        Y[0][0], [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )


def test_reformat_data_windows(monkeypatch):
    monkeypatch.setattr(cdu, "SL", 10)
    monkeypatch.setattr(cdu, "CL_max", 4)
    X0 = np.arange(25)
    Y0 = [np.arange(25)]
    Xd, Yd = cdu.reformat_data(X0, Y0)
    # num_points = ceil(25/10) = 3 ; X window = SL + CL_max = 14 ; Y window = SL = 10
    assert Xd.shape == (3, 14)
    assert Yd[0].shape == (3, 10)


def test_create_datapoints_shape_and_padding(monkeypatch):
    monkeypatch.setattr(cdu, "SL", 10)
    monkeypatch.setattr(cdu, "CL_max", 4)
    X, Y = cdu.create_datapoints("ACGTACGTAC", "0120000210")
    assert X.shape == (1, 14, 4)        # (num_points, SL+CL_max, 4)
    assert Y[0].shape == (1, 10, 3)     # (num_points, SL, 3)
    # CL_max//2 = 2 'N' padding rows at each end are all-zero
    assert X[0, :2].sum() == 0
    assert X[0, -2:].sum() == 0
    # interior is correct one-hot (first real base 'A', second 'C')
    np.testing.assert_array_equal(X[0, 2], [1, 0, 0, 0])
    np.testing.assert_array_equal(X[0, 3], [0, 1, 0, 0])


def test_split_chromosomes_human_named():
    seq_dict = {c: Rec("A" * 100) for c in ["chr1", "chr2", "chr3", "chr7", "chr8"]}
    train, test = cdu.split_chromosomes(seq_dict, method="human")
    # SpliceAI human split: test = chr1,3,5,7,9 ; everything else trains
    assert "chr1" in test and "chr3" in test and "chr7" in test
    assert "chr2" in train and "chr8" in train


def test_split_chromosomes_human_warns_on_nonstandard_names(recwarn):
    seq_dict = {"1": Rec("A" * 100), "2": Rec("A" * 50)}   # no 'chr' prefix
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        cdu.split_chromosomes(seq_dict, method="human")
    assert any("UCSC-style chromosome names" in str(w.message) for w in recwarn.list)


def test_split_chromosomes_random_partition_is_complete():
    seq_dict = {f"c{i}": Rec("A" * (100 * (i + 1))) for i in range(6)}
    train, test = cdu.split_chromosomes(seq_dict, method="random", split_ratio=0.7)
    assert set(train) | set(test) == set(seq_dict)
    assert set(train).isdisjoint(set(test))


def test_split_chromosomes_invalid_method():
    with pytest.raises(ValueError):
        cdu.split_chromosomes({"chr1": Rec("A" * 10)}, method="bogus")


def test_split_train_val_counts_and_completeness():
    # 7 parallel fields (NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL)
    data = [list(range(10)) for _ in range(7)]
    train, val = cdu.split_train_val(data, val_split_ratio=0.2)
    assert len(val[0]) == 2 and len(train[0]) == 8
    # no record is lost or duplicated across the split
    assert set(train[0]) | set(val[0]) == set(range(10))
    assert set(train[0]).isdisjoint(set(val[0]))
