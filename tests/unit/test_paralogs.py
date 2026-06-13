"""Unit tests for openspliceai/create_data/paralogs.py.

The whole module needs minimap2's python binding (``mappy``); if it is missing the
entire file is skipped. Expected values were grounded by running the functions on the
synthetic sequences below before asserting.
"""
import os

import h5py
import numpy as np
import pytest

# Skip the entire module if minimap2's python binding isn't installed.
pytest.importorskip("mappy")

import openspliceai.create_data.paralogs as paralogs


def _make_record(name, seq):
    """Build one 7-field column record [NAME,CHROM,STRAND,TX_START,TX_END,SEQ,LABEL]."""
    return [name, "chr1", "+", "1", str(len(seq)), seq, "0" * len(seq)]


def _columns(*records):
    """Transpose a list of 7-field records into the column-major list-of-lists layout."""
    return [list(col) for col in zip(*records)]


@pytest.mark.slow
def test_remove_paralogous_sequences_drops_shared_keeps_unique(tmp_path):
    rng = np.random.default_rng(0)

    def rseq(n):
        return "".join(rng.choice(list("ACGT"), size=n))

    shared = rseq(500)   # identical sequence present in BOTH train and test
    unique = rseq(500)   # present only in test, unrelated to the training sequence

    train_data = _columns(_make_record("t0", shared))
    test_data = _columns(
        _make_record("x_shared", shared),
        _make_record("x_unique", unique),
    )

    filtered_train, filtered_test = paralogs.remove_paralogous_sequences(
        train_data, test_data,
        min_identity=0.8, min_coverage=0.5,
        output_dir=str(tmp_path), exp="test",
    )

    # The shared sequence is paralogous to training -> removed; the unique one survives.
    assert filtered_test[0] == ["x_unique"]
    assert filtered_test[5] == [unique]
    # Train side is returned unchanged.
    assert filtered_train[0] == ["t0"]
    # A removed-paralogs log is written for this experiment name.
    assert os.path.exists(os.path.join(str(tmp_path), "removed_paralogs_test.txt"))


def test_write_h5_file_roundtrip(tmp_path):
    data = [
        ["g0", "g1"],            # NAME
        ["chr1", "chr2"],        # CHROM
        ["+", "-"],              # STRAND
        ["1", "100"],            # TX_START
        ["10", "120"],           # TX_END
        ["ACGTACGTAC", "TTGGCCAA"],  # SEQ
        ["0120000210", "00120000"],  # LABEL
    ]
    paralogs.write_h5_file(str(tmp_path), "train", data)

    h5path = os.path.join(str(tmp_path), "datafile_train.h5")
    assert os.path.exists(h5path)
    with h5py.File(h5path, "r") as f:
        assert sorted(f.keys()) == [
            "CHROM", "LABEL", "NAME", "SEQ", "STRAND", "TX_END", "TX_START",
        ]
        assert f["NAME"][0].decode() == "g0"
        assert f["SEQ"][1].decode() == "TTGGCCAA"
        assert f["STRAND"][1].decode() == "-"
        assert len(f["SEQ"]) == 2


def test_write_h5_file_no_trailing_slash_path(tmp_path):
    """output_dir without a trailing slash must still produce datafile_<type>.h5.

    (paralogs.write_h5_file uses os.path.join, so this is the safe/expected behavior.)
    """
    out = os.path.join(str(tmp_path), "out")
    os.makedirs(out, exist_ok=True)
    data = [["g0"], ["chr1"], ["+"], ["1"], ["4"], ["ACGT"], ["0000"]]
    paralogs.write_h5_file(out, "test", data)
    assert os.path.exists(os.path.join(out, "datafile_test.h5"))
