"""Integration tests for openspliceai/merge_data/merge_dataset.py.

merge_dataset(args) concatenates the X*/Y* shards from several input directories' dataset_{split}.h5
files into one merged dataset_{split}.h5 under args.output_dir, re-keying X0,X1,... sequentially.

args namespace (read from merge_dataset.py):
  - chr_split: 'test' (only test split) or 'train-test' (both)
  - input_dir: list of directories, each containing dataset_{split}.h5
  - output_dir: directory to write merged dataset_{split}.h5

Expected shapes/keys were grounded by running merge_dataset on the synthetic fixtures first.
"""
import os
import types

import h5py
import pytest

from tests.fixtures.synthetic import write_dataset_h5


@pytest.mark.integration
def test_merge_dataset_combines_shards_from_two_dirs(tmp_path):
    in1 = tmp_path / "in1"
    in2 = tmp_path / "in2"
    out = tmp_path / "out"
    in1.mkdir()
    in2.mkdir()

    # Each input dir contributes one shard (X0/Y0) for the 'test' split.
    write_dataset_h5(str(in1 / "dataset_test.h5"), n_windows=2, seed=1)
    write_dataset_h5(str(in2 / "dataset_test.h5"), n_windows=3, seed=2)

    args = types.SimpleNamespace(
        chr_split="test",
        input_dir=[str(in1), str(in2)],
        output_dir=str(out),
    )
    merged_path = out / "dataset_test.h5"

    from openspliceai.merge_data.merge_dataset import merge_dataset

    merge_dataset(args)

    assert merged_path.exists()
    with h5py.File(str(merged_path), "r") as f:
        keys = sorted(f.keys())
        # Two input shards (one per dir) -> X0,X1,Y0,Y1, sequentially re-keyed.
        assert keys == ["X0", "X1", "Y0", "Y1"]
        # First shard came from in1 (2 windows), second from in2 (3 windows).
        # X is (n, SL+CL_max, 4); Y is (1, n, SL, 3).
        assert f["X0"].shape[0] == 2
        assert f["X1"].shape[0] == 3
        assert f["X0"].shape[1:] == (15000, 4)
        assert f["X1"].shape[1:] == (15000, 4)
        assert f["Y0"].shape == (1, 2, 5000, 3)
        assert f["Y1"].shape == (1, 3, 5000, 3)


@pytest.mark.integration
def test_merge_dataset_preserves_data_values(tmp_path):
    in1 = tmp_path / "in1"
    out = tmp_path / "out"
    in1.mkdir()
    write_dataset_h5(str(in1 / "dataset_test.h5"), n_windows=2, seed=7)

    args = types.SimpleNamespace(
        chr_split="test",
        input_dir=[str(in1)],
        output_dir=str(out),
    )

    from openspliceai.merge_data.merge_dataset import merge_dataset

    merge_dataset(args)

    src = in1 / "dataset_test.h5"
    dst = out / "dataset_test.h5"
    with h5py.File(str(src), "r") as fs, h5py.File(str(dst), "r") as fd:
        # Single dir, single shard -> X0/Y0 copied verbatim.
        assert (fs["X0"][:] == fd["X0"][:]).all()
        assert (fs["Y0"][:] == fd["Y0"][:]).all()


@pytest.mark.integration
def test_merge_dataset_train_test_writes_both_splits(tmp_path):
    in1 = tmp_path / "in1"
    out = tmp_path / "out"
    in1.mkdir()
    for split in ("test", "train"):
        write_dataset_h5(str(in1 / f"dataset_{split}.h5"), n_windows=2, seed=3)

    args = types.SimpleNamespace(
        chr_split="train-test",
        input_dir=[str(in1)],
        output_dir=str(out),
    )

    from openspliceai.merge_data.merge_dataset import merge_dataset

    merge_dataset(args)

    for split in ("test", "train"):
        mf = out / f"dataset_{split}.h5"
        assert mf.exists()
        with h5py.File(str(mf), "r") as f:
            assert sorted(f.keys()) == ["X0", "Y0"]


@pytest.mark.integration
def test_merge_dataset_creates_output_dir_if_missing(tmp_path):
    in1 = tmp_path / "in1"
    out = tmp_path / "does" / "not" / "exist"  # nested, missing
    in1.mkdir()
    write_dataset_h5(str(in1 / "dataset_test.h5"), n_windows=2, seed=5)

    args = types.SimpleNamespace(
        chr_split="test",
        input_dir=[str(in1)],
        output_dir=str(out),
    )

    from openspliceai.merge_data.merge_dataset import merge_dataset

    merge_dataset(args)
    assert os.path.isdir(str(out))
    assert (out / "dataset_test.h5").exists()
