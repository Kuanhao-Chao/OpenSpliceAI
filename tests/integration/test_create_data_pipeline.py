"""End-to-end create-data pipeline tests on the mini 2-chromosome genome fixture.

Each test runs the production two-stage flow back-to-back:
    create_datafile.create_datafile(args)  ->  create_dataset.create_dataset(args)
and inspects the real dataset_*.h5 outputs. All shapes/values were grounded by running
the pipeline on the fixture before asserting.
"""
import os
import random
import types

import h5py
import pytest

from openspliceai.create_data import create_datafile, create_dataset, verify_h5_file


def _make_args(gff, fasta, out, **overrides):
    args = types.SimpleNamespace(
        annotation_gff=gff,
        genome_fasta=fasta,
        output_dir=out,
        parse_type="canonical",
        biotype="protein-coding",
        chr_split="test",
        split_method="random",
        val_split_ratio=0.1,
        split_ratio=0.8,
        canonical_only=False,
        remove_paralogs=False,
        write_fasta=False,
        verify_h5=False,
        flanking_size=80,
        min_identity=0.8,
        min_coverage=0.5,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.mark.integration
def test_create_data_test_split_shapes(tmp_path, mini_genome_gff):
    """chr_split='test' writes dataset_test.h5 with the production windowed shapes."""
    random.seed(0)
    fasta, gff = mini_genome_gff
    out = os.path.join(str(tmp_path), "out")  # deliberately no trailing slash

    args = _make_args(gff, fasta, out, chr_split="test")
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)

    ds = os.path.join(out, "dataset_test.h5")
    assert os.path.exists(ds)
    with h5py.File(ds, "r") as f:
        assert "X0" in f and "Y0" in f
        n = f["X0"].shape[0]
        assert n >= 1
        # X = (n, SL+CL_max=15000, 4) int8 ; Y = (1, n, SL=5000, 3) int8
        assert f["X0"].shape == (n, 15000, 4)
        assert f["X0"].dtype == "int8"
        assert f["Y0"].shape == (1, n, 5000, 3)
        assert f["Y0"].dtype == "int8"


@pytest.mark.integration
def test_create_data_output_dir_without_trailing_slash(tmp_path, mini_genome_gff):
    """Path robustness: output_dir without a trailing slash must still work end-to-end."""
    random.seed(0)
    fasta, gff = mini_genome_gff
    out = os.path.join(str(tmp_path), "out")  # NO trailing slash

    args = _make_args(gff, fasta, out, chr_split="test")
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)

    assert os.path.exists(os.path.join(out, "datafile_test.h5"))
    assert os.path.exists(os.path.join(out, "dataset_test.h5"))


@pytest.mark.integration
def test_create_data_write_fasta(tmp_path, mini_genome_gff):
    """--write-fasta=True writes a test.fa alongside the h5 datasets."""
    random.seed(0)
    fasta, gff = mini_genome_gff
    out = os.path.join(str(tmp_path), "out")

    args = _make_args(gff, fasta, out, chr_split="test", write_fasta=True)
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)

    fa = os.path.join(out, "test.fa")
    assert os.path.exists(fa)
    assert os.path.getsize(fa) > 0


@pytest.mark.integration
def test_create_data_train_test_split(tmp_path, mini_genome_gff):
    """chr_split='train-test' with val_split_ratio=0.5 yields non-empty train+val+test."""
    random.seed(0)
    fasta, gff = mini_genome_gff
    out = os.path.join(str(tmp_path), "out")

    args = _make_args(gff, fasta, out, chr_split="train-test", val_split_ratio=0.5)
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)

    train_ds = os.path.join(out, "dataset_train.h5")
    test_ds = os.path.join(out, "dataset_test.h5")
    val_ds = os.path.join(out, "dataset_validation.h5")
    assert os.path.exists(train_ds)
    assert os.path.exists(test_ds)
    assert os.path.exists(val_ds)

    # validation must be non-empty since val_split_ratio=0.5
    with h5py.File(val_ds, "r") as f:
        assert "X0" in f
        assert f["X0"].shape[0] >= 1
        assert f["X0"].shape[1:] == (15000, 4)


@pytest.mark.integration
def test_create_data_verify_h5_writes_plot(tmp_path, mini_genome_gff):
    """--verify-h5 reads the produced dataset and writes a verify_*.png summary plot."""
    random.seed(0)
    fasta, gff = mini_genome_gff
    out = os.path.join(str(tmp_path), "out")

    args = _make_args(gff, fasta, out, chr_split="test", verify_h5=True)
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)
    verify_h5_file.verify_h5(args)

    assert os.path.exists(os.path.join(out, "verify_test.png"))
