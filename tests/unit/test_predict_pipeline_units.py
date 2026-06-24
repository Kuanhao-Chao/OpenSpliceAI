"""Unit coverage for the predict inference/output pipeline.

Covers the pieces the existing predict tests skip: the in-memory ``predict()`` API,
``create_datapoints``/``convert_sequences`` windowing for both the .h5 and .pt datafile
paths, ``get_prediction``'s flush-to-file branch + ensemble averaging, and
``write_batch_to_bed``/``generate_bed`` BED coordinate math on both strands (genomic-name
vs absolute-coordinate fallback, the strict score>threshold cutoff, undefined-strand skip).
"""
import io
import os

import h5py
import numpy as np
import pytest
import torch

import openspliceai.predict.predict as pr
import openspliceai.predict.utils as pu


# --- in-memory predict() API ---------------------------------------------------------

def test_predict_inmemory_returns_softmax_over_sequence(packaged_80nt_state):
    seq = "".join(np.random.default_rng(0).choice(list("ACGT"), size=200))
    y = pr.predict(seq, packaged_80nt_state, flanking_size=80)
    assert y.shape == (200, 3)                     # cropped back to input length
    sums = y.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)   # softmax rows
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# --- create_datapoints (predict variant: CL_max == flanking size) --------------------

def test_predict_create_datapoints_window_shape():
    X = pr.create_datapoints("ACGT" * 20, SL=5000, CL_max=80)
    assert X.ndim == 3 and X.shape[1] == 5080 and X.shape[2] == 4


# --- convert_sequences: both datafile backends ---------------------------------------

def test_convert_sequences_h5_backend(tmp_path):
    out = str(tmp_path) + "/"
    seqs = ["ACGT" * 30, "TTGGCCAA" * 20]
    dataset_path, LEN = pr.convert_sequences(str(tmp_path / "datafile.h5"), out, SL=5000,
                                             CL_max=80, SEQ=seqs)
    assert dataset_path.endswith("dataset.h5") and os.path.exists(dataset_path)
    assert len(LEN) == 2
    with h5py.File(dataset_path, "r") as f:
        assert "X0" in f and f["X0"].shape[-1] == 4


def test_convert_sequences_pt_backend(tmp_path):
    dataset_path, LEN = pr.convert_sequences(str(tmp_path / "datafile.txt"), str(tmp_path),
                                             SL=5000, CL_max=80, SEQ=["ACGT" * 30])
    assert dataset_path.endswith("dataset.pt") and os.path.exists(dataset_path)
    assert LEN == [len(torch.load(dataset_path))]


# --- get_prediction: flush branch + ensemble averaging -------------------------------

def _write_predict_dataset(path, n=4, seed=0):
    """dataset.h5 with one shard X0 = (n, SL+CL=5080, 4) int8 (load_shard transposes it)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, 5080))
    X = np.zeros((n, 5080, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    with h5py.File(path, "w") as f:
        f.create_dataset("X0", data=X)


def test_get_prediction_flushes_h5_and_ensembles(tmp_path, model_80nt):
    ds = str(tmp_path / "dataset.h5")
    _write_predict_dataset(ds, n=4)
    out = str(tmp_path) + "/"
    # flush_predict_threshold=0 forces a flush after every batch (exercises flush_predictions)
    predict_path = pr.get_prediction([model_80nt], ds, torch.device("cpu"), batch_size=2,
                                     output_dir=out, flush_predict_threshold=0)
    assert predict_path.endswith("predict.h5")
    with h5py.File(predict_path, "r") as f:
        assert f["predictions"].shape == (4, 3, 5000)   # (n, channels, SL)


def test_get_prediction_pt_backend(tmp_path, model_80nt):
    # build a dataset.pt of shape (n, 5080, 4) (get_prediction permutes 0,2,1)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 4, size=(2, 5080))
    X = np.zeros((2, 5080, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    ds = str(tmp_path / "dataset.pt")
    torch.save(torch.tensor(X, dtype=torch.int8), ds)
    out = str(tmp_path) + "/"
    predict_path = pr.get_prediction([model_80nt], ds, torch.device("cpu"), batch_size=2,
                                     output_dir=out)
    assert predict_path.endswith("predict.pt")
    preds = torch.load(predict_path)
    assert preds.shape == (2, 3, 5000)


def test_get_prediction_ensemble_is_mean_of_members(tmp_path, model_80nt):
    """Two members -> the saved prediction equals the arithmetic mean of member outputs."""
    import copy
    m2 = copy.deepcopy(model_80nt)
    for p in m2.parameters():           # perturb so the two members differ
        p.data.add_(0.01)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 4, size=(2, 5080))
    X = np.zeros((2, 5080, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    ds = str(tmp_path / "dataset.pt")
    torch.save(torch.tensor(X, dtype=torch.int8), ds)
    pp = pr.get_prediction([model_80nt, m2], ds, torch.device("cpu"), batch_size=2,
                           output_dir=str(tmp_path) + "/")
    ensemble = torch.load(pp)
    xb = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        manual = (model_80nt(xb) + m2(xb)) / 2
    assert torch.allclose(ensemble, manual, atol=1e-5)


# --- write_batch_to_bed: coordinate math + strand handling ---------------------------

def _preds(acceptor=0.9, donor=0.0, length=5):
    t = torch.zeros(1, 3, length)
    t[:, 0, :] = 1.0 - acceptor - donor
    t[:, 1, :] = acceptor
    t[:, 2, :] = donor
    return t


def test_write_batch_to_bed_plus_strand_genomic_coords():
    ab, db = io.StringIO(), io.StringIO()
    pr.write_batch_to_bed("g chr1:101-200(+)", _preds(acceptor=0.9), ab, db)
    lines = [ln for ln in ab.getvalue().splitlines() if ln]
    assert lines, "expected acceptor BED lines"
    chrom, cs, ce, name, score, strand = lines[0].split("\t")
    assert chrom == "chr1" and strand == "+" and name.endswith("_Acceptor")
    # acceptor window (-1,0) at pos1 -> seq_start=0 ; chrom_start = (101-1)+0 = 100
    assert int(cs) == 100 and int(ce) == 101


def test_write_batch_to_bed_minus_strand_genomic_coords():
    ab, db = io.StringIO(), io.StringIO()
    pr.write_batch_to_bed("g chr1:101-200(-)", _preds(acceptor=0.9), ab, db)
    chrom, cs, ce, *_ = ab.getvalue().splitlines()[0].split("\t")
    # minus strand: chrom_start = end_1b - seq_end ; pos1 -> seq=(0,1) -> 200-1=199
    assert chrom == "chr1" and int(cs) == 199 and int(ce) == 200


def test_write_batch_to_bed_absolute_fallback_when_name_has_no_coords():
    ab, db = io.StringIO(), io.StringIO()
    pr.write_batch_to_bed("myseq+", _preds(donor=0.9, acceptor=0.0), ab, db)
    out = db.getvalue()
    assert "absolute_coordinates" in out and out.splitlines()[0].startswith("myseq+")


def test_write_batch_to_bed_threshold_is_strict():
    ab, db = io.StringIO(), io.StringIO()
    pr.write_batch_to_bed("g chr1:1-10(+)", _preds(acceptor=1e-6), ab, db, threshold=1e-6)
    assert ab.getvalue() == ""   # score == threshold is NOT written (strict >)


def test_write_batch_to_bed_skips_undefined_strand(capsys):
    ab, db = io.StringIO(), io.StringIO()
    pr.write_batch_to_bed("g chr1:1-10(.)", _preds(acceptor=0.9), ab, db)
    assert ab.getvalue() == "" and db.getvalue() == ""
    assert "Undefined strand" in capsys.readouterr().out


# --- generate_bed end-to-end (predict.h5 -> BED) -------------------------------------

def test_generate_bed_from_h5(tmp_path):
    n, length = 2, 8
    preds = np.zeros((n, 3, length), dtype=np.float32)
    preds[:, 1, :] = 0.9   # acceptor channel high everywhere
    pf = tmp_path / "predict.h5"
    with h5py.File(pf, "w") as f:
        f.create_dataset("predictions", data=preds)
    out = str(tmp_path) + "/"
    pr.generate_bed(str(pf), ["chr1:1-40(+)"], [n], out)   # sum(LEN)=2=len(preds)
    acc = (tmp_path / "acceptor_predictions.bed").read_text().splitlines()
    assert (tmp_path / "donor_predictions.bed").exists()
    assert acc and acc[0].split("\t")[0] == "chr1" and "_Acceptor" in acc[0]


def test_generate_bed_asserts_len_matches_predictions(tmp_path):
    pf = tmp_path / "predict.h5"
    with h5py.File(pf, "w") as f:
        f.create_dataset("predictions", data=np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(AssertionError):
        pr.generate_bed(str(pf), ["chr1:1-10(+)"], [5], str(tmp_path) + "/")  # 5 != 2


# --- predict/utils helpers -----------------------------------------------------------

def test_initialize_constants_cl_max_is_flanking_size():
    consts = pu.initialize_constants(80)
    assert consts["CL_max"] == 80 and consts["SL"] == 5000


def test_initialize_paths_creates_dir(tmp_path):
    p = pu.initialize_paths(str(tmp_path), 80, 5000)
    assert os.path.isdir(p) and "SpliceAI_5000_80" in p


def test_log_memory_usage_runs(capsys):
    pu.log_memory_usage()
    assert "Memory usage" in capsys.readouterr().err
