"""Branch + debug coverage for predict/predict.py.

Drives the turbo ``predict_and_write`` path directly, the ``load_pytorch_models``
flanking-size table branches + checkpoint-mismatch handling, ``convert_sequences`` file
backends/edge cases, ``get_sequences`` splitting + minus-strand reverse-complement, and
sweeps the ``debug=True`` print branches across the pipeline."""
import os

import h5py
import numpy as np
import pytest
import torch

import openspliceai.predict.predict as pr


def _dataset_h5(path, n=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, 5080))
    X = np.zeros((n, 5080, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    with h5py.File(path, "w") as f:
        f.create_dataset("X0", data=X)
    return path


# --- predict_and_write (turbo path), both backends, debug on -------------------------

def test_predict_and_write_h5_debug(tmp_path, model_80nt):
    ds = _dataset_h5(str(tmp_path / "dataset.h5"), n=4)
    out = str(tmp_path) + "/"
    pr.predict_and_write([model_80nt], ds, torch.device("cpu"), 2,
                         ["geneA chr1:1-10(+)", "geneB chr2:1-10(-)"], [2, 2], out, debug=True)
    assert os.path.exists(out + "acceptor_predictions.bed")
    assert os.path.exists(out + "donor_predictions.bed")


def test_predict_and_write_pt(tmp_path, model_80nt):
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 4, size=(4, 5080))
    X = np.zeros((4, 5080, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    ds = str(tmp_path / "dataset.pt")
    torch.save(torch.tensor(X, dtype=torch.int8), ds)
    out = str(tmp_path) + "/"
    pr.predict_and_write([model_80nt], ds, torch.device("cpu"), 2,
                         ["chr1:1-10(+)", "chr2:1-10(+)"], [2, 2], out)
    assert os.path.exists(out + "acceptor_predictions.bed")


# --- load_pytorch_models: table branches + mismatch handling -------------------------

@pytest.mark.parametrize("cl", [400, 2000, 10000])
def test_load_pytorch_models_builds_table_then_exits_on_mismatch(packaged_80nt_state, cl):
    """An 80nt checkpoint loaded as a larger flanking size builds that architecture
    (exercising the W/AR table branch) then exits because no model loads."""
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(packaged_80nt_state, torch.device("cpu"), 5000, cl)


def test_load_pytorch_models_size_mismatch_branch(tmp_path, packaged_80nt_state):
    """A checkpoint with a correctly-named but wrongly-shaped tensor hits the explicit
    'size mismatch' handling, then exits with no usable model."""
    state = torch.load(packaged_80nt_state, map_location="cpu")
    key = next(iter(state))
    state[key] = torch.zeros(state[key].shape[0] + 1, *state[key].shape[1:])  # wrong shape
    bad = str(tmp_path / "bad.pt")
    torch.save(state, bad)
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(bad, torch.device("cpu"), 5000, 80)


# --- convert_sequences: file backends + edge cases -----------------------------------

def test_convert_sequences_reads_seq_from_h5_when_not_provided(tmp_path):
    datafile = str(tmp_path / "datafile.h5")
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(datafile, "w") as f:
        f.create_dataset("SEQ", data=np.asarray(["ACGT" * 30], dtype=dt), dtype=dt)
    dataset_path, LEN = pr.convert_sequences(datafile, str(tmp_path) + "/", 5000, 80, debug=True)
    assert os.path.exists(dataset_path) and len(LEN) == 1


def test_convert_sequences_reads_seq_from_txt_when_not_provided(tmp_path):
    datafile = str(tmp_path / "datafile.txt")
    with open(datafile, "w") as f:
        f.write(">n1\n" + "ACGT" * 30 + "\n")
    dataset_path, LEN = pr.convert_sequences(datafile, str(tmp_path), 5000, 80, debug=True)
    assert dataset_path.endswith("dataset.pt") and len(LEN) == 1


def test_convert_sequences_decodes_bytes_seq_in_pt_path(tmp_path):
    dataset_path, LEN = pr.convert_sequences(str(tmp_path / "d.txt"), str(tmp_path), 5000, 80,
                                             SEQ=[b"ACGT" * 30])
    assert os.path.exists(dataset_path) and LEN == [1]


def test_convert_sequences_last_chunk_remainder(tmp_path):
    seqs = ["ACGT" * 30, "TTGG" * 25, "GGCC" * 20]
    dataset_path, LEN = pr.convert_sequences(str(tmp_path / "d.h5"), str(tmp_path) + "/", 5000,
                                             80, chunk_size=2, SEQ=seqs)
    with h5py.File(dataset_path, "r") as f:
        assert "X0" in f and "X1" in f   # 3 seqs, chunk_size 2 -> 2 chunks (last is remainder)
    assert len(LEN) == 3


# --- split_fasta create_name fallback (no chr in name) -------------------------------

def test_split_fasta_non_chr_name_uses_record_name(tmp_path):
    from pyfaidx import Fasta
    seq = "".join(np.random.default_rng(0).choice(list("ACGT"), size=250))
    fa = tmp_path / "plain.fa"
    fa.write_text(">plainseq\n" + seq + "\n")
    genes = Fasta(str(fa), one_based_attributes=True, read_long_names=False,
                  sequence_always_upper=True)
    out = tmp_path / "split.fa"
    pr.split_fasta(genes, str(out), CL_max=80, split_fasta_threshold=100)
    headers = [ln[1:] for ln in out.read_text().splitlines() if ln.startswith(">")]
    assert headers[0].startswith("plainseq:") and headers[0].endswith("(.)")


# --- get_sequences: splitting + minus-strand reverse-complement + debug --------------

def test_get_sequences_splits_long_sequence_into_h5(tmp_path):
    seq = "".join(np.random.default_rng(4).choice(list("ACGT"), size=300))
    fa = tmp_path / "long.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    # hdf_threshold_len=0 -> use_hdf ; split_threshold=100 -> need_splitting (hits the break)
    datafile, NAME, SEQ = pr.get_sequences(str(fa), str(tmp_path) + "/", CL_max=80,
                                           hdf_threshold_len=0, split_fasta_threshold=100,
                                           debug=True)
    assert datafile.endswith(".h5") and os.path.exists(datafile)
    assert len(NAME) == len(SEQ) and len(NAME) >= 3   # split into >=3 segments


def test_get_sequences_neg_strand_reverse_complements(tmp_path):
    from Bio.Seq import Seq
    seq = "".join(np.random.default_rng(5).choice(list("ACGT"), size=120))
    fa = tmp_path / "s.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    _datafile, NAME, SEQ = pr.get_sequences(str(fa), str(tmp_path) + "/", CL_max=80,
                                            hdf_threshold_len=10 ** 9,
                                            split_fasta_threshold=10 ** 9, neg_strands=["chr1"])
    assert NAME == ["chr1:-"]
    assert SEQ[0] == str(Seq(seq).reverse_complement())


# --- debug sweeps for the remaining if-debug branches --------------------------------

def test_create_datapoints_debug(capsys):
    pr.create_datapoints("ACGT" * 20, SL=5000, CL_max=80, debug=True)
    assert "DEBUG" in capsys.readouterr().err


def test_get_prediction_h5_debug(tmp_path, model_80nt):
    ds = _dataset_h5(str(tmp_path / "dataset.h5"), n=4)
    pr.get_prediction([model_80nt], ds, torch.device("cpu"), 2, str(tmp_path) + "/",
                      flush_predict_threshold=0, debug=True)


def test_generate_bed_debug(tmp_path):
    preds = np.zeros((2, 3, 6), dtype=np.float32)
    preds[:, 2, :] = 0.9
    pf = tmp_path / "predict.h5"
    with h5py.File(pf, "w") as f:
        f.create_dataset("predictions", data=preds)
    pr.generate_bed(str(pf), ["chr1:1-20(+)"], [2], str(tmp_path) + "/", debug=True)
    assert (tmp_path / "donor_predictions.bed").exists()


def test_generate_bed_pt_backend(tmp_path):
    preds = torch.zeros(2, 3, 6)
    preds[:, 1, :] = 0.9
    pf = tmp_path / "predict.pt"
    torch.save(preds, pf)
    # the .pt branch only loads when batch_ypred is passed (non-None)
    pr.generate_bed(str(pf), ["chr1:1-20(+)"], [2], str(tmp_path) + "/", batch_ypred=preds)
    assert (tmp_path / "acceptor_predictions.bed").exists()


# --- process_gff + load_pytorch_models error/exit paths ------------------------------

def test_process_gff_skips_malformed_line(tmp_path):
    fa = tmp_path / "f.fa"
    fa.write_text(">chr1\n" + "ACGT" * 10 + "\n")
    gff = tmp_path / "f.gff"
    gff.write_text("##gff-version 3\nshort\tline\twith\tfew\tfields\n"
                   "chr1\tt\tgene\t1\t20\t.\t+\t.\tID=g\n")
    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/")
    # malformed line skipped; the valid gene is still extracted
    assert ">g chr1:1-20(+)" in open(out_fa).read()


def test_load_pytorch_models_empty_dir_exits(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(str(d), torch.device("cpu"), 5000, 80)


def test_load_pytorch_models_dir_with_only_corrupt_checkpoint_exits(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "bad.pt").write_text("not a torch checkpoint")
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(str(d), torch.device("cpu"), 5000, 80)


def test_load_pytorch_models_unloadable_file_exits(tmp_path):
    bad = tmp_path / "x.pt"
    bad.write_text("garbage")
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(str(bad), torch.device("cpu"), 5000, 80)


def test_load_pytorch_models_nonexistent_path_exits(tmp_path):
    with pytest.raises(SystemExit):
        pr.load_pytorch_models(str(tmp_path / "nope.pt"), torch.device("cpu"), 5000, 80)
