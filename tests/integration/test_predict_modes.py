"""End-to-end predict_cli mode coverage on tiny synthetic FASTAs (CPU).

Exercises the HDF5 datafile path (turbo + --predict-all), the 5-model ensemble, and the
annotation-GFF (process_gff) path. The non-HDF5 (.pt) turbo path is covered in
tests/integration/test_predict_pt_path.py and is intentionally not duplicated here.

The output base directory is grounded against predict.utils.initialize_paths, which
returns ``{output_dir}/SpliceAI_{SL}_{flanking_size}/`` (SL=5000, flanking=80).
"""
import glob
import os
import types

import numpy as np
import pytest

from openspliceai.predict import predict as pr
from openspliceai.predict import utils as pred_utils


def _write_small_fasta(path, length=1200, seed=0):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGT"), size=length))
    with open(path, "w") as fh:
        fh.write(">chr1\n" + seq + "\n")


def _make_args(input_sequence, model, output_dir, predict_all, hdf_threshold,
               annotation_file=None):
    return types.SimpleNamespace(
        input_sequence=str(input_sequence),
        model=model,
        flanking_size=80,
        output_dir=str(output_dir),
        annotation_file=annotation_file,
        threshold=1e-6,
        debug=False,
        predict_all=predict_all,
        hdf_threshold=hdf_threshold,
        flush_threshold=500,
        split_threshold=1500000,
        chunk_size=100,
    )


def _output_base(output_dir):
    """The directory predict_cli writes BED/predict files into (grounded via initialize_paths)."""
    from openspliceai.constants import SL
    return pred_utils.initialize_paths(str(output_dir), 80, SL)


@pytest.mark.integration
@pytest.mark.slow
def test_turbo_hdf5_path_writes_beds(tmp_path, packaged_80nt_state):
    """hdf_threshold=0 forces the HDF5 datafile/dataset path; turbo mode writes both BEDs."""
    fa = tmp_path / "mini.fa"
    _write_small_fasta(str(fa))
    out = tmp_path / "out"

    pr.predict_cli(_make_args(fa, packaged_80nt_state, out, predict_all=False, hdf_threshold=0))

    base = _output_base(out)
    acc = os.path.join(base, "acceptor_predictions.bed")
    don = os.path.join(base, "donor_predictions.bed")
    assert os.path.exists(acc) and os.path.exists(don)
    # HDF5 datafile/dataset were really produced (proves the HDF branch, not the .pt branch)
    assert os.path.exists(os.path.join(base, "datafile.h5"))
    assert os.path.exists(os.path.join(base, "dataset.h5"))
    # the .pt branch's predict.pt must NOT exist for turbo HDF5 mode
    assert not os.path.exists(os.path.join(base, "predict.pt"))


@pytest.mark.integration
@pytest.mark.slow
def test_predict_all_hdf5_writes_predict_h5_and_beds(tmp_path, packaged_80nt_state):
    """predict_all=True with hdf_threshold=0 writes an intermediate predict.h5 plus BEDs."""
    fa = tmp_path / "mini.fa"
    _write_small_fasta(str(fa))
    out = tmp_path / "out"

    pr.predict_cli(_make_args(fa, packaged_80nt_state, out, predict_all=True, hdf_threshold=0))

    base = _output_base(out)
    assert os.path.exists(os.path.join(base, "predict.h5"))
    assert os.path.exists(os.path.join(base, "acceptor_predictions.bed"))
    assert os.path.exists(os.path.join(base, "donor_predictions.bed"))


@pytest.mark.integration
@pytest.mark.slow
def test_ensemble_directory_writes_beds(tmp_path, packaged_80nt_dir):
    """Pointing --model at a directory ensembles all 5 checkpoints (mean) and still writes BEDs."""
    fa = tmp_path / "mini.fa"
    _write_small_fasta(str(fa))
    out = tmp_path / "out"

    pr.predict_cli(_make_args(fa, packaged_80nt_dir, out, predict_all=False, hdf_threshold=0))

    beds = glob.glob(str(out / "**" / "*_predictions.bed"), recursive=True)
    acc = [b for b in beds if "acceptor" in b]
    don = [b for b in beds if "donor" in b]
    assert acc and don, beds


@pytest.mark.integration
@pytest.mark.slow
def test_annotation_gff_path_extracts_genes_and_writes_beds(tmp_path, packaged_80nt_state):
    """A GFF annotation routes through process_gff (-> *_genes.fa) before prediction."""
    fa = tmp_path / "mini.fa"
    _write_small_fasta(str(fa))

    gff = tmp_path / "mini.gff"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\tt\tgene\t100\t900\t.\t+\t.\tID=g1\n"
        "chr1\tt\tmRNA\t100\t900\t.\t+\t.\tID=m1;Parent=g1\n"
        "chr1\tt\texon\t100\t900\t.\t+\t.\tID=e1;Parent=m1\n"
    )
    out = tmp_path / "out"

    pr.predict_cli(_make_args(fa, packaged_80nt_state, out, predict_all=False,
                              hdf_threshold=0, annotation_file=str(gff)))

    base = _output_base(out)
    # process_gff writes '{base}{fasta_basename}_genes.fa'
    assert os.path.exists(os.path.join(base, "mini_genes.fa"))
    assert os.path.exists(os.path.join(base, "acceptor_predictions.bed"))
    assert os.path.exists(os.path.join(base, "donor_predictions.bed"))
