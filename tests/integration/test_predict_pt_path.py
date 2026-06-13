"""End-to-end smoke test proving Fix #5: the non-HDF5 (.pt) predict path runs.

Previously this path raised NameError (`model`/`i` undefined), a tensor `.transpose(0,2,1)`
error, and `str.decode`. Forcing a huge --hdf-threshold makes predict use the .pt path. A tiny
synthetic FASTA (one window) keeps the CPU forward pass fast.
"""
import glob
import os
import types

import numpy as np
import pytest


def _write_small_fasta(path, length=1200, seed=0):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGT"), size=length))
    with open(path, "w") as fh:
        fh.write(">testseq\n" + seq + "\n")


@pytest.mark.integration
@pytest.mark.slow
def test_predict_pt_path_writes_bed(tmp_path, packaged_80nt_state):
    fa = tmp_path / "mini.fa"
    _write_small_fasta(str(fa))
    out = tmp_path / "out"

    from openspliceai.predict import predict as pr

    args = types.SimpleNamespace(
        input_sequence=str(fa),
        model=packaged_80nt_state,
        flanking_size=80,
        output_dir=str(out),
        annotation_file=None,
        threshold=1e-6,           # low threshold so the BED write path actually executes
        debug=False,
        predict_all=False,        # turbo mode -> predict_and_write (.pt branch)
        hdf_threshold=10 ** 9,    # force the non-HDF5 (.pt) path
        flush_threshold=500,
        split_threshold=1500000,
        chunk_size=100,
    )
    pr.predict_cli(args)

    beds = glob.glob(str(out / "**" / "*_predictions.bed"), recursive=True)
    acc = [b for b in beds if "acceptor" in b]
    don = [b for b in beds if "donor" in b]
    assert acc and don, beds
    # the .pt write path executed and produced at least some predictions
    assert os.path.getsize(acc[0]) > 0 or os.path.getsize(don[0]) > 0
