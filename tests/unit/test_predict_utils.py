"""Unit tests for pure helpers in openspliceai/predict/predict.py."""
import io

import torch

import openspliceai.predict.predict as pp


def test_create_datapoints_shape_and_padding():
    SL, CL_max = 20, 8
    X = pp.create_datapoints("ACGT" * 50, SL=SL, CL_max=CL_max)   # 200bp input
    assert X.shape[1] == SL + CL_max         # window width
    assert X.shape[2] == 4                    # one-hot channels
    # CL_max//2 = 4 'N' padding rows lead the first window
    assert X[0, :4].sum() == 0


def _spiked_predictions():
    """One window, length 6, acceptor spike at pos 3, donor spike at pos 4."""
    preds = torch.zeros(1, 3, 6)
    preds[0, 1, 3] = 0.9   # acceptor channel
    preds[0, 2, 4] = 0.8   # donor channel
    return preds


def _cols(text):
    return [line.split("\t") for line in text.strip().splitlines() if line.strip()]


def test_write_batch_to_bed_plus_strand():
    ab, db = io.StringIO(), io.StringIO()
    pp.write_batch_to_bed("g1::chr5:100-200(+)", _spiked_predictions(), ab, db, threshold=0.5)
    a = _cols(ab.getvalue())[0]
    d = _cols(db.getvalue())[0]
    # acceptor window (-1,0) at pos 3 -> seq[2:3]; + strand chrom_start=(100-1)+2
    assert a[0] == "chr5" and a[1] == "101" and a[2] == "102" and a[5] == "+"
    # donor window (0,1) at pos 4 -> seq[4:5]; chrom_start=(100-1)+4
    assert d[0] == "chr5" and d[1] == "103" and d[2] == "104" and d[5] == "+"


def test_write_batch_to_bed_minus_strand():
    ab, db = io.StringIO(), io.StringIO()
    pp.write_batch_to_bed("g1::chr5:100-200(-)", _spiked_predictions(), ab, db, threshold=0.5)
    a = _cols(ab.getvalue())[0]
    d = _cols(db.getvalue())[0]
    # minus strand: chrom_start = end_1b - seq_end, chrom_end = end_1b - seq_start
    assert a[0] == "chr5" and a[1] == "197" and a[2] == "198" and a[5] == "-"
    assert d[0] == "chr5" and d[1] == "195" and d[2] == "196" and d[5] == "-"


def test_write_batch_to_bed_threshold_filters():
    ab, db = io.StringIO(), io.StringIO()
    pp.write_batch_to_bed("g1::chr5:100-200(+)", _spiked_predictions(), ab, db, threshold=0.95)
    assert ab.getvalue().strip() == ""   # 0.9 < 0.95
    assert db.getvalue().strip() == ""   # 0.8 < 0.95


def test_write_batch_to_bed_absolute_fallback():
    ab, db = io.StringIO(), io.StringIO()
    # no 'chrN:start-end(strand)' pattern -> absolute-coordinate output, strand from last char
    pp.write_batch_to_bed("scaffold123+", _spiked_predictions(), ab, db, threshold=0.5)
    a = _cols(ab.getvalue())[0]
    assert a[0] == "scaffold123+" and a[1] == "2" and a[2] == "3"
    assert a[-1] == "absolute_coordinates"
