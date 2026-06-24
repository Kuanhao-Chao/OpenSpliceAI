"""Characterization locks for the two one-hot encoders.

A 3-agent bug-hunt + direct probes found the encoders correct (lowercase handled, N ->
all-zero row, no byte-order surprise in the latin-1 trick). These tests pin that
conclusion: encode -> argmax-decode must round-trip A/C/G/T exactly, N/invalid must map
to an all-zero row, and the channel order must be A,C,G,T in both implementations and in
the IN_MAP/OUT_MAP lookup tables that drive the production ``create_datapoints`` path.
"""
import numpy as np
import pytest

from openspliceai.create_data.utils import IN_MAP, OUT_MAP, create_datapoints
from openspliceai.variant.utils import one_hot_encode as variant_one_hot

ACGT = "ACGT"


def _decode_acgt(onehot):
    """(L, 4) one-hot over A,C,G,T -> string; all-zero rows decode to 'N'."""
    onehot = np.asarray(onehot)
    out = []
    for row in onehot:
        s = int(row.sum())
        out.append("N" if s == 0 else ACGT[int(np.argmax(row))])
    return "".join(out)


# --- variant.one_hot_encode (string-based, latin-1 modulo trick) ---------------------

def test_variant_encoder_roundtrips_random_acgt():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(ACGT), size=500))
    enc = variant_one_hot(seq)
    assert enc.shape == (500, 4)
    assert (enc.sum(axis=1) == 1).all()           # every ACGT base is exactly one hot bit
    assert _decode_acgt(enc) == seq               # exact round-trip


def test_variant_encoder_is_case_insensitive():
    seq = "ACGTacgtAcGt"
    np.testing.assert_array_equal(variant_one_hot(seq), variant_one_hot(seq.upper()))


@pytest.mark.parametrize("bad", ["N", "n", "X", "-", "R"])
def test_variant_encoder_maps_non_acgt_to_zero_row(bad):
    enc = variant_one_hot(f"A{bad}T")
    assert enc[1].sum() == 0                       # the invalid base -> all-zero row
    assert enc[0].tolist() == [1, 0, 0, 0]         # neighbours unaffected
    assert enc[2].tolist() == [0, 0, 0, 1]


# --- IN_MAP / OUT_MAP lookup tables (drive create_datapoints) ------------------------

def test_in_map_channel_order_and_padding():
    # index 0 = N/pad -> all zero; 1..4 = A,C,G,T -> identity one-hot in ACGT order
    assert IN_MAP[0].tolist() == [0, 0, 0, 0]
    for code, _nt in enumerate(ACGT, start=1):
        assert int(np.argmax(IN_MAP[code])) == code - 1
        assert int(IN_MAP[code].sum()) == 1


def test_out_map_channel_order_and_padding():
    # 0 = no-splice, 1 = acceptor, 2 = donor, 3 = pad/ignore (all-zero)
    for code in (0, 1, 2):
        assert int(np.argmax(OUT_MAP[code])) == code
        assert int(OUT_MAP[code].sum()) == 1
    assert OUT_MAP[3].tolist() == [0, 0, 0]


# --- production create_datapoints round-trip -----------------------------------------

def test_create_datapoints_roundtrips_sequence_through_real_windowing():
    """The SL-window carries CL_max//2 padding on each side; the original bases sit at
    offset CL_max//2 and must decode back exactly (channel order preserved end-to-end)."""
    from openspliceai.constants import CL_max

    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list(ACGT), size=120))
    label = "0" * len(seq)
    X, Y = create_datapoints(seq, label)

    assert X.ndim == 3 and X.shape[-1] == 4
    window = X[0]
    decoded = _decode_acgt(window[CL_max // 2: CL_max // 2 + len(seq)])
    assert decoded == seq
    # the padded flanks are all-N (all-zero rows)
    assert window[: CL_max // 2].sum() == 0


def test_create_datapoints_label_positions_map_through_out_map():
    """Acceptor(1)/donor(2) labels land on the right output channel at the right index."""
    from openspliceai.constants import CL_max  # noqa: F401  (kept for symmetry/readability)

    seq = "ACGT" * 30                 # 120 bp
    label = list("0" * len(seq))
    label[10] = "1"                   # acceptor
    label[20] = "2"                   # donor
    label = "".join(label)
    _X, Y = create_datapoints(seq, label)
    y = np.asarray(Y[0])[0]          # (SL, 3)
    assert int(np.argmax(y[10])) == 1
    assert int(np.argmax(y[20])) == 2
    assert int(np.argmax(y[0])) == 0
