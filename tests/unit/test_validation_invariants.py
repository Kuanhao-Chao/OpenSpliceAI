"""Invariants established by the predict/variant validation audit (see validation/VALIDATION_REPORT.md).

These lock the conclusions of the keras-vs-original-SpliceAI equivalence study and the per-concern
verdicts with fast, CPU-only checks (no TensorFlow, no GPU, no trained weights required).
"""
import numpy as np
import pytest
import torch

from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.variant.utils import one_hot_encode

# The (flanking_size -> (W, AR)) schedule duplicated in predict.py and variant/utils.py.
HYPERPARAMS = {
    80:    ([11, 11, 11, 11], [1, 1, 1, 1]),
    400:   ([11] * 8, [1, 1, 1, 1, 4, 4, 4, 4]),
    2000:  ([11] * 8 + [21] * 4, [1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10]),
    10000: ([11] * 8 + [21] * 4 + [41] * 4,
            [1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25]),
}


@pytest.mark.parametrize("flank", [80, 400, 2000, 10000])
def test_cl_formula_equals_flanking_size(flank):
    """CL = 2*sum(AR*(W-1)) must equal the flanking size for every supported config."""
    W, AR = (np.asarray(x) for x in HYPERPARAMS[flank])
    assert 2 * int(np.sum(AR * (W - 1))) == flank


@pytest.mark.parametrize("flank", [80, 400, 2000, 10000])
def test_model_crops_exactly_flanking_size(flank):
    """A forward pass trims exactly `flank` positions (Cropping1D removes CL//2 each end)."""
    W, AR = (np.asarray(x) for x in HYPERPARAMS[flank])
    model = SpliceAI(32, W, AR).eval()
    out_len = 8
    x = torch.zeros(1, 4, flank + out_len)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, out_len)


def test_forward_applies_softmax_so_ensemble_averages_probabilities():
    """Concern #1: model output is post-softmax (channels sum to 1), so averaging models
    (predict.py / variant utils) averages probabilities — matching original SpliceAI."""
    W, AR = np.asarray([11, 11, 11, 11]), np.asarray([1, 1, 1, 1])
    model = SpliceAI(32, W, AR).eval()
    with torch.no_grad():
        y = model(torch.rand(1, 4, 5080))
    sums = y.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_pytorch_minus_strand_flip_is_reverse_complement():
    """Concern #5: torch.flip(x, dims=[1,2]) on a (1,4,L) one-hot equals the reverse-complement
    one-hot (channel axis A,C,G,T reversed = complement; length axis reversed = reverse)."""
    seq = "ACGTAACCGGTTAGCTN"
    oh = one_hot_encode(seq)                                  # (L, 4) rows over A,C,G,T
    t = torch.tensor(oh.T[None].astype("float32"))           # (1, 4, L)
    flipped = torch.flip(t, dims=[1, 2]).numpy()[0].T        # back to (L, 4)
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    rc = "".join(comp[b] for b in reversed(seq))
    assert np.array_equal(flipped, one_hot_encode(rc))


def test_exon_parse_matches_original_on_trailing_comma():
    """Exon-list parse delta: OpenSpliceAI's `split(',') if i` equals the original's
    `re.split(',')[:-1]` on the trailing-comma tables shipped with both tools."""
    import re
    cell = "999,5999,12345,"
    original = [int(i) for i in re.split(",", cell)[:-1]]
    openspliceai = [int(i) for i in cell.split(",") if i]
    assert original == openspliceai == [999, 5999, 12345]


def test_one_hot_encode_channel_order_is_acgt():
    """Channel order A,C,G,T underpins the RC-by-flip identity above and delta channel indexing."""
    assert np.array_equal(one_hot_encode("ACGT"), np.eye(4, dtype=one_hot_encode("A").dtype))
