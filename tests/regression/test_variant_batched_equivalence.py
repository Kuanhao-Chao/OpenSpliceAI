"""Regression: the batched variant path must equal the sequential path exactly.

``get_delta_scores_batched`` (used by ``variant --batch-size > 1``) buffers every
(variant, alt, gene) window and runs the model in sub-batches instead of one window at a
time. Because the SpliceAI forward pass is per-sample independent (1-D conv + eval-mode
BatchNorm using *running* stats, never batch stats) the batched output must be **identical**
to scoring each record on its own with ``get_delta_scores``.

We assert exact equality of the formatted delta-score strings (allele|symbol|DS...|DP...),
across several ``batch_size`` values (incl. sizes smaller than the window count, to exercise
sub-batching) and both ``mask`` settings. The test runs the *real* packaged 80nt PyTorch
checkpoint on CPU over the synthetic ``write_variant_inputs`` fixture
(SNV + deletion + insertion + multi-allelic). ``OSAI_TF32`` is pinned off so the batched
math is bit-reproducible (TF32 is a GPU fast path that adds ~1e-3 noise; it is a no-op on CPU
but we set it defensively in case the suite ever runs on a GPU box).
"""
import os

import pytest

# Pin off the A100 TF32 fast path so batched == sequential is bit-exact even on a GPU runner.
os.environ["OSAI_TF32"] = "0"


@pytest.fixture(scope="module")
def pytorch_annotator(tmp_path_factory, repo_root):
    from tests.fixtures.synthetic import write_variant_inputs
    from openspliceai.variant.utils import Annotator

    state = repo_root / "models" / "openspliceai-honeybee" / "80nt" / "model_80nt_rs10.pt"
    if not state.exists():
        pytest.skip(f"packaged 80nt checkpoint not found: {state}")

    d = tmp_path_factory.mktemp("variant_batch_ann")
    ref, ann, vcf = write_variant_inputs(d)
    annotator = Annotator(ref, ann, model_path=str(state), model_type="pytorch", CL=80)
    return annotator, vcf


def _records(vcf_path):
    import pysam
    return list(pysam.VariantFile(vcf_path))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
@pytest.mark.parametrize("mask", [0, 1])
def test_batched_equals_sequential(pytorch_annotator, batch_size, mask):
    from openspliceai.variant.utils import get_delta_scores, get_delta_scores_batched

    annotator, vcf = pytorch_annotator
    records = _records(vcf)

    sequential = [
        get_delta_scores(rec, annotator, dist_var=50, mask=mask, flanking_size=80)
        for rec in records
    ]
    batched = get_delta_scores_batched(
        records, annotator, dist_var=50, mask=mask, flanking_size=80, batch_size=batch_size
    )

    assert len(batched) == len(sequential)
    # Per-record, the formatted score strings must match exactly (same order, same values).
    for rec, seq_scores, bat_scores in zip(records, sequential, batched):
        assert bat_scores == seq_scores, (
            f"batched != sequential for {rec.chrom}:{rec.pos} "
            f"(batch_size={batch_size}, mask={mask})\n"
            f"  sequential={seq_scores}\n  batched   ={bat_scores}"
        )


def test_batched_total_score_count_matches(pytorch_annotator):
    """Sanity: across all records the batched path emits the same 5 scores as sequential."""
    from openspliceai.variant.utils import get_delta_scores, get_delta_scores_batched

    annotator, vcf = pytorch_annotator
    records = _records(vcf)

    seq_total = sum(
        len(get_delta_scores(rec, annotator, dist_var=50, mask=0, flanking_size=80))
        for rec in records
    )
    bat_total = sum(
        len(r) for r in get_delta_scores_batched(
            records, annotator, dist_var=50, mask=0, flanking_size=80, batch_size=4
        )
    )
    assert seq_total == bat_total == 5   # SNV(1) + del(1) + ins(1) + multiallelic(2)
