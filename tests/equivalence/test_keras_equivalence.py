"""Numerical-equivalence regression: OpenSpliceAI `variant --model-type keras` must reproduce the
ORIGINAL Illumina ``spliceai`` tool exactly, since both run byte-identical Keras weights through the
same delta-score algorithm.

This is the fast, in-process regression lock for the keystone validation result. The definitive
study (real genes, high scores, mask x distance grid, all exact) is in validation/VALIDATION_REPORT.md;
here we drive both tools' ``get_delta_scores`` over the synthetic SNV/deletion/insertion/multiallelic
fixture and assert the formatted 10-field strings are identical.

Equivalence holds ONLY at flanking_size=10000, because original SpliceAI hardcodes wid = 10000 + cov.

Marked ``keras``+``slow``: auto-skips when TensorFlow is absent; also skips if the bundled weights or
the original ``spliceai`` package are unavailable.
"""
import pytest

pytestmark = [pytest.mark.keras, pytest.mark.slow, pytest.mark.integration]


def _orig_get_delta(rec, ann, dist, mask):
    from spliceai.utils import get_delta_scores
    return list(get_delta_scores(rec, ann, dist, mask))


@pytest.mark.parametrize("dist,mask", [(50, 0), (50, 1), (500, 0)])
def test_keras_matches_original_spliceai(variant_inputs, repo_root, dist, mask):
    pytest.importorskip("spliceai")                     # original Illumina tool
    pytest.importorskip("spliceai.utils")
    import pysam

    model_dir = repo_root / "models" / "spliceai" / "SpliceAI_models_release"
    if not (model_dir / "spliceai1.h5").exists():
        pytest.skip("bundled SpliceAI Keras weights not found")

    ref, ann, vcf = variant_inputs

    # original spliceai annotator
    try:
        from spliceai.utils import Annotator as OrigAnnotator
        orig_ann = OrigAnnotator(ref, ann)
    except Exception as e:
        pytest.skip(f"original spliceai Annotator unavailable: {e}")

    # OpenSpliceAI keras annotator over the SAME 5 bundled weights + SAME annotation
    from openspliceai.variant.utils import Annotator as OSAnnotator, get_delta_scores
    os_ann = OSAnnotator(ref, ann, model_path=str(model_dir), model_type="keras", CL=10000)

    n_compared = 0
    for rec in pysam.VariantFile(vcf):
        orig = _orig_get_delta(rec, orig_ann, dist, mask)
        os_scores = list(get_delta_scores(rec, os_ann, dist, mask, flanking_size=10000, precision=2))
        assert os_scores == orig, (
            f"mismatch at {rec.chrom}:{rec.pos} {rec.ref}->{rec.alts}\n"
            f"  original     : {orig}\n  openspliceai : {os_scores}"
        )
        n_compared += len(os_scores)
    assert n_compared >= 1     # the fixture yields SNV/del/ins/multiallelic annotations
