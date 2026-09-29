"""Original Keras weights give equal formatted DS/DP for shared supported alleles.

OpenSpliceAI development additionally scores MNV/delins alleles for which original
SpliceAI 1.3.1 returns placeholders, and rejects REF spans beyond its realignment
boundary. These extensions are tested separately; they are not parity claims.
"""
import pytest

pytestmark = [pytest.mark.keras, pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def keras_annotators(tmp_path_factory, repo_root):
    pytest.importorskip("spliceai")
    pytest.importorskip("spliceai.utils")
    from spliceai.utils import Annotator as OrigAnnotator
    from openspliceai.variant.utils import Annotator as OSAnnotator
    from tests.fixtures.synthetic import write_variant_inputs
    model_dir = repo_root / "models" / "spliceai" / "SpliceAI_models_release"
    if not (model_dir / "spliceai1.h5").exists():
        pytest.skip("bundled SpliceAI Keras weights not found")
    reference, annotation, vcf = write_variant_inputs(tmp_path_factory.mktemp("keras_parity"))
    # Available backends and weights must initialize successfully; errors fail.
    original = OrigAnnotator(reference, annotation)
    current = OSAnnotator(reference, annotation, model_path=str(model_dir), model_type="keras", CL=10000)
    try:
        yield original, current, vcf
    finally:
        original.ref_fasta.close()
        current.ref_fasta.close()


@pytest.mark.parametrize("dist,mask", [(50, 0), (50, 1), (500, 0)])
def test_keras_matches_original_spliceai(keras_annotators, dist, mask):
    import pysam
    from spliceai.utils import get_delta_scores as original_scores
    from openspliceai.variant.utils import get_delta_scores
    original, current, vcf = keras_annotators
    n_compared = 0
    n_extensions = 0
    with pysam.VariantFile(vcf) as variants:
        for record in variants:
            # Original SpliceAI does not provide numeric MNV/delins scores. Its
            # larger REF acceptance boundary also permits unsupported realignment.
            if len(record.ref) > dist + 1 or (
                len(record.ref) > 1 and any(len(alt) > 1 for alt in record.alts)
            ):
                n_extensions += 1
                continue
            expected = list(original_scores(record, original, dist, mask))
            actual = list(get_delta_scores(record, current, dist, mask,
                                           flanking_size=10000, precision=2))
            assert actual == expected, (
                f"mismatch at {record.chrom}:{record.pos} {record.ref}->{record.alts}\n"
                f"  original: {expected}\n  openspliceai: {actual}"
            )
            n_compared += len(actual)
    assert n_compared >= 5  # SNV + deletion + insertion + two alternate alleles
    assert n_extensions >= 3  # explicitly account for excluded extension fixtures
