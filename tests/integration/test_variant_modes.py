"""End-to-end integration tests for the ``variant`` subcommand.

Drives ``variant.variant(args)`` (the real CLI entry) over the synthetic ``variant_inputs``
fixture: a ~12kb reference, a custom annotation TSV (one + strand gene over the variant
region), and a VCF with an SNV / deletion / insertion / multi-allelic record near pos 6000.
All assertions were grounded by first running the flow and observing the output VCF.
"""
import types

import pytest


def _count_annotated(out_vcf):
    """Return (#records-with-OpenSpliceAI, list-of-all-score-strings)."""
    import pysam
    n = 0
    score_strings = []
    for rec in pysam.VariantFile(out_vcf):
        if "OpenSpliceAI" in rec.info:
            n += 1
            val = rec.info["OpenSpliceAI"]            # tuple of '|'-joined strings
            score_strings.extend(val)
    return n, score_strings


@pytest.mark.integration
def test_variant_cli_pytorch_single_model(tmp_path, variant_inputs, packaged_80nt_state):
    from openspliceai.variant import variant as variant_mod

    ref, ann, vcf = variant_inputs
    out = tmp_path / "out" / "annotated.vcf"

    args = types.SimpleNamespace(
        ref_genome=ref,
        annotation=ann,
        input_vcf=vcf,
        output_vcf=str(out),
        distance=50,
        mask=0,
        model=packaged_80nt_state,
        flanking_size=80,
        model_type="pytorch",
        precision=2,
    )
    variant_mod.variant(args)

    assert out.exists()
    n, scores = _count_annotated(str(out))
    assert n >= 1
    # at least one annotation splits into exactly 10 '|'-separated fields
    assert any(len(s.split("|")) == 10 for s in scores)
    # ALLELE|SYMBOL|... with the custom gene symbol
    assert all(s.split("|")[1] == "GENE1" for s in scores)


@pytest.mark.integration
@pytest.mark.slow
def test_variant_cli_pytorch_ensemble_dir(tmp_path, variant_inputs, packaged_80nt_dir):
    """Pointing --model at a directory ensembles all 5 rs10-14 checkpoints and still annotates."""
    from openspliceai.variant import variant as variant_mod

    ref, ann, vcf = variant_inputs
    out = tmp_path / "out_ens" / "annotated.vcf"

    args = types.SimpleNamespace(
        ref_genome=ref,
        annotation=ann,
        input_vcf=vcf,
        output_vcf=str(out),
        distance=50,
        mask=0,
        model=packaged_80nt_dir,
        flanking_size=80,
        model_type="pytorch",
        precision=2,
    )
    variant_mod.variant(args)

    assert out.exists()
    n, scores = _count_annotated(str(out))
    assert n >= 1
    assert any(len(s.split("|")) == 10 for s in scores)


@pytest.mark.integration
@pytest.mark.keras
@pytest.mark.slow
def test_variant_keras_delta_score_snv(variant_inputs, repo_root):
    """The bundled original SpliceAI Keras 10000nt model scores the SNV.

    Uses a SINGLE keras model (model_type='keras' with one .h5 file) rather than the full
    5-model 'SpliceAI' ensemble — the keras path is the heaviest test (TF + a 10000nt network),
    and one model exercises the same code path at ~1/5 the cost. The ``keras`` marker auto-skips
    when tensorflow is absent; we also guard with a skip if the bundled weights are missing.
    """
    import pysam

    single_model = repo_root / "models" / "spliceai" / "SpliceAI_models_release" / "spliceai1.h5"
    if not single_model.exists():
        pytest.skip("bundled SpliceAI Keras .h5 model not found")

    from openspliceai.variant.utils import Annotator, get_delta_scores

    ref, ann, vcf = variant_inputs
    try:
        annotator = Annotator(ref, ann, model_path=str(single_model), model_type="keras", CL=10000)
    except Exception as e:  # tensorflow import / model load failure
        pytest.skip(f"keras Annotator could not be built: {e}")

    assert annotator.keras is True
    assert len(annotator.models) == 1

    snv = next(iter(pysam.VariantFile(vcf)))     # chr_test 6000 C -> A
    scores = get_delta_scores(snv, annotator, 50, 0, flanking_size=10000)
    assert len(scores) == 1
    assert len(scores[0].split("|")) == 10
    assert scores[0].split("|")[0] == "A"
