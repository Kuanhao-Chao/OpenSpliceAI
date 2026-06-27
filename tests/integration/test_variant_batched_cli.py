"""End-to-end coverage for the ``variant`` subcommand's batched-inference path (the v0.0.7
``--batch-size`` feature) and its argument guards, which the default-batch_size integration
test does not exercise."""
import types

import pytest


@pytest.mark.integration
def test_variant_cli_batched_path_writes_annotations(tmp_path, variant_inputs, packaged_80nt_state):
    from openspliceai.variant import variant as variant_mod
    ref, ann, vcf = variant_inputs
    out = tmp_path / "annotated.vcf"
    args = types.SimpleNamespace(
        input_vcf=str(vcf), output_vcf=str(out), ref_genome=str(ref), annotation=str(ann),
        model=str(packaged_80nt_state), flanking_size=80, distance=50, mask=0,
        model_type="pytorch", precision=2, batch_size=4)   # batch_size>1 -> batched buffering path
    variant_mod.variant(args)
    assert out.exists()
    text = out.read_text()
    assert "##INFO=<ID=OpenSpliceAI" in text          # header line added
    assert "OpenSpliceAI=" in text                    # at least one annotated record


def test_variant_cli_missing_required_arg_exits(tmp_path):
    from openspliceai.variant import variant as variant_mod
    args = types.SimpleNamespace(
        input_vcf=None, output_vcf=str(tmp_path / "o.vcf"), ref_genome="r", annotation="a",
        model="m", flanking_size=80, distance=50, mask=0, model_type="pytorch",
        precision=2, batch_size=1)
    with pytest.raises(SystemExit):
        variant_mod.variant(args)


def test_variant_cli_bad_input_vcf_exits(tmp_path):
    from openspliceai.variant import variant as variant_mod
    junk = tmp_path / "not_a_vcf.txt"
    junk.write_text("this is not a VCF\n")
    args = types.SimpleNamespace(
        input_vcf=str(junk), output_vcf=str(tmp_path / "o.vcf"), ref_genome="r", annotation="a",
        model="m", flanking_size=80, distance=50, mask=0, model_type="pytorch",
        precision=2, batch_size=1)
    with pytest.raises(SystemExit):
        variant_mod.variant(args)
