"""Regression: ``variant`` must accept an output path with no directory component.

Found during the validation audit: ``os.makedirs(os.path.dirname(output_vcf))`` raised
``FileNotFoundError: ''`` when --output-vcf was a bare filename (no directory). The fix guards the
empty-dirname case. This locks it using the fast pytorch 80nt path.
"""
import types

import pytest


@pytest.mark.integration
def test_variant_bare_output_filename(tmp_path, monkeypatch, variant_inputs, packaged_80nt_state):
    from openspliceai.variant import variant as variant_mod

    ref, ann, vcf = variant_inputs
    monkeypatch.chdir(tmp_path)                 # so the bare filename writes into the tmp dir

    args = types.SimpleNamespace(
        ref_genome=ref, annotation=ann, input_vcf=vcf,
        output_vcf="annotated.vcf",             # <-- bare filename, dirname('') == ''
        distance=50, mask=0,
        model=packaged_80nt_state, flanking_size=80,
        model_type="pytorch", precision=2,
    )
    variant_mod.variant(args)                   # must not raise FileNotFoundError

    assert (tmp_path / "annotated.vcf").exists()
