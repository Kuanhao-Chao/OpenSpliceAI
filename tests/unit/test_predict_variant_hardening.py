"""Hardening + window-centering checks from the predict/variant validation audit."""
import pytest
import torch

from openspliceai.constants import SL


def test_load_pytorch_models_rejects_unsupported_flanking(packaged_80nt_state):
    """Hardening: predict.load_pytorch_models must raise (not silently build an 80nt model)
    when given a flanking size outside {80,400,2000,10000}, mirroring variant/utils.py."""
    from openspliceai.predict import predict as predict_mod
    with pytest.raises(ValueError):
        predict_mod.load_pytorch_models(packaged_80nt_state, torch.device("cpu"), SL, 999)


@pytest.mark.parametrize("flank,dist", [(80, 50), (400, 50), (80, 1000)])
def test_ref_window_is_centered_on_ref_allele(variant_inputs, flank, dist):
    """Concern #4: variant's window seq[pos-wid//2-1 : pos+wid//2] places the REF allele exactly
    at index wid//2 — the check get_delta_scores relies on before scoring."""
    from pyfaidx import Fasta
    ref_path, _ann, vcf_path = variant_inputs
    fa = Fasta(ref_path, sequence_always_upper=True, rebuild=False)
    chrom = list(fa.keys())[0]
    cov = 2 * dist + 1
    wid = flank + cov
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            pos, ref = int(c[1]), c[3]
            seq = str(fa[chrom][pos - wid // 2 - 1: pos + wid // 2])
            assert len(seq) == wid
            assert seq[wid // 2: wid // 2 + len(ref)].upper() == ref.upper()
