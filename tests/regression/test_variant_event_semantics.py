"""Hand-constructed probabilities pin AG/AL/DG/DL signs, coordinates and masks."""
import types

import pysam
import pytest
import torch
from pyfaidx import Fasta

from openspliceai.variant.utils import get_delta_scores
from tests.fixtures.synthetic import write_variant_inputs


@pytest.mark.parametrize("backend", ["pytorch", "keras"])
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("mask", [0, 1])
@pytest.mark.parametrize("annotated_distance", [-2, 1])
def test_four_event_scores_have_known_values_and_genomic_offsets(tmp_path, strand, mask, annotated_distance, backend):
    reference, _, vcf = write_variant_inputs(tmp_path)
    with pysam.VariantFile(vcf) as source:
        record = next(iter(source))

    class Profile(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def predict(self, inputs):
            return self.forward(inputs).permute(0, 2, 1).numpy()

        def forward(self, inputs):
            self.calls += 1
            profile = torch.zeros(1, 3, 101)
            if self.calls == 1:  # reference probabilities
                profile[0, 1, 48] = .7
                profile[0, 2, 52] = .8
            else:  # alternative probabilities
                profile[0, 1, 51] = .9
                profile[0, 2, 49] = .6
            profile[:, 0] = 1 - profile[:, 1:].sum(dim=1)
            return profile.flip(-1) if strand == "-" else profile

    fasta = Fasta(reference)
    annotator = types.SimpleNamespace(
        keras=(backend == "keras"), models=[Profile()], ref_fasta=fasta,
        get_name_and_strand=lambda *args: (["GENE"], [strand], [0]),
        get_pos_data=lambda *args: (-5000, 5000, annotated_distance),
    )
    try:
        fields = get_delta_scores(record, annotator, 50, mask, flanking_size=80)[0].split("|")
    finally:
        fasta.close()
    # Positions are relative to the VCF's genomic coordinate for both strands.
    assert [int(value) for value in fields[6:]] == [1, -2, -1, 2]
    scores = [.9, .7, .6, .8]
    if mask:
        if annotated_distance == 1:
            scores[0] = 0.  # gains at an annotated site are suppressed
            scores[1] = 0.  # losses at an unannotated site are suppressed
        scores[3] = 0.      # donor loss at +2 is unannotated in either case
    assert [float(value) for value in fields[2:6]] == scores
