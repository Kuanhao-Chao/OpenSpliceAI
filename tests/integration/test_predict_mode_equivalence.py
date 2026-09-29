"""Streaming/storage choices must preserve every prediction value and coordinate."""
from pathlib import Path

import numpy as np
import pytest

from tests.integration.test_predict_modes import _make_args, _output_base
from openspliceai.predict.predict import predict_cli


@pytest.mark.integration
def test_four_prediction_modes_produce_identical_bed_rows(tmp_path, packaged_80nt_state):
    # Cross a 5kb model-output boundary and include a short second entry.
    rng = np.random.default_rng(24)
    sequence = "".join(rng.choice(list("ACGT"), size=5201))
    fasta = tmp_path / "input.fa"
    fasta.write_text(f">chr1\n{sequence}\n>chr2\nACGTNACGT\n")
    outputs = []
    for storage in (0, 10**9):
        for stored in (False, True):
            output = tmp_path / f"out_{storage}_{stored}"
            predict_cli(_make_args(fasta, packaged_80nt_state, output, stored, storage))
            base = Path(_output_base(output))
            beds = [(base / f"{kind}_predictions.bed").read_text().splitlines()
                    for kind in ("acceptor", "donor")]
            assert all(beds)
            outputs.append(beds)
    for actual in outputs[1:]:
        assert actual == outputs[0]
