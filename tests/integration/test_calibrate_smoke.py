"""End-to-end smoke test proving Fix #1: `calibrate` runs (it previously crashed on launch).

The crash was in the dataset-loading / index-generation lines (calibrate.py:124-125,142), which
run *before* the temperature branch. We exercise them via the fast ``--temperature-file`` path
(loading a fixed temperature vector) to avoid the slow 2000-epoch optimisation while still running
the full load -> indices -> loaders -> evaluate/visualize pipeline.
"""
import types

import pytest
import torch


@pytest.mark.integration
@pytest.mark.slow
def test_calibrate_runs_end_to_end(tmp_path, calibrate_datasets, packaged_80nt_state):
    from openspliceai.calibrate import calibrate as cal

    # Pre-made temperature vector (num_classes=3) so calibrate takes the fast load path.
    temp_file = tmp_path / "temperature.pt"
    torch.save(torch.ones(3), temp_file)

    out = tmp_path / "calout"
    args = types.SimpleNamespace(
        output_dir=str(out),
        flanking_size=80,
        train_dataset=calibrate_datasets["train"],
        test_dataset=calibrate_datasets["test"],
        pretrained_model=packaged_80nt_state,
        temperature_file=str(temp_file),
        random_seed=42,
    )
    # Before Fix #1 this raised ValueError/AttributeError on calibrate.py:124-125.
    cal.calibrate(args)

    # Artifacts written on the load-temperature path.
    assert (out / "calibrated_model.pt").exists()
    assert (out / "temperature.txt").exists()
