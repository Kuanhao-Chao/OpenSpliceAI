"""Independent checks for remaining calibration API execution gaps."""
import h5py
import numpy as np
import torch

from openspliceai.calibrate.calibrate_utils import calculate_brier_scores
from openspliceai.calibrate.temperature_scaling import (
    ModelWithTemperature, _ECELoss, load_data_from_shard,
)
from openspliceai.checkpoints import CalibratedSpliceAI


def test_brier_reports_background_and_absent_classes_in_fixed_order():
    labels = np.array([0, 0])
    probabilities = np.array([[.8, .1, .1], [.6, .3, .1]])
    perfect = np.array([[1., 0., 0.], [1., 0., 0.]])
    before, after = calculate_brier_scores(labels, probabilities, perfect)
    # Hand-computed binary probability errors; absent positives still contribute.
    np.testing.assert_allclose(before, [.1, .05, .01], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(after, [0., 0., 0.])


def test_legacy_calibration_reader_preserves_base_and_label_positions(tmp_path):
    path = tmp_path/'legacy.h5'
    with h5py.File(path, 'w') as source:
        source['X7'] = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]])
        source['Y7'] = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        inputs, labels = load_data_from_shard(source, 7)
    assert inputs.shape == (1, 4, 3) and labels.shape == (1, 3, 3)
    np.testing.assert_array_equal(inputs[0, :, 2], [0, 0, 0, 1])
    np.testing.assert_array_equal(labels[0], np.eye(3))


def test_calibrated_model_exposes_architecture_context(model_80nt):
    wrapper = CalibratedSpliceAI(model_80nt, torch.ones(3))
    assert wrapper.CL == 80
    with torch.no_grad():
        assert wrapper(torch.zeros(1, 4, 101)).shape == (1, 3, 21)


def test_preview_metrics_are_labeled_and_match_hand_computed_oracle(capsys):
    wrapper = ModelWithTemperature(torch.nn.Identity(), 3)
    # Uniform logits have NLL log(3), accuracy 1/2 and confidence 1/3.
    wrapper.logits = torch.zeros(2, 3)
    wrapper.labels = torch.tensor([0, 1])
    wrapper._compute_and_log_metrics(torch.nn.CrossEntropyLoss(), _ECELoss(), 'After')
    assert capsys.readouterr().out.strip() == 'After (preview) - NLL: 1.0986, ECE: 0.1667'
