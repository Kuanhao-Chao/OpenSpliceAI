"""Independent expectations for corrected training contracts (not legacy snapshots)."""
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from openspliceai.train_base import utils as training


@pytest.mark.parametrize('alpha,gamma', [(0.25, 0), (0.7, 1), (1, 4)])
def test_focal_value_and_gradient_against_categorical_formula(alpha, gamma):
    probabilities = torch.tensor([[[.7, .1], [.2, .8], [.1, .1]]], dtype=torch.float64,
                                 requires_grad=True)
    labels = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]], dtype=torch.float64)
    observed = training.focal_loss(labels, probabilities, alpha=alpha, gamma=gamma)
    correct = np.array([.7, .8])
    expected = np.mean(-alpha * (1 - correct)**gamma * np.log(correct))
    assert observed.item() == pytest.approx(expected, abs=1e-9)
    observed.backward()
    expected_gradient = alpha * (gamma * (1 - correct)**max(gamma-1, 0) * np.log(correct)
                                 - (1-correct)**gamma/correct) / 2
    np.testing.assert_allclose(probabilities.grad.detach().numpy()[0, [0, 1], [0, 1]],
                               expected_gradient, rtol=1e-7, atol=1e-9)


def test_focal_parameters_reject_nonfinite_or_negative_values():
    y = torch.tensor([[[1.], [0.], [0.]]])
    for kwargs in ({'gamma': -1}, {'gamma': float('nan')}, {'alpha': -1}, {'alpha': float('inf')}):
        with pytest.raises(ValueError):
            training.focal_loss(y, y, **kwargs)


def test_single_window_is_retained_during_context_crop():
    x, y = torch.zeros(1, 4, 10005), torch.zeros(1, 3, 5)
    cropped, labels = training.clip_datapoints(x, y, 80, 10000, 2)
    assert cropped.shape == (1, 4, 85)
    assert labels.shape == (1, 3, 5)


def test_shard_loader_keeps_final_sample(tmp_path):
    path = tmp_path / 'tiny.h5'
    with h5py.File(path, 'w') as handle:
        handle['X0'] = np.zeros((4, 10005, 4), dtype=np.int8)
        y = np.zeros((1, 4, 5, 3), dtype=np.int8)
        y[..., 0] = 1
        handle['Y0'] = y
        batches = list(training.load_data_from_shard(handle, 0, torch.device('cpu'), 3, {}))
    assert [len(batch[0]) for batch in batches] == [3, 1]


def test_absent_classes_keep_donor_metrics_aligned(tmp_path):
    files = training.create_metric_files(str(tmp_path))
    labels = torch.tensor([[[0., 0.], [0., 0.], [1., 1.]]])
    training.metrics(labels.clone(), labels, files, 'validation')
    from pathlib import Path
    assert float(Path(files['donor_precision']).read_text()) == 1
    assert float(Path(files['acceptor_precision']).read_text()) == 0


def test_validation_loss_uses_all_samples(tmp_path):
    files = training.create_metric_files(str(tmp_path))
    labels = torch.zeros(1200, 3, 1)
    labels[:, 0, :] = 1
    probabilities = torch.empty_like(labels)
    probabilities[:1000, 0] = .9
    probabilities[1000:, 0] = .1
    probabilities[:, 1:] = (1 - probabilities[:, :1])/2
    result = training.model_evaluation([labels], [probabilities], files,
                                       'validation', 'cross_entropy_loss')
    assert result.item() == pytest.approx(-(1000*np.log(.9)+200*np.log(.1))/1200, abs=1e-6)


def test_environment_seeds_all_initialization_generators():
    args = SimpleNamespace(flanking_size=80, random_seed=123)
    training.setup_environment(args)
    expected = (random.random(), np.random.random(), torch.rand(5))
    training.setup_environment(args)
    actual = (random.random(), np.random.random(), torch.rand(5))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
