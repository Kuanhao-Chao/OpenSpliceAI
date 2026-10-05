"""Independent calibration objectives, streaming totals and artifact round trips."""
import numpy as np
import pytest
import torch

from openspliceai.calibrate.streaming import LogitCache, evaluate_cache
from openspliceai.calibrate.temperature_scaling import ModelWithTemperature
from openspliceai.checkpoints import (atomic_torch_save, calibrated_checkpoint,
                                    unpack_checkpoint, CheckpointError)


class Constant(torch.nn.Module):
    def __init__(self, logits=(0., 0., 0.)):
        super().__init__()
        self.register_buffer('logits', torch.tensor(logits))

    def forward(self, inputs):
        return self.logits.view(1, 3, 1).expand(len(inputs), 3, inputs.shape[-1]-80)


def loader():
    y = torch.zeros(5, 3, 5)
    y[:, 1] = 1
    return [(torch.zeros(5, 4, 10005), y)]


@pytest.mark.parametrize('early_stopping,expected', [(False, 4), (True, 2)])
def test_calibration_honors_epochs_and_early_stopping(early_stopping, expected):
    model = Constant()
    wrapper = ModelWithTemperature(model, 3)
    wrapper.set_temperature(loader(), {'CL': 80}, epochs=4,
                            early_stopping=early_stopping, patience=2)
    assert len(wrapper.history)-1 == expected
    assert wrapper.observation_count == 25
    assert wrapper.history[-1]['nll'] == pytest.approx(np.log(3), abs=1e-6)


def test_chunked_gradient_matches_full_objective(tmp_path):
    fitted = []
    for chunk_size in (3, 100):
        wrapper = ModelWithTemperature(Constant((3., -1., 1.)), 3)
        with LogitCache(wrapper.model, loader(), 'cpu', {'CL': 80}, directory=tmp_path,
                        chunk_rows=chunk_size) as cache:
            wrapper.fit_cache(cache, epochs=3)
            assert min(row['nll'] for row in wrapper.history) < wrapper.history[0]['nll']
            fitted.append(wrapper.temperature.detach().clone())
        assert not list(tmp_path.iterdir())
    torch.testing.assert_close(fitted[0], fitted[1], rtol=1e-6, atol=1e-6)


def test_streamed_totals_are_exact_and_plot_sample_is_bounded(tmp_path):
    model = Constant((3., -1., 1.))
    with LogitCache(model, loader(), 'cpu', {'CL': 80}, directory=tmp_path, chunk_rows=3) as cache:
        before, after, probs, scaled, labels = evaluate_cache(cache, lambda x: x/2,
                                                            maximum_plot_samples=7)
    expected_probability = np.exp(np.array([3., -1., 1.]))
    expected_probability /= expected_probability.sum()
    assert before.count == after.count == 25
    assert before.nll == pytest.approx(-np.log(expected_probability[1]), abs=1e-6)
    np.testing.assert_allclose(before.brier, (expected_probability-np.array([0., 1., 0.]))**2,
                               atol=1e-7)
    assert all(before.curve(index)[2].sum() == 25 for index in range(3))
    assert probs.shape == scaled.shape == (7, 3)
    assert len(labels) == 7


def test_cache_failure_releases_partial_files(tmp_path):
    class Broken(Constant):
        def forward(self, inputs):
            raise RuntimeError('injected inference failure')
    with pytest.raises(RuntimeError, match='injected'):
        with LogitCache(Broken(), loader(), 'cpu', {'CL': 80}, directory=tmp_path):
            pass
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('backend', ['predict', 'variant'])
def test_calibrated_checkpoint_loads_into_actual_inference(backend, model_80nt, tmp_path):
    from openspliceai.predict.predict import load_pytorch_models as predict_load
    from openspliceai.variant.utils import load_pytorch_models as variant_load
    model_80nt.apply_softmax = False
    temperature = torch.tensor([1., 2., 4.])
    path = tmp_path/'calibrated.pt'
    atomic_torch_save(calibrated_checkpoint(model_80nt, temperature, 80), path)
    payload = torch.load(path, weights_only=True)
    assert payload['format_version'] == 1
    if backend == 'predict':
        models, _ = predict_load(str(path), torch.device('cpu'), 100, 80)
    else:
        models = variant_load(str(path), 80)
    inputs = torch.randn(2, 4, 180)
    with torch.no_grad():
        expected = torch.softmax(model_80nt(inputs)/temperature.view(1, 3, 1), dim=1)
        actual = models[0](inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(2, 100))


@pytest.mark.parametrize('field,value', [('format_version', 99), ('flanking_size', 400),
                                      ('class_names', ['donor', 'acceptor', 'non_splice']),
                                      ('temperature', torch.tensor([0., 1., 1.])),
                                      ('temperature', torch.tensor([1., float('nan'), 1.]))])
def test_portable_checkpoint_metadata_is_validated(model_80nt, field, value):
    payload = calibrated_checkpoint(model_80nt, torch.ones(3), 80)
    payload[field] = value
    with pytest.raises(ValueError):
        unpack_checkpoint(payload, 80)


def test_atomic_checkpoint_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path/'model.pt'
    path.write_bytes(b'previous complete checkpoint')
    def fail(*args, **kwargs):
        raise OSError('injected serialization failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='injected'):
        atomic_torch_save({}, path)
    assert path.read_bytes() == b'previous complete checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_library_checkpoint_raises_typed_exception(tmp_path):
    from openspliceai.variant.utils import load_pytorch_models
    with pytest.raises(CheckpointError):
        load_pytorch_models(str(tmp_path/'missing.pt'), 80)
