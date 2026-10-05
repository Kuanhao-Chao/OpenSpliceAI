"""Evaluation preserves checkpoint state; failures release files they opened."""
import importlib
import types

import h5py
import numpy as np
import pytest
import torch

from openspliceai.calibrate.temperature_scaling import ModelWithTemperature
from openspliceai.train_base import utils as training
from tests.fixtures.synthetic import write_dataset_h5


def test_temperature_wrapper_scales_sequence_class_axis(model_80nt):
    wrapper = ModelWithTemperature(model_80nt, 3)
    wrapper.temperature.data.copy_(torch.tensor([1., 2., 4.]))
    x = torch.randn(2, 4, 120)
    with torch.no_grad():
        expected = model_80nt(x) / torch.tensor([1., 2., 4.]).view(1, 3, 1)
        actual = wrapper(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loaded_temperature_evaluation_does_not_update_batchnorm(model_80nt, tmp_path):
    from openspliceai.calibrate.temperature_scaling import get_validation_loader
    path = tmp_path / "validation.h5"
    write_dataset_h5(path, n_windows=2)
    temperature = tmp_path / "temperature.pt"
    torch.save(torch.tensor([1., 2., 3.]), temperature)
    model_80nt.train()
    before = {name: value.clone() for name, value in model_80nt.state_dict().items()}
    wrapper = ModelWithTemperature(model_80nt, 3)
    with h5py.File(path, "r") as handle:
        loader = get_validation_loader(handle, [0], 2)
        wrapper.load_temperature(temperature, loader, {"CL": 80, "N_GPUS": 2})
    assert not model_80nt.training
    for name, value in model_80nt.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_validation_disables_gradients_without_changing_values(model_80nt, tmp_path):
    path = tmp_path / "validation.h5"
    write_dataset_h5(path, n_windows=2)
    seen = []
    hook = model_80nt.register_forward_hook(lambda m, args, result: seen.append(result.requires_grad))
    before = {name: value.clone() for name, value in model_80nt.state_dict().items()}
    try:
        with h5py.File(path, "r") as handle:
            result = training.valid_epoch(
                model_80nt, handle, np.array([0]), 2, "cross_entropy_loss", torch.device("cpu"),
                {"CL": 80, "N_GPUS": 2, "RANDOM_SEED": 42},
                training.create_metric_files(str(tmp_path)), 80, "validation"
            )
        assert torch.isfinite(result)
        assert seen and not any(seen)
        for name, value in model_80nt.state_dict().items():
            assert torch.equal(value, before[name]), name
    finally:
        hook.remove()


def test_partial_dataset_open_failure_closes_previous_files(tmp_path, monkeypatch):
    path = tmp_path / "dataset_train.h5"
    write_dataset_h5(path, n_windows=2)
    opened = []
    original = h5py.File

    def track(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(training.h5py, "File", track)
    args = types.SimpleNamespace(train_dataset=str(path), test_dataset=str(tmp_path / "test.h5"))
    with pytest.raises(FileNotFoundError):
        training.load_datasets(args)
    assert opened and all(not handle.id.valid for handle in opened)


@pytest.mark.parametrize("command", ["train", "transfer"])
def test_command_initialization_failure_closes_datasets(command, tmp_path, monkeypatch):
    module = importlib.import_module(f"openspliceai.{command}.{command}")
    handles = [h5py.File(tmp_path / f"{i}.h5", "w") for i in range(3)]
    monkeypatch.setattr(module, "setup_environment", lambda args: torch.device("cpu"))
    monkeypatch.setattr(module, "load_datasets", lambda args: (*handles, 0))
    monkeypatch.setattr(module, "generate_indices", lambda *args: ([], [], []))
    if command != "calibrate":
        monkeypatch.setattr(module, "initialize_paths", lambda args: (str(tmp_path),)*4)
    init = "initialize_model_and_optim_transfer" if command == "transfer" else "initialize_model_and_optim"

    def fail(*args, **kwargs):
        raise RuntimeError("intentional initialization failure")

    monkeypatch.setattr(module, init, fail)
    args = types.SimpleNamespace(output_dir=str(tmp_path), flanking_size=80, epochs=1,
                                 scheduler="MultiStepLR", pretrained_model="missing.pt", unfreeze=1,
                                 unfreeze_all=True)
    try:
        with pytest.raises(RuntimeError, match="intentional"):
            getattr(module, command)(args)
        assert all(not handle.id.valid for handle in handles)
    finally:
        for handle in handles:
            handle.close()


def test_forgetting_setup_failure_closes_optional_datasets(tmp_path, monkeypatch):
    transfer = importlib.import_module("openspliceai.transfer.transfer")
    path = tmp_path / "genomic.h5"
    write_dataset_h5(path, n_windows=2)
    opened = []
    original = h5py.File

    def track(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(transfer.h5py, "File", track)
    args = types.SimpleNamespace(genomic_eval_dataset=str(path), distill_weight=1., distill_shards=None)
    with pytest.raises(ValueError, match="distill-shards"):
        transfer.setup_forgetting_mitigation(args, {}, torch.device("cpu"), np.array([0]), str(tmp_path))
    assert opened and all(not handle.id.valid for handle in opened)
