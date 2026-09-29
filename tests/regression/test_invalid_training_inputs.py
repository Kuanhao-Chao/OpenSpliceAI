"""Reject datasets/checkpoints that cannot contribute a usable training signal."""
import importlib

import numpy as np
import pytest
import torch

from openspliceai.calibrate.temperature_scaling import ModelWithTemperature, get_validation_loader
from openspliceai.train_base.utils import model_evaluation


@pytest.mark.parametrize("temperature", [torch.ones(2), torch.tensor([1., float("nan"), 1.])])
def test_invalid_temperature_vector_is_rejected(tmp_path, model_80nt, temperature):
    path = tmp_path / "temperature.pt"
    torch.save(temperature, path)
    wrapper = ModelWithTemperature(model_80nt, 3)
    with pytest.raises(ValueError, match="Temperature"):
        wrapper.load_temperature(path, [], {"CL": 80, "N_GPUS": 2})


def test_empty_evaluation_has_clear_error():
    with pytest.raises(ValueError, match="No evaluation batches"):
        model_evaluation([], [], {}, "validation", "cross_entropy_loss")


def test_empty_calibration_split_has_clear_error():
    with pytest.raises(ValueError, match="No calibration shards"):
        get_validation_loader({}, [], 2)


@pytest.mark.parametrize("target", ["student", "teacher"])
def test_transfer_rejects_checkpoint_with_no_matching_parameters(tmp_path, target):
    transfer = importlib.import_module("openspliceai.transfer.transfer")
    path = tmp_path / "empty.pt"
    torch.save({}, path)
    with pytest.raises(ValueError, match="matching parameters"):
        if target == "student":
            transfer.initialize_model_and_optim_transfer(torch.device("cpu"), 80, 10,
                "MultiStepLR", path, 1, True)
        else:
            transfer.build_frozen_teacher(torch.device("cpu"),
                {"L": 32, "W": np.asarray([11]*4), "AR": np.asarray([1]*4)}, path)
