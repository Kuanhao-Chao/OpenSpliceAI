"""Optional CUDA check for the same trained model and inputs used on CPU."""
import copy

import pytest
import torch

from openspliceai.predict.predict import load_pytorch_models

pytestmark = pytest.mark.gpu


def test_cuda_model_matches_cpu_single_and_batch(packaged_80nt_state):
    cpu = load_pytorch_models(packaged_80nt_state, torch.device("cpu"), 5000, 80)[0][0]
    gpu = copy.deepcopy(cpu).to("cuda")
    inputs = torch.randn(3, 4, 180)
    with torch.no_grad():
        expected = cpu(inputs)
        actual = gpu(inputs.to("cuda")).cpu()
        singles = torch.cat([gpu(row[None].to("cuda")).cpu() for row in inputs])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(singles, actual, rtol=1e-4, atol=1e-5)
