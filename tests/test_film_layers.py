import torch

from openspliceai.train_base.openspliceai import ResidualUnit


def _copy_shared_state(src, dst):
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key, tensor in src_state.items():
        if key in dst_state and dst_state[key].shape == tensor.shape:
            dst_state[key] = tensor.clone()
    dst.load_state_dict(dst_state, strict=False)


def test_residual_unit_without_rbp_matches_plain():
    torch.manual_seed(0)
    plain = ResidualUnit(4, 1, 1)
    torch.manual_seed(0)
    conditioned = ResidualUnit(4, 1, 1, film_dim=2)
    _copy_shared_state(plain, conditioned)
    x = torch.randn(2, 4, 16)
    skip = torch.zeros_like(x)
    out_plain, _ = plain(x, skip)
    out_cond, _ = conditioned(x, skip, rbp_batch=None)
    assert torch.allclose(out_plain, out_cond, atol=1e-5)


def test_residual_unit_applies_film_shift():
    unit = ResidualUnit(1, 1, 1, film_dim=1)
    with torch.no_grad():
        unit.conv1.weight.zero_()
        unit.conv1.bias.zero_()
        unit.conv2.weight.zero_()
        unit.conv2.bias.zero_()
        unit.batchnorm1.weight.fill_(1.0)
        unit.batchnorm1.bias.zero_()
        unit.batchnorm2.weight.fill_(1.0)
        unit.batchnorm2.bias.zero_()
        unit.batchnorm1.running_mean.zero_()
        unit.batchnorm1.running_var.fill_(1.0)
        unit.batchnorm2.running_mean.zero_()
        unit.batchnorm2.running_var.fill_(1.0)
        unit.film.affine.weight.zero_()
        bias = torch.zeros(2)
        bias[0] = 1.0  # gamma
        bias[1] = 2.0  # beta
        unit.film.affine.bias.copy_(bias)
    x = torch.zeros(1, 1, 8)
    skip = torch.zeros_like(x)
    rbp = torch.ones(1, 1)
    out, _ = unit(x, skip, rbp_batch=rbp)
    expected = x + 2.0
    assert torch.allclose(out, expected, atol=1e-6)
