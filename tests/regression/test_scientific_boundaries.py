"""Independent checks of padding, schedules, strict transfer and split schemas."""
import types

import h5py
import numpy as np
import pytest
import torch

from openspliceai.data_schema import shard_indices, validate_shard
from openspliceai.train_base import utils
from openspliceai.calibrate.streaming import LogitCache, ShardDataset, CalibrationStats, evaluate_cache
from openspliceai.checkpoints import unpack_checkpoint, load_checkpoint, CheckpointError
from openspliceai.transfer.transfer import initialize_model_and_optim_transfer, build_frozen_teacher
from tests.regression.test_portable_calibration import Constant


@pytest.mark.parametrize('loss', ['cross_entropy_loss', 'focal_loss'])
def test_padding_does_not_change_observed_loss_or_gradient(loss):
    labels = torch.tensor([[[0., 1.], [1., 0.], [0., 0.]]])
    logits = torch.tensor([[[.3, .8], [.6, .1], [.1, .1]]], requires_grad=True)
    reference = utils.compute_loss(labels, logits, loss)
    reference.backward()
    expected_gradient = logits.grad.clone()
    padded_labels = torch.nn.functional.pad(labels, (0, 3))
    padded_logits = torch.nn.functional.pad(logits.detach(), (0, 3), value=1/3).requires_grad_()
    actual = utils.compute_loss(padded_labels, padded_logits, loss)
    actual.backward()
    assert actual.item() == pytest.approx(reference.item())
    torch.testing.assert_close(padded_logits.grad[..., :2], expected_gradient)
    assert torch.count_nonzero(padded_logits.grad[..., 2:]) == 0
    with pytest.raises(ValueError, match='observed label'):
        utils.compute_loss(torch.zeros_like(labels), logits.detach(), loss)


def test_calibration_cache_excludes_padding():
    labels = torch.zeros(1, 3, 5)
    labels[:, 1, :2] = 1
    with LogitCache(Constant(), [(torch.zeros(1, 4, 10005), labels)], 'cpu', {'CL': 80}) as cache:
        assert len(cache) == 2
        assert cache.preview()[1].tolist() == [1, 1]


@pytest.mark.parametrize('epochs', [1, 3, 10])
def test_multistep_schedule_matches_hand_computed_epoch_decay(epochs):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.)
    scheduler = utils.initialize_scheduler(optimizer, epochs, 'MultiStepLR')
    milestones = set(range(max(1, epochs-5), epochs))
    for completed in range(1, epochs+1):
        optimizer.step()
        scheduler.step()
        expected = .5**sum(milestone <= completed for milestone in milestones)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)


def test_cosine_schedule_uses_fractional_epoch_formula():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    scheduler = utils.initialize_scheduler(optimizer, 10, 'CosineAnnealingWarmRestarts')
    for fraction in (.25, .5, .75, 1., 4.75, 5.):
        optimizer.step()
        scheduler.step(fraction)
        progress = fraction % 5
        expected = 1e-5 + (.1-1e-5)*(1+np.cos(np.pi*progress/5))/2
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)


def test_head_only_transfer_changes_head_and_preserves_frozen_buffers(packaged_80nt_state):
    model, optimizer, _, _ = initialize_model_and_optim_transfer(
        torch.device('cpu'), 80, 1, 'MultiStepLR', packaged_80nt_state, 0, False)
    assert {name for name, value in model.named_parameters() if value.requires_grad} == {'final_conv.weight', 'final_conv.bias'}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()
    prediction = model(torch.randn(2, 4, 100))
    loss = -prediction[:, 1].log().mean()
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    assert not torch.equal(after['final_conv.weight'], before['final_conv.weight'])
    for name in before:
        if not name.startswith('final_conv.'):
            assert torch.equal(after[name], before[name]), name


def test_strict_student_and_teacher_reject_cross_context_checkpoint(packaged_80nt_state):
    with pytest.raises(CheckpointError, match='incompatible'):
        initialize_model_and_optim_transfer(torch.device('cpu'), 400, 2, 'MultiStepLR',
                                            packaged_80nt_state, 0, False)
    _, _, _, params = initialize_model_and_optim_transfer(torch.device('cpu'), 400, 2, 'MultiStepLR',
                                            packaged_80nt_state, 0, False, allow_partial_checkpoint=True)
    with pytest.raises(CheckpointError, match='teacher'):
        build_frozen_teacher(torch.device('cpu'), params, packaged_80nt_state)


@pytest.mark.parametrize('field', ['train_dataset', 'test_dataset'])
def test_validation_aliases_are_rejected(tmp_path, field):
    path = str(tmp_path/'same.h5')
    with pytest.raises(ValueError, match='distinct'):
        utils.resolve_validation_dataset(types.SimpleNamespace(validation_dataset=path, **{field:path}))


def test_noncontiguous_shards_and_metadata_are_supported(tmp_path):
    with h5py.File(tmp_path/'shards.h5', 'w') as handle:
        handle.create_group('metadata')
        for index, count in ((2, 0), (7, 2)):
            handle[f'X{index}'] = np.zeros((count, 10005, 4), dtype='i1')
            handle[f'Y{index}'] = np.zeros((1, count, 5, 3), dtype='i1')
        assert shard_indices(handle) == [2, 7]
        data = ShardDataset(handle, [2, 7])
        assert len(data) == 2 and data[0][0].shape == (4, 10005)
        with pytest.raises(IndexError):
            data[2]
        with pytest.raises(ValueError, match='missing'):
            ShardDataset(handle, [9])


@pytest.mark.parametrize('x,y', [((2,5,3),(1,2,5,3)), ((2,5,4),(2,5,3)),
                                 ((2,5,4),(1,3,5,3)), ((2,6,4),(1,2,5,3)),
                                 ((2,4,4),(1,2,5,3))])
def test_invalid_shard_shapes_fail_before_training(tmp_path, x, y):
    with h5py.File(tmp_path/'invalid.h5', 'w') as handle:
        handle['X0'], handle['Y0'] = np.zeros(x), np.zeros(y)
        with pytest.raises(ValueError):
            validate_shard(handle, 0)


def test_nonfinite_checkpoint_and_corrupt_file_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='finite'):
        unpack_checkpoint({'weight':torch.tensor([float('nan')])}, 80)
    path = tmp_path/'corrupt.pt'
    path.write_text('broken')
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_cache_sizes_and_bin_counts_are_checked():
    for args in ((0,15), (30,0)):
        with pytest.raises(ValueError):
            CalibrationStats(*args)
    with pytest.raises(ValueError):
        LogitCache(None, None, 'cpu', {}, chunk_rows=0)
    with pytest.raises(ValueError):
        evaluate_cache(None, None, maximum_plot_samples=0)
