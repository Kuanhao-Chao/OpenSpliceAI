"""Malformed inputs must fail before fitting, scoring or publishing artifacts."""
import argparse
import io
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from openspliceai import openspliceai as cli
from openspliceai.calibrate import calibrate, temperature_scaling
from openspliceai.calibrate.streaming import LogitCache
from openspliceai.checkpoints import unpack_checkpoint, calibrated_checkpoint
from openspliceai.checkpoints import validate_temperature
from openspliceai.data_schema import shard_indices, validate_encoding
from openspliceai.train_base import utils as training
from openspliceai.variant import utils as scoring
from openspliceai.variant import variant
from tests.regression.test_portable_calibration import Constant, loader


@pytest.mark.parametrize('function,value', [(cli.positive_int, '0'), (cli.positive_int, '-1'),
    (cli.nonnegative_float, 'nan'), (cli.nonnegative_float, '-1'), (cli.probability, '1.1')])
def test_cli_numeric_types_reject_invalid_values(function, value):
    with pytest.raises(argparse.ArgumentTypeError):
        function(value)


@pytest.mark.parametrize('name,value', [('distance', -1), ('distance', 5000), ('distance', 1.5),
    ('mask', 2), ('flanking_size', 100), ('precision', -1), ('precision', 13), ('precision', 1.5),
    ('batch_size', 0), ('batch_size', 1.5)])
def test_scoring_library_options_are_validated(name, value):
    options = dict(distance=50, mask=0, flanking_size=80, precision=2, batch_size=1)
    options[name] = value
    with pytest.raises(ValueError):
        scoring.validate_scoring_options(**options)


@pytest.mark.parametrize('name', ['OSAI_TF32', 'OSAI_CUDNN_BENCH', 'OSAI_DETERMINISTIC'])
def test_inference_environment_rejects_unknown_settings(monkeypatch, name):
    monkeypatch.setenv(name, 'true')
    with pytest.raises(ValueError, match='0 or 1'):
        with scoring.inference_settings():
            pass


@pytest.mark.parametrize('attributes', [{'random_seed': -1}, {'epochs': 0}, {'patience': 0}])
def test_environment_checks_iteration_and_seed_bounds(attributes):
    with pytest.raises(ValueError):
        training.setup_environment(SimpleNamespace(flanking_size=80, **attributes))


def test_validation_requires_resolvable_distinct_filename():
    with pytest.raises(ValueError, match='validation-dataset'):
        training.resolve_validation_dataset(SimpleNamespace(train_dataset='arbitrary.h5'))


@pytest.mark.parametrize('context,shape_x,shape_y', [(-1, (1,4,10002),(1,3,2)),
    (10002,(1,4,10002),(1,3,2)), (81,(1,4,10002),(1,3,2)),
    (80,(1,4,10002),(2,3,2)), (80,(1,3,10002),(1,3,2))])
def test_invalid_context_crop_fails(context, shape_x, shape_y):
    with pytest.raises(ValueError):
        training.clip_datapoints(torch.zeros(shape_x), torch.zeros(shape_y), context, 10000)


@pytest.mark.parametrize('labels,probabilities', [(torch.zeros(1,2,1), torch.zeros(1,2,1)),
    (torch.tensor([[[1.],[0.],[0.]]]), torch.tensor([[[float('nan')],[0.],[0.]]])),
    (torch.tensor([[[1.],[1.],[0.]]]), torch.ones(1,3,1)/3)])
def test_loss_rejects_invalid_class_layout_and_values(labels, probabilities):
    with pytest.raises(ValueError):
        training.categorical_crossentropy_2d(labels, probabilities)


def test_focal_vector_alpha_matches_independent_classwise_oracle():
    labels = torch.eye(3).T.unsqueeze(0)
    probabilities = torch.tensor([[[.5,.2,.1],[.3,.6,.3],[.2,.2,.6]]])
    alpha = [.1,.3,.7]
    true_prob = np.array([.5,.6,.6])
    expected = np.mean(-np.array(alpha)*(1-true_prob)**2*np.log(true_prob))
    assert training.focal_loss(labels, probabilities, alpha, 2).item() == pytest.approx(expected)


def test_loss_scheduler_and_focal_configuration_reject_unknowns():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)
    for epochs, name in ((0,'MultiStepLR'), (1,'unknown')):
        with pytest.raises(ValueError):
            training.initialize_scheduler(optimizer, epochs, name)
    with pytest.raises(ValueError, match='loss'):
        training.compute_loss(torch.ones(1,3,1)/3, torch.ones(1,3,1)/3, 'unknown')
    with pytest.raises(ValueError):
        training.configure_training_params(SimpleNamespace(random_seed=42, focal_alpha=-1), {})
    with pytest.raises(ValueError):
        training.print_topl_statistics(np.array([]), np.array([]), io.StringIO())


@pytest.mark.parametrize('keys', [('X01','Y01'), ('X0',)])
def test_shards_reject_noncanonical_or_unpaired_names(tmp_path, keys):
    with h5py.File(tmp_path/'input.h5','w') as handle:
        for key in keys:
            handle[key] = np.zeros(1)
        with pytest.raises(ValueError):
            shard_indices(handle)


@pytest.mark.parametrize('value', [-1., .5, float('inf')])
def test_encoding_rejects_fractional_negative_and_nonfinite_values(value):
    with pytest.raises(ValueError):
        validate_encoding(np.array([[value,0,0,0]]), np.array([[1,0,0]]))


def test_calibration_options_and_explicit_memory_bound(tmp_path):
    model = Constant()
    with pytest.raises(ValueError):
        temperature_scaling.ModelWithTemperature(model, 2)
    wrapper = temperature_scaling.ModelWithTemperature(model, 3)
    with pytest.raises(ValueError):
        wrapper.temperature_scale(torch.zeros(1,4))
    for epochs, patience in ((0, 1), (1, 0)):
        with pytest.raises(ValueError):
            wrapper.fit_cache(None, epochs, patience=patience)
    logits, labels = calibrate.get_logits_labels(model, loader(), 'cpu', {'CL':80})
    assert logits.shape == (25,3) and labels.shape == (25,)
    with pytest.raises(ValueError, match='memory bound'):
        calibrate.get_logits_labels(model, loader(), 'cpu', {'CL':80}, maximum_observations=10)
    with pytest.raises(ValueError):
        temperature_scaling.get_validation_loader(None, [], 0)
    with pytest.raises(ValueError):
        calibrate.calibrate(SimpleNamespace(loss='focal_loss'))


@pytest.mark.parametrize('logits', [(float('nan'),0.,0.), (0.,0.)])
def test_invalid_cached_model_output_closes_disk_files(tmp_path, logits):
    class Bad(Constant):
        def forward(self, inputs):
            return torch.tensor(logits).view(1,-1,1).expand(len(inputs),-1,5)
    with pytest.raises(ValueError):
        with LogitCache(Bad(), loader(), 'cpu', {'CL':80}, directory=tmp_path):
            pass
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('payload', [{}, {'value': 1}, [torch.ones(1)]])
def test_checkpoints_require_nonempty_tensor_mapping(payload):
    with pytest.raises(ValueError):
        unpack_checkpoint(payload, 80)


def test_nested_calibrated_checkpoints_are_rejected(model_80nt):
    payload = calibrated_checkpoint(model_80nt, torch.ones(3), 80)
    payload['state_dict'] = calibrated_checkpoint(model_80nt, torch.ones(3), 80)
    with pytest.raises(ValueError, match='Nested'):
        unpack_checkpoint(payload,80)


@pytest.mark.parametrize('dtype', [torch.int64, torch.complex64])
def test_temperature_requires_real_floating_point_values(dtype):
    with pytest.raises(ValueError):
        validate_temperature(torch.ones(3,dtype=dtype))


def test_calibrated_artifact_cannot_mislabel_its_architecture(model_80nt):
    with pytest.raises(ValueError,match='context-matched'):
        calibrated_checkpoint(model_80nt,torch.ones(3),400)


@pytest.mark.parametrize('fields', ['C|G|0', 'C||0|0|0|0|0|0|0|0', 'C|G|0|0|0|0|1.5|0|0|0'])
def test_vcf_annotation_schema_rejects_malformed_fields(tmp_path, fields):
    path = tmp_path/'output.vcf'
    path.write_text('##fileformat=VCFv4.2\n##contig=<ID=chr1,length=20>\n'
        '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="scores">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        f'chr1\t2\t.\tA\tC\t.\tPASS\tOpenSpliceAI={fields}\n')
    with pytest.raises(RuntimeError):
        variant._validate_output(path, 1)


@pytest.mark.parametrize('contents', [b'', b'not gzip', b'\x1f\x8btruncated'])
def test_compressed_vcf_integrity_is_required(tmp_path, contents):
    path = tmp_path/'broken.vcf.gz'
    path.write_bytes(contents)
    with pytest.raises(RuntimeError):
        variant._validate_output(path,1)
