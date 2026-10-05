"""Optional CUDA check for the same trained model and inputs used on CPU."""
import copy

import pytest
import torch
import numpy as np
from openspliceai.model_config import model_hyperparameters
from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.calibrate.temperature_scaling import ModelWithTemperature
from openspliceai.calibrate.streaming import LogitCache
from openspliceai.checkpoints import calibrated_checkpoint, unpack_checkpoint
from openspliceai.variant import utils as scoring

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


@pytest.mark.parametrize('context', [80,400,2000,10000])
def test_cuda_four_contexts_inference_gradient_and_calibration(context, tmp_path):
    kernels, _, widths, rates, _ = model_hyperparameters(context)
    cpu = SpliceAI(kernels, widths, rates).eval()
    gpu = copy.deepcopy(cpu).cuda()
    bases = torch.randint(0, 4, (2, context+16))
    inputs = torch.nn.functional.one_hot(bases, num_classes=4).permute(0, 2, 1).float()
    with scoring.inference_settings(), torch.no_grad():
        expected = cpu(inputs)
        actual = gpu(inputs.cuda()).cpu()
        singles = torch.cat([gpu(row[None].cuda()).cpu() for row in inputs])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(singles, actual, rtol=1e-4, atol=1e-5)
    gpu.train()
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=1e-4)
    prediction = gpu(inputs.cuda())
    loss = -prediction[:, 1].log().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None)
    optimizer.step()
    gpu.apply_softmax = False
    wrapper = ModelWithTemperature(gpu, 3).cuda()
    labels = torch.zeros(2, 3, 16)
    labels[:, 0] = 1
    # Cache input follows the real stored-context contract, independent of model CL.
    stored = torch.nn.functional.pad(inputs, ((10000-context)//2,)*2)
    with LogitCache(gpu, [(stored,labels)], torch.device('cuda'), {'CL':context}, directory=tmp_path) as cache:
        wrapper.fit_cache(cache, epochs=2)
        assert wrapper.temperature.device.type == 'cuda'
        assert min(row['nll'] for row in wrapper.history) <= wrapper.history[0]['nll']
    state, temperature = unpack_checkpoint(calibrated_checkpoint(gpu, wrapper.temperature, context), context)
    assert all(value.device.type == 'cpu' for value in state.values())
    assert temperature.device.type == 'cpu'


@pytest.mark.parametrize('strand', ['+','-'])
@pytest.mark.parametrize('mask', [0,1])
def test_cuda_variant_scores_match_cpu_and_batched(tmp_path, packaged_80nt_state, monkeypatch, strand, mask):
    import pysam
    from tests.fixtures.synthetic import write_variant_inputs
    reference, annotation, vcf = write_variant_inputs(tmp_path)
    text = open(annotation).read().replace('\t+\t', '\t'+strand+'\t')
    with open(annotation,'w') as handle:
        handle.write(text)
    monkeypatch.setattr(scoring, 'setup_device', lambda:torch.device('cpu'))
    with scoring.Annotator(reference, annotation, packaged_80nt_state, 'pytorch', 80) as annotator:
        with pysam.VariantFile(vcf) as source:
            records=list(source)
        expected=[scoring.get_delta_scores(record, annotator,50,mask,80,6) for record in records]
        annotator.models=[model.cuda() for model in annotator.models]
        monkeypatch.setattr(scoring,'setup_device', lambda:torch.device('cuda'))
        sequential=[scoring.get_delta_scores(record,annotator,50,mask,80,6) for record in records]
        batched=scoring.get_delta_scores_batched(records,annotator,50,mask,80,6,4)
    for reference_rows, actual_rows, batch_rows in zip(expected,sequential,batched):
        assert len(reference_rows)==len(actual_rows)==len(batch_rows)
        for ref,actual,batch in zip(reference_rows,actual_rows,batch_rows):
            a,b,c=[value.split('|') for value in (ref,actual,batch)]
            assert a[:2]==b[:2]==c[:2] and a[6:]==b[6:]==c[6:]
            np.testing.assert_allclose(np.asarray(b[2:6],float),np.asarray(a[2:6],float),rtol=1e-4,atol=2e-5)
            np.testing.assert_allclose(np.asarray(c[2:6],float),np.asarray(a[2:6],float),rtol=1e-4,atol=2e-5)
