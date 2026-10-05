"""Public scoring calls preserve streams, files, and caller inference settings."""
import gzip
import types

import pytest
import torch

from openspliceai.variant import utils
from openspliceai.variant import variant as cli
from tests.unit.test_variant_output_atomic import _args, _write_vcf, _FakeAnnotator


def test_default_inference_is_fp32_and_restores_caller_flags(monkeypatch):
    monkeypatch.delenv('OSAI_TF32', raising=False)
    monkeypatch.delenv('OSAI_CUDNN_BENCH', raising=False)
    monkeypatch.delenv('OSAI_DETERMINISTIC', raising=False)
    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.deterministic,
              torch.are_deterministic_algorithms_enabled())
    seen = []
    def inspect(*args):
        seen.append((torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
                     torch.backends.cuda.matmul.allow_tf32))
        raise RuntimeError('injected scoring error')
    annotator = types.SimpleNamespace(get_name_and_strand=inspect)
    record = types.SimpleNamespace(chrom='chr1', pos=2, ref='A', alts=['C'])
    with pytest.raises(RuntimeError, match='injected'):
        utils.get_delta_scores_batched([record], annotator, 1, 0, 80)
    after = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled())
    assert seen == [(False, False, False)]
    assert after == before


def test_variant_diagnostics_do_not_contaminate_stdout(tmp_path, monkeypatch, capsys):
    source, output = tmp_path/'source.vcf', tmp_path/'result.vcf'
    _write_vcf(source)
    monkeypatch.setattr(cli, 'Annotator', lambda *args: _FakeAnnotator())
    monkeypatch.setattr(cli, 'get_delta_scores', lambda *args: [])
    cli.variant(_args(source, output))
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'variant' in captured.err


def test_compressed_vcf_is_bgzip_and_published_atomically(tmp_path, monkeypatch):
    source, output = tmp_path/'source.vcf', tmp_path/'result.vcf.gz'
    _write_vcf(source)
    monkeypatch.setattr(cli, 'Annotator', lambda *args: _FakeAnnotator())
    monkeypatch.setattr(cli, 'get_delta_scores', lambda *args: [])
    cli.variant(_args(source, output))
    assert output.read_bytes()[:2] == b'\x1f\x8b'
    with gzip.open(output, 'rt') as handle:
        text = handle.read()
    assert text.startswith('##fileformat=VCF')
    assert len([line for line in text.splitlines() if not line.startswith('#')]) == 2


@pytest.mark.parametrize('value', ['nan', 'inf', '2.0'])
def test_invalid_numeric_annotation_is_not_published(tmp_path, value):
    output = tmp_path/'bad.vcf'
    output.write_text('##fileformat=VCFv4.2\n##contig=<ID=chr1,length=20>\n'
                      '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="scores">\n'
                      '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
                      f'chr1\t2\t.\tA\tC\t.\tPASS\tOpenSpliceAI=C|G|{value}|0|0|0|0|0|0|0\n')
    with pytest.raises(RuntimeError, match='malformed'):
        cli._validate_output(output, 1)


def test_invalid_library_arguments_raise_value_error(tmp_path):
    args = _args('missing.vcf', tmp_path/'out.vcf')
    args.ref_genome = None
    with pytest.raises(ValueError):
        cli.variant(args)
