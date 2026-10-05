"""Unrelated contigs/genes cannot change scores for an exact existing contig."""
import pysam
import pytest

from openspliceai.variant.utils import Annotator, get_delta_scores, get_delta_scores_batched
from tests.fixtures.synthetic import write_variant_inputs
from openspliceai.variant.utils import _resolve_chrom


@pytest.mark.parametrize('source,contigs,expected', [('1',{'chr1'},'chr1'),
    ('chr1',{'1'},'1'), ('chr1',{'chr1','1'},'chr1'), ('NC_001',{'chr1'},'NC_001')])
def test_chr_alias_only_applies_to_existing_contig(source,contigs,expected):
    assert _resolve_chrom(source,contigs)==expected


@pytest.mark.parametrize('batched', [False,True])
def test_mixed_prefix_reference_and_annotation_preserve_existing_scores(tmp_path,packaged_80nt_state,batched):
    reference,annotation,vcf=write_variant_inputs(tmp_path)
    for filename in (reference,annotation,vcf):
        with open(filename) as handle:
            text=handle.read().replace('chr_test','NC_000001.1')
        with open(filename,'w') as handle:
            handle.write(text)
    with pysam.VariantFile(vcf) as handle:
        records=list(handle)
    def score(ref,ann):
        with Annotator(ref,ann,packaged_80nt_state,'pytorch',80) as scorer:
            if batched:
                return get_delta_scores_batched(records,scorer,50,0,80,6,4)
            return [get_delta_scores(record,scorer,50,0,80,6) for record in records]
    expected=score(reference,annotation)
    assert len(expected[0])==1
    mixed_reference=tmp_path/'mixed.fa'
    with open(reference) as source:
        mixed_reference.write_text('>chr1\n'+'A'*12000+'\n'+source.read())
    with open(annotation) as source:
        lines=source.read().splitlines()
    mixed_annotation=tmp_path/'mixed.tsv'
    mixed_annotation.write_text(lines[0]+'\nDUMMY\tchr1\t+\t999\t9000\t999,5999,\t5000,9000,\n'+lines[1]+'\n')
    assert score(str(mixed_reference),str(mixed_annotation))==expected
