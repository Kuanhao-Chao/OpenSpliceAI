"""Real annotation parsing and safe shard merging across scientific edge cases."""
from types import SimpleNamespace

from Bio import SeqIO
import gffutils
import h5py
import pytest

from openspliceai.create_data import create_datafile, create_dataset, utils, paralogs
from openspliceai.merge_data.merge_dataset import merge_dataset
from tests.fixtures.synthetic import MINI_GFF, MINI_GENOME, write_mini_genome_and_gff
from tests.regression.test_data_integrity import columns, encoded


@pytest.mark.parametrize('replacement,expected', [
    (('gene_biotype=protein_coding','unrelated=value'), 'gene_biotype'),
    (('\t-\t.\tID=geneM','\t.\t.\tID=geneM'), 'strand'),
    (('\texon\t1\t5','\texon\t1\t25'), 'outside'),
    (('\texon\t12\t20','\texon\t5\t20'), 'Overlapping')])
def test_invalid_gene_annotation_is_rejected(tmp_path, replacement, expected):
    fasta = tmp_path/'input.fa'
    fasta.write_text('>chr1\n'+MINI_GENOME+'\n')
    annotation = MINI_GFF.replace(*replacement)
    database = gffutils.create_db(annotation, ':memory:', from_string=True, force=True)
    try:
        with pytest.raises(ValueError, match=expected):
            create_datafile.get_sequences_and_labels(database, str(tmp_path),
                SeqIO.to_dict(SeqIO.parse(fasta,'fasta')), {'chr1':0},'test')
    finally:
        database.conn.close()


@pytest.mark.parametrize('mode,biotype', [('bad','all'),('canonical','bad')])
def test_unknown_transcript_options_fail_before_creating_outputs(tmp_path, mode, biotype):
    with pytest.raises(ValueError):
        create_datafile.get_sequences_and_labels(None,str(tmp_path),{}, {},'test',parse_type=mode,biotype=biotype)
    assert not list(tmp_path.iterdir())


def test_gtf_gene_type_and_transcript_features_are_supported(tmp_path):
    annotation = ('chr1\ttest\tgene\t1\t20\t.\t+\t.\tgene_id "g"; gene_type "protein_coding";\n'
                  'chr1\ttest\ttranscript\t1\t20\t.\t+\t.\tgene_id "g"; transcript_id "t";\n'
                  'chr1\ttest\texon\t1\t5\t.\t+\t.\tgene_id "g"; transcript_id "t";\n'
                  'chr1\ttest\texon\t12\t20\t.\t+\t.\tgene_id "g"; transcript_id "t";\n')
    database = gffutils.create_db(annotation,':memory:',from_string=True,force=True,
                                  disable_infer_genes=True,disable_infer_transcripts=True)
    fasta=tmp_path/'input.fa'
    fasta.write_text('>chr1\n'+'A'*20+'\n')
    try:
        rows=create_datafile.get_sequences_and_labels(database,str(tmp_path),
            SeqIO.to_dict(SeqIO.parse(fasta,'fasta')),{'chr1':0},'test',canonical_only=False)
        assert rows[0]==['g'] and rows[6]==['00002000000100000000']
    finally:
        database.conn.close()


def test_requested_paralogy_filter_runs_for_test_and_validation(tmp_path, monkeypatch):
    fasta,gff=write_mini_genome_and_gff(tmp_path)
    calls=[]
    def filter_split(train,held_out,identity,coverage,directory,experiment):
        calls.append(experiment)
        return train,held_out
    monkeypatch.setattr(paralogs,'remove_paralogous_sequences',filter_split)
    args=SimpleNamespace(annotation_gff=gff,genome_fasta=fasta,output_dir=str(tmp_path/'data'),
        parse_type='canonical',biotype='all',chr_split='train-test',split_method='random',
        split_ratio=.5,val_split_ratio=.5,canonical_only=False,write_fasta=False,remove_paralogs=True,
        min_identity=.8,min_coverage=.5,random_seed=42)
    create_datafile.create_datafile(args)
    assert calls==['test','validation']


def test_unsigned_corrupt_annotation_cache_is_rebuilt(tmp_path):
    gff=tmp_path/'source.gff'
    gff.write_text(MINI_GFF)
    cache=tmp_path/'annotation.db'
    cache.write_bytes(b'corrupt SQLite cache')
    database=utils.create_or_load_db(gff,cache)
    try:
        assert database['geneM'].end==20
    finally:
        database.conn.close()


@pytest.mark.parametrize('ratio', [-1,2])
def test_data_split_ratios_are_validated_before_io(ratio):
    with pytest.raises(ValueError):
        create_datafile.create_datafile(SimpleNamespace(split_ratio=ratio,val_split_ratio=.1))
    with pytest.raises(ValueError):
        utils.split_train_val(columns(),ratio)
    with pytest.raises(ValueError):
        utils.split_chromosomes({},split_ratio=ratio)


def test_empty_gene_group_split_is_supported():
    empty=[[] for _ in range(7)]
    assert utils.split_train_val(empty,1)==(empty,empty)
    with pytest.raises(ValueError):
        utils.split_train_val([[1]],.1)


def test_merge_failure_preserves_previous_output_and_source(tmp_path):
    source=tmp_path/'input'
    source.mkdir()
    with h5py.File(source/'dataset_test.h5','w') as handle:
        encoded(handle,3)
        handle['X7']=[1]  # unpaired shard must never be silently omitted
    output=tmp_path/'out'
    output.mkdir()
    destination=output/'dataset_test.h5'
    destination.write_bytes(b'previous complete file')
    args=SimpleNamespace(chr_split='test',input_dir=[source],output_dir=output)
    with pytest.raises(ValueError,match='matching'):
        merge_dataset(args)
    assert destination.read_bytes()==b'previous complete file'
    assert list(output.iterdir())==[destination]
    with pytest.raises(ValueError,match='distinct'):
        merge_dataset(SimpleNamespace(chr_split='test',input_dir=[source],output_dir=source))


@pytest.mark.parametrize('split,biotype,inputs', [('bad','all',['x']),('test','bad',['x']),('test','all',[])])
def test_merge_rejects_unknown_options(tmp_path,split,biotype,inputs):
    with pytest.raises(ValueError):
        merge_dataset(SimpleNamespace(chr_split=split,biotype=biotype,input_dir=inputs,output_dir=tmp_path))


def test_merge_includes_validation_and_rejects_mixed_presence(tmp_path):
    sources=[tmp_path/'first',tmp_path/'second']
    for source in sources:
        source.mkdir()
        for split in ('train','test','validation'):
            with h5py.File(source/f'dataset_{split}.h5','w') as handle:
                encoded(handle,9)
                handle.create_group('metadata')
    args=SimpleNamespace(chr_split='train-test',input_dir=sources,output_dir=tmp_path/'out')
    merge_dataset(args)
    with h5py.File(tmp_path/'out/dataset_validation.h5') as handle:
        assert sorted(handle)==['X0','X1','Y0','Y1']
    (sources[0]/'dataset_validation.h5').unlink()
    with pytest.raises(ValueError,match='Validation'):
        merge_dataset(args)


def test_merge_rejects_mixed_context_shapes(tmp_path):
    with h5py.File(tmp_path/'dataset_test.h5','w') as handle:
        encoded(handle,1)
        encoded(handle,2)
        del handle['X2']
        handle['X2']=handle['X1'][:,:82,:]
    with pytest.raises(ValueError,match='context'):
        merge_dataset(SimpleNamespace(chr_split='test',input_dir=[tmp_path],output_dir=tmp_path/'out'))


@pytest.mark.parametrize('biotype', ['bad'])
def test_dataset_encoder_rejects_unknown_biotype(tmp_path,biotype):
    with pytest.raises(ValueError):
        create_dataset.create_dataset(SimpleNamespace(chr_split='test',biotype=biotype,output_dir=str(tmp_path)))
