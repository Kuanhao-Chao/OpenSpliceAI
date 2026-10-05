"""Data creation accepts advertised biotypes and keeps gene isoforms together."""
import types

import h5py
import pytest
from Bio import SeqIO

from openspliceai.create_data import create_datafile, create_dataset, utils
from tests.fixtures.synthetic import write_mini_genome_and_gff


@pytest.mark.parametrize('biotype', ['all', 'non-coding'])
def test_advertised_biotypes_create_loadable_hdf5(tmp_path, biotype):
    fasta, gff = write_mini_genome_and_gff(tmp_path)
    args = types.SimpleNamespace(annotation_gff=gff, genome_fasta=fasta, output_dir=str(tmp_path),
                                 parse_type='canonical', biotype=biotype, chr_split='test',
                                 split_method='random', split_ratio=0., val_split_ratio=.1,
                                 canonical_only=False, write_fasta=False, remove_paralogs=False)
    create_datafile.create_datafile(args)
    create_dataset.create_dataset(args)
    suffix = '_ncRNA' if biotype == 'non-coding' else ''
    with h5py.File(tmp_path/f'dataset_test{suffix}.h5') as handle:
        assert handle['X0'].shape[-1] == 4
        assert handle['Y0'].shape[-1] == 3


def test_validation_keeps_all_records_of_a_gene_together():
    names = ['gene1']*8 + ['gene2']*2 + ['gene3']*2
    fields = [names]+[list(range(len(names))) for _ in range(6)]
    train, validation = utils.split_train_val(fields, .5)
    assert set(train[0]).isdisjoint(validation[0])
    assert sorted(train[1]+validation[1]) == list(range(len(names)))


def test_isoform_labels_do_not_accumulate_from_previous_transcripts(tmp_path):
    fasta, gff = write_mini_genome_and_gff(tmp_path)
    db = utils.create_or_load_db(gff, str(tmp_path/'annotation.db'))
    sequences = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    rows = create_datafile.get_sequences_and_labels(db, str(tmp_path), sequences,
             {chrom: 0 for chrom in sequences}, 'test', parse_type='all_isoforms', canonical_only=False)
    indices = [i for i, name in enumerate(rows[0]) if name == 'g3']
    transcripts = list(db.children('g3', featuretype='mRNA', order_by='start'))
    gene = db['g3']
    for index, transcript in zip(indices, transcripts):
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))
        expected = [0]*(gene.end-gene.start+1)
        for left, right in zip(exons, exons[1:]):
            expected[left.end-gene.start] = 2
            expected[right.start-gene.start] = 1
        assert rows[6][index] == ''.join(map(str, expected))
