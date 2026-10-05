"""Split windows retain context but own each biological output position once."""
from io import StringIO

from pyfaidx import Fasta
import torch

from openspliceai.predict import predict


def scores(length):
    return torch.full((1, 3, length), 1/3)


def test_split_chr_fasta_emits_each_donor_once(tmp_path):
    original = tmp_path/'input.fa'
    original.write_text('>chr1\n'+'A'*250+'\n')
    split = tmp_path/'split.fa'
    with Fasta(original) as records:
        predict.split_fasta(records, split, 80, 100)
    acceptors, donors = StringIO(), StringIO()
    with Fasta(split, read_long_names=True) as records:
        for record in records:
            predict.write_batch_to_bed(record.long_name+':+', scores(len(record)), acceptors, donors, .1)
    rows = [row.split('\t') for row in donors.getvalue().splitlines()]
    assert len(rows) == 250
    assert {int(row[1]) for row in rows} == set(range(250))
    assert all(row[0] == 'chr1' and len(row) == 6 for row in rows)


def test_minus_strand_split_coordinates_follow_reverse_complement(tmp_path):
    original = tmp_path/'gene.fa'
    original.write_text('>geneX chr2:1000-1249(-)\n'+'A'*250+'\n')
    split = tmp_path/'split.fa'
    with Fasta(original, read_long_names=True) as records:
        predict.split_fasta(records, split, 80, 100)
    first = split.read_text().splitlines()[0][1:].split(' OSAI_', 1)[0]
    assert first == 'geneX chr2:1110-1249(-)'


def test_padding_beyond_declared_sequence_length_is_not_emitted():
    acceptors, donors = StringIO(), StringIO()
    predict.write_batch_to_bed('gene chr1:101-103(+):+', scores(10), acceptors, donors, .1)
    assert [int(line.split('\t')[1]) for line in donors.getvalue().splitlines()] == [100, 101, 102]


def test_non_chr_contig_keeps_genomic_coordinates():
    acceptors, donors = StringIO(), StringIO()
    predict.write_batch_to_bed('gene NC_0123.1:101-110(+):+', scores(10), acceptors, donors, .1)
    first = donors.getvalue().splitlines()[0].split('\t')
    assert first[0] == 'NC_0123.1'
    assert first[1:3] == ['100', '101']
