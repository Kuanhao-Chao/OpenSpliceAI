"""Regression lock for a FALSE-POSITIVE audit finding.

An audit flagged the minus-strand donor/acceptor labeling in create_datafile.py as wrong.
It is actually CORRECT: for the '-' strand the code derives d_idx from exons[i+1].start and
a_idx from exons[i].end via the reverse-complement transform (len-1-pos), which places the
donor/acceptor at the biologically correct sites with canonical GT/AG motifs. This test pins
that behavior so it is never "fixed" into a regression.
"""
import openspliceai.create_data.create_datafile as cdf
from tests.fixtures.synthetic import build_mini_db_and_seqdict


def _labels_by_gene(tmp_path):
    db, seq_dict = build_mini_db_and_seqdict(tmp_path)
    res = cdf.get_sequences_and_labels(
        db, str(tmp_path), seq_dict, {"chr1": 0}, "test",
        parse_type="canonical", biotype="protein-coding", canonical_only=True,
    )
    NAME, CHROM, STRAND, TXS, TXE, SEQ, LABEL = res
    return {name: (strand, seq, label) for name, strand, seq, label in zip(NAME, STRAND, SEQ, LABEL)}


def test_minus_strand_donor_acceptor_positions_and_motifs(tmp_path):
    genes = _labels_by_gene(tmp_path)
    strand, seq, label = genes["geneM"]
    assert strand == "-"
    donors = [i for i, c in enumerate(label) if c == "2"]
    acceptors = [i for i, c in enumerate(label) if c == "1"]
    # Biologically-correct RC positions for the engineered 2-exon minus gene.
    assert donors == [8]
    assert acceptors == [15]
    # Canonical motifs are preserved under canonical_only=True (would be dropped if mislabeled).
    assert seq[donors[0] + 1: donors[0] + 3] == "GT"
    assert seq[acceptors[0] - 2: acceptors[0]] == "AG"


def test_plus_strand_donor_acceptor_positions_and_motifs(tmp_path):
    genes = _labels_by_gene(tmp_path)
    strand, seq, label = genes["geneP"]
    assert strand == "+"
    donors = [i for i, c in enumerate(label) if c == "2"]
    acceptors = [i for i, c in enumerate(label) if c == "1"]
    assert donors == [4]
    assert acceptors == [11]
    assert seq[donors[0] + 1: donors[0] + 3] == "GT"
    assert seq[acceptors[0] - 2: acceptors[0]] == "AG"
