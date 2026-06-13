"""Unit tests for create-data stage-1 helpers in
openspliceai/create_data/{utils.py,create_datafile.py}.

Every expected value here was grounded by running the function on the real fixture
inputs (the mini gffutils DBs from tests.fixtures.synthetic) before asserting.
"""
import os
from collections import namedtuple

import gffutils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import openspliceai.create_data.utils as cdu
import openspliceai.create_data.create_datafile as cdf
from tests.fixtures.synthetic import (
    MINI_GFF,
    build_mini_db_and_seqdict,
    write_mini_genome_and_gff,
)

Rec = namedtuple("Rec", ["seq"])


# --- utils.check_and_count_motifs ----------------------------------------------------

def test_check_and_count_motifs_counts_donor_and_acceptor():
    # Donor (label 2) reads seq[i+1:i+3]; acceptor (label 1) reads seq[i-2:i].
    #        index: 0123456789
    seq = "AAGTCCAGTT"
    labels = [0, 2, 0, 0, 0, 1, 0, 0, 0, 0]   # donor at idx1 -> 'GT', acceptor at idx5 -> 'TC'
    donor_counts, acceptor_counts = {}, {}
    cdu.check_and_count_motifs(seq, labels, donor_counts, acceptor_counts)
    assert donor_counts == {"GT": 1}
    assert acceptor_counts == {"TC": 1}


def test_check_and_count_motifs_accumulates_repeats():
    # Donor at index i reads seq[i+1:i+3]. Place two donors so both read 'GT'.
    #        index: 01234567
    seq = "AGTAGTAA"
    labels = [2, 0, 0, 2, 0, 0, 0, 0]   # idx0 -> seq[1:3]='GT'; idx3 -> seq[4:6]='GT'
    donor_counts, acceptor_counts = {}, {}
    cdu.check_and_count_motifs(seq, labels, donor_counts, acceptor_counts)
    assert donor_counts == {"GT": 2}
    assert acceptor_counts == {}


# --- utils.get_chromosome_lengths ----------------------------------------------------

def test_get_chromosome_lengths():
    seq_dict = {"chr1": Rec("A" * 100), "chr2": Rec("A" * 50), "chrX": Rec("A" * 7)}
    assert cdu.get_chromosome_lengths(seq_dict) == {"chr1": 100, "chr2": 50, "chrX": 7}


# --- utils.create_or_load_db ---------------------------------------------------------

def test_create_or_load_db_creates_then_loads(tmp_path):
    gff = tmp_path / "mini.gff"
    gff.write_text(MINI_GFF)
    db_file = str(tmp_path / "mini.gff_db")

    assert not os.path.exists(db_file)
    db1 = cdu.create_or_load_db(str(gff), db_file=db_file)
    assert os.path.exists(db_file)
    n1 = sum(1 for _ in db1.all_features())

    # Second call must *load* the existing file (FeatureDB), not recreate it.
    db2 = cdu.create_or_load_db(str(gff), db_file=db_file)
    assert isinstance(db2, gffutils.FeatureDB)
    n2 = sum(1 for _ in db2.all_features())

    # MINI_GFF defines 2 genes, 2 mRNAs, 4 exons = 8 features.
    assert n1 == n2 == 8


# --- create_datafile.get_sequences_and_labels ----------------------------------------

def _all_chroms(seq_dict):
    return {c: 0 for c in seq_dict}


def test_get_sequences_and_labels_canonical_picks_longest_transcript(tmp_path):
    """parse_type='canonical' keeps only the single longest transcript per gene."""
    fasta, gff = write_mini_genome_and_gff(tmp_path)
    db = cdu.create_or_load_db(gff, db_file=gff + "_db")
    from Bio import SeqIO
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    out = tmp_path / "out"
    out.mkdir()

    data = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, _all_chroms(seq_dict), "test",
        parse_type="canonical", biotype="protein-coding",
        canonical_only=False, write_fasta=False,
    )
    names = data[0]
    # g6 is lncRNA (excluded); g3 has two transcripts -> canonical yields exactly one row.
    assert names == ["g1", "g2", "g3", "g4", "g5"]
    assert names.count("g3") == 1


def test_get_sequences_and_labels_all_isoforms_emits_each_transcript(tmp_path):
    """parse_type='all_isoforms' emits one SEQ row per transcript (g3 -> 2 rows)."""
    fasta, gff = write_mini_genome_and_gff(tmp_path)
    db = cdu.create_or_load_db(gff, db_file=gff + "_db")
    from Bio import SeqIO
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    out = tmp_path / "out"
    out.mkdir()

    data = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, _all_chroms(seq_dict), "test",
        parse_type="all_isoforms", biotype="protein-coding",
        canonical_only=False, write_fasta=False,
    )
    names = data[0]
    assert names == ["g1", "g2", "g3", "g3", "g4", "g5"]
    assert names.count("g3") == 2
    # all_isoforms produces strictly more rows than canonical for this gene set.
    assert len(names) == 6


def test_biotype_protein_coding_excludes_lncRNA(tmp_path):
    """biotype='protein-coding' drops the lncRNA gene g6; 'non-coding' keeps only g6."""
    fasta, gff = write_mini_genome_and_gff(tmp_path)
    db = cdu.create_or_load_db(gff, db_file=gff + "_db")
    from Bio import SeqIO
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    out = tmp_path / "out"
    out.mkdir()

    pc = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, _all_chroms(seq_dict), "test",
        parse_type="canonical", biotype="protein-coding",
        canonical_only=False, write_fasta=False,
    )
    assert "g6" not in pc[0]

    nc = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, _all_chroms(seq_dict), "test",
        parse_type="canonical", biotype="non-coding",
        canonical_only=False, write_fasta=False,
    )
    assert nc[0] == ["g6"]


def test_two_exon_gene_marks_donor_and_acceptor(tmp_path):
    """The mini two-gene DB (one + and one - strand, each 2 exons) marks both labels."""
    db, seq_dict = build_mini_db_and_seqdict(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    data = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, {"chr1": 0}, "test",
        parse_type="canonical", biotype="protein-coding",
        canonical_only=False, write_fasta=False,
    )
    assert data[0] == ["geneM", "geneP"]
    for label_str in data[6]:
        assert "1" in label_str  # acceptor
        assert "2" in label_str  # donor


def test_single_exon_gene_yields_zero_splice_labels(tmp_path):
    """A gene with a single exon has no intron -> no donor/acceptor labels."""
    single_gff = (
        "##gff-version 3\n"
        "chr1\ttest\tgene\t1\t20\t.\t+\t.\tID=gS;gene_biotype=protein_coding\n"
        "chr1\ttest\tmRNA\t1\t20\t.\t+\t.\tID=mS;Parent=gS\n"
        "chr1\ttest\texon\t1\t20\t.\t+\t.\tID=eS1;Parent=mS\n"
    )
    gff_path = tmp_path / "single.gff"
    gff_path.write_text(single_gff)
    db = gffutils.create_db(str(gff_path), ":memory:", force=True,
                            keep_order=True, merge_strategy="merge")
    seq_dict = {"chr1": SeqRecord(Seq("A" * 20), id="chr1")}
    out = tmp_path / "out"
    out.mkdir()

    data = cdf.get_sequences_and_labels(
        db, str(out), seq_dict, {"chr1": 0}, "test",
        parse_type="canonical", biotype="protein-coding",
        canonical_only=False, write_fasta=False,
    )
    assert data[0] == ["gS"]
    label_str = data[6][0]
    assert set(label_str) == {"0"}        # all no-splice
    assert "1" not in label_str and "2" not in label_str
