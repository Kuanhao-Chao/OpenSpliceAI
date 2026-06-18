"""Unit tests for predict internals: process_gff, split_fasta/create_name, the
get_sequences HDF5-vs-txt decision, and load_pytorch_models.

Every expected value here was grounded by running the function on the real input and
observing the output (no invented numbers).
"""
import os

import numpy as np
import pytest
import torch
from Bio.Seq import Seq
from pyfaidx import Fasta

import openspliceai.predict.predict as pr


# --- process_gff -------------------------------------------------------------------

def test_process_gff_reverse_complements_minus_strand(tmp_path):
    """A minus-strand gene must be written reverse-complemented; ground the RC ourselves."""
    seq = "AAACGTACGTTTTGGGGCCCCACGT"  # 25bp chr1
    fa = tmp_path / "mini.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    gff = tmp_path / "mini.gff"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\tt\tgene\t3\t12\t.\t-\t.\tID=geneM\n"
        "chr1\tt\tmRNA\t3\t12\t.\t-\t.\tID=mM;Parent=geneM\n"
        "chr1\tt\texon\t3\t12\t.\t-\t.\tID=eM;Parent=mM\n"
    )

    # output_dir is used as a literal prefix: f'{output_dir}{basename}_genes.fa'
    out_dir = str(tmp_path) + "/"
    out_fa = pr.process_gff(str(fa), str(gff), out_dir)

    assert out_fa == out_dir + "mini_genes.fa"
    assert os.path.exists(out_fa)

    lines = [l for l in open(out_fa).read().splitlines() if l]
    # header carries gene id + coords + strand
    assert lines[0] == ">geneM chr1:3-12(-)"

    # ground the expected reverse-complement of the 1-based 3..12 gene region
    gene_fwd = seq[3 - 1:12]                       # ACGTACGTTT
    expected_rc = str(Seq(gene_fwd).reverse_complement())  # AAACGTACGT
    assert lines[1] == expected_rc


def test_process_gff_includes_genomic_flank(tmp_path):
    """gene_flank>0 extends the extracted region by real genomic context on each
    side and reports the EXTENDED coordinates in the header (so BED still maps to
    genome). Regression for issue #16: a bare gene body is otherwise N-padded."""
    seq = "ACGTACGTAC" * 6  # 60bp deterministic chr1
    fa = tmp_path / "mini.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    gff = tmp_path / "mini.gff"
    gff.write_text("##gff-version 3\nchr1\tt\tgene\t21\t40\t.\t+\t.\tID=geneP\n")

    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/", gene_flank=10)
    lines = [l for l in open(out_fa).read().splitlines() if l]

    # 1-based gene 21..40, +/-10 flank -> 11..50, clamped within the 60bp contig
    assert lines[0] == ">geneP chr1:11-50(+)"
    assert lines[1] == seq[11 - 1:50]          # 40bp of real sequence, no 'N'
    assert len(lines[1]) == 40


def test_process_gff_flank_clamps_at_contig_ends(tmp_path):
    """Flanking is clamped to the contig bounds (never runs off the ends)."""
    seq = "ACGT" * 5  # 20bp chr1
    fa = tmp_path / "mini.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    gff = tmp_path / "mini.gff"
    gff.write_text("##gff-version 3\nchr1\tt\tgene\t5\t15\t.\t+\t.\tID=g\n")

    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/", gene_flank=10)
    lines = [l for l in open(out_fa).read().splitlines() if l]
    # 5-10 -> clamp to 1 ; 15+10 -> clamp to 20 (contig length)
    assert lines[0] == ">g chr1:1-20(+)"
    assert lines[1] == seq                      # whole contig


def test_process_gff_flank_minus_strand_revcomp(tmp_path):
    """With flank, a minus-strand gene's EXTENDED region is reverse-complemented."""
    seq = "AAACGTACGTTTTGGGGCCCCACGT"  # 25bp chr1 (same as the legacy test)
    fa = tmp_path / "mini.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    gff = tmp_path / "mini.gff"
    gff.write_text("##gff-version 3\nchr1\tt\tgene\t8\t17\t.\t-\t.\tID=gm\n")

    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/", gene_flank=5)
    lines = [l for l in open(out_fa).read().splitlines() if l]
    # 8-5=3 ; 17+5=22 -> extended region 3..22, reverse-complemented
    assert lines[0] == ">gm chr1:3-22(-)"
    assert lines[1] == str(Seq(seq[3 - 1:22]).reverse_complement())


def test_process_gff_default_is_legacy_bare_body(tmp_path):
    """Default gene_flank=0 preserves the original bare-gene-body behavior."""
    seq = "ACGTACGTAC" * 6
    fa = tmp_path / "mini.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    gff = tmp_path / "mini.gff"
    gff.write_text("##gff-version 3\nchr1\tt\tgene\t21\t40\t.\t+\t.\tID=geneP\n")

    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/")  # no gene_flank
    lines = [l for l in open(out_fa).read().splitlines() if l]
    assert lines[0] == ">geneP chr1:21-40(+)"
    assert lines[1] == seq[21 - 1:40]


def test_process_gff_skips_non_gene_features(tmp_path):
    """Only 'gene' features are extracted; mRNA/exon rows alone produce an empty fasta."""
    seq = "ACGT" * 20
    fa = tmp_path / "f.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    gff = tmp_path / "f.gff"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\tt\tmRNA\t1\t40\t.\t+\t.\tID=m1\n"
        "chr1\tt\texon\t1\t40\t.\t+\t.\tID=e1;Parent=m1\n"
    )
    out_fa = pr.process_gff(str(fa), str(gff), str(tmp_path) + "/")
    assert open(out_fa).read() == ""   # no gene rows -> nothing written


# --- split_fasta / create_name -----------------------------------------------------

def _read_segments(path):
    headers, seqs = [], []
    for line in open(path).read().splitlines():
        if line.startswith(">"):
            headers.append(line[1:])
        elif line.strip():
            seqs.append(line)
    return headers, seqs


def test_split_fasta_produces_overlapping_segments(tmp_path):
    """A 250bp seq split at threshold 100 with CL_max 80 yields 3 overlapping segments."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=250))
    fa = tmp_path / "long.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    genes = Fasta(str(fa), one_based_attributes=True, read_long_names=False,
                  sequence_always_upper=True)
    out = tmp_path / "split.fa"
    pr.split_fasta(genes, str(out), CL_max=80, split_fasta_threshold=100)

    headers, seqs = _read_segments(str(out))
    # grounded: 250 / 100 -> 3 windows, each padded by CL_max//2 = 40 of flanking
    assert headers == ["chr1:1-140(.)", "chr1:61-240(.)", "chr1:161-250(.)"]
    assert len(seqs) == 3
    # segments overlap: segment1 ends at 140, segment2 starts at 61 (overlap 61..140)
    assert len(seqs[0]) == 140 and len(seqs[1]) == 180 and len(seqs[2]) == 90
    # the overlapping region is identical in both adjacent segments
    assert seqs[0][60:] == seqs[1][:80]


def test_create_name_adjusts_coordinates_for_chr_pattern(tmp_path):
    """When the header matches 'chrN:start-end(strand)', segment coords are shifted by abs_start."""
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), size=250))
    fa = tmp_path / "long.fa"
    # genes-fa style long name carrying absolute coordinates on chr2
    fa.write_text(">geneX chr2:1000-2000(+)\n" + seq + "\n")

    genes = Fasta(str(fa), one_based_attributes=True, read_long_names=True,
                  sequence_always_upper=True)
    out = tmp_path / "split.fa"
    pr.split_fasta(genes, str(out), CL_max=80, split_fasta_threshold=100)

    headers, _ = _read_segments(str(out))
    # create_name: start = abs_start - 1 + start_pos ; abs_start = 1000
    # so segment relative 1-140 -> 1000..1139 ; relative 61-240 -> 1060..1239 ; 161-250 -> 1160..1249
    assert headers == [
        "geneX chr2:1000-1139(+)",
        "geneX chr2:1060-1239(+)",
        "geneX chr2:1160-1249(+)",
    ]
    # prefix and strand preserved from the original header
    assert all(h.startswith("geneX chr2:") and h.endswith("(+)") for h in headers)


def test_split_fasta_short_sequence_not_split(tmp_path):
    """A sequence <= threshold is written as a single (renamed) segment."""
    seq = "ACGT" * 10  # 40bp
    fa = tmp_path / "short.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    genes = Fasta(str(fa), one_based_attributes=True, read_long_names=False,
                  sequence_always_upper=True)
    out = tmp_path / "split.fa"
    pr.split_fasta(genes, str(out), CL_max=80, split_fasta_threshold=100)
    headers, seqs = _read_segments(str(out))
    assert headers == ["chr1:1-40(.)"]
    assert seqs == [seq]


# --- get_sequences: HDF5-vs-txt datafile decision ----------------------------------

@pytest.mark.integration
def test_get_sequences_large_threshold_writes_txt(tmp_path):
    rng = np.random.default_rng(2)
    seq = "".join(rng.choice(list("ACGT"), size=300))
    fa = tmp_path / "in.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    out_dir = str(tmp_path) + "/"
    datafile, NAME, SEQ = pr.get_sequences(
        str(fa), out_dir, CL_max=80, hdf_threshold_len=10 ** 9, split_fasta_threshold=10 ** 9
    )
    assert datafile.endswith(".txt")
    assert datafile == out_dir + "datafile.txt"
    assert NAME == ["chr1:+"]          # forward strand appended by default
    assert os.path.exists(datafile)


@pytest.mark.integration
def test_get_sequences_zero_threshold_writes_h5(tmp_path):
    rng = np.random.default_rng(2)
    seq = "".join(rng.choice(list("ACGT"), size=300))
    fa = tmp_path / "in.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    out_dir = str(tmp_path) + "/"
    datafile, NAME, SEQ = pr.get_sequences(
        str(fa), out_dir, CL_max=80, hdf_threshold_len=0, split_fasta_threshold=10 ** 9
    )
    assert datafile.endswith(".h5")
    assert datafile == out_dir + "datafile.h5"
    assert os.path.exists(datafile)


# --- get_sequences: low-context warning (issue #16) --------------------------------

def test_get_sequences_warns_on_short_sequence(tmp_path, capsys):
    """A sequence shorter than CL_max is N-padded downstream and scores poorly;
    get_sequences must warn the user (regression for issue #16)."""
    seq = "ACGT" * 10  # 40bp < CL_max=80
    fa = tmp_path / "short.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    pr.get_sequences(str(fa), str(tmp_path) + "/", CL_max=80,
                     hdf_threshold_len=10 ** 9, split_fasta_threshold=10 ** 9)

    err = capsys.readouterr().err
    assert "shorter than the model's required context" in err
    assert "CL_max=80" in err
    assert "real flanking" in err.lower() or "flanking" in err.lower()


def test_get_sequences_no_warn_when_long_enough(tmp_path, capsys):
    """A sequence at least CL_max long must NOT trigger the low-context warning."""
    rng = np.random.default_rng(7)
    seq = "".join(rng.choice(list("ACGT"), size=120))  # 120bp >= CL_max=80
    fa = tmp_path / "long.fa"
    fa.write_text(">chr1\n" + seq + "\n")

    pr.get_sequences(str(fa), str(tmp_path) + "/", CL_max=80,
                     hdf_threshold_len=10 ** 9, split_fasta_threshold=10 ** 9)

    err = capsys.readouterr().err
    assert "shorter than the model's required context" not in err


# --- load_pytorch_models -----------------------------------------------------------

def test_load_pytorch_models_single_file(packaged_80nt_state):
    """A single state_dict path -> exactly one loaded model (4th arg is the flanking size)."""
    device = torch.device("cpu")
    models, params = pr.load_pytorch_models(packaged_80nt_state, device, 5000, 80)
    assert len(models) == 1
    assert params["CL"] == 80
    assert params["SL"] == 5000


@pytest.mark.slow
def test_load_pytorch_models_directory_ensemble(packaged_80nt_dir):
    """A directory of 5 rs10-14 checkpoints -> 5 loaded models for ensembling."""
    device = torch.device("cpu")
    models, params = pr.load_pytorch_models(packaged_80nt_dir, device, 5000, 80)
    assert len(models) == 5
    # all models are SpliceAI modules in eval mode on CPU
    for m in models:
        assert not m.training
