"""Builders for tiny synthetic fixtures (HDF5 datasets, mini gffutils DBs, FASTA).

These intentionally mirror the *real* on-disk schemas used by OpenSpliceAI so that
integration tests exercise the production loaders, not a simplified stand-in.
"""
import os

import numpy as np


def random_onehot_X(n, width, rng):
    """(n, width, 4) int8 one-hot over A/C/G/T."""
    idx = rng.integers(0, 4, size=(n, width))
    X = np.zeros((n, width, 4), dtype=np.int8)
    for c in range(4):
        X[..., c] = (idx == c)
    return X


def random_onehot_Y(n, length, rng):
    """(n, length, 3) int8 one-hot; mostly class 0 with a sprinkle of acceptor(1)/donor(2)."""
    cls = np.zeros((n, length), dtype=np.int64)
    k = max(2, length // 1000)
    for i in range(n):
        pos = rng.choice(length, size=k, replace=False)
        cls[i, pos[: k // 2 or 1]] = 1
        cls[i, pos[k // 2 or 1:]] = 2
    Y = np.zeros((n, length, 3), dtype=np.int8)
    for c in range(3):
        Y[..., c] = (cls == c)
    return Y


def write_dataset_h5(path, n_windows=2, seed=0):
    """Write one shard (X0/Y0) using the production schema.

    X0 = (n, SL+CL_max, 4) int8 ; Y0 = (1, n, SL, 3) int8  (note the extra leading dim
    on Y, matching create_dataset.py and the loader's ``h5f['Y..'][0, ...]``).
    """
    import h5py
    from openspliceai.constants import SL, CL_max

    rng = np.random.default_rng(seed)
    X = random_onehot_X(n_windows, SL + CL_max, rng)
    Y = random_onehot_Y(n_windows, SL, rng)[None, ...]  # extra nesting dim
    with h5py.File(path, "w") as f:
        f.create_dataset("X0", data=X)
        f.create_dataset("Y0", data=Y)


def write_calibrate_datasets(dirpath, n_windows=2):
    """Create dataset_{train,validation,test}.h5 side by side; return {name: path}."""
    paths = {}
    for i, name in enumerate(["train", "validation", "test"]):
        p = os.path.join(str(dirpath), f"dataset_{name}.h5")
        write_dataset_h5(p, n_windows=n_windows, seed=i)
        paths[name] = p
    return paths


# --- mini annotation + genome for create-data labeling tests -------------------------

# A 50bp chr1 with two protein-coding genes engineered so canonical GT..AG motifs land
# exactly at the computed donor/acceptor positions on BOTH strands:
#   geneM (minus, 1-20): exons [1-5],[12-20]; intron 6-11 = "CTGGAC"
#   geneP (plus,  31-50): exons [31-35],[42-50]; intron 36-41 = "GTGGAG"
MINI_GENOME = ("AAAAA" + "CTGGAC" + "TTTTTTTTT"     # 1..20  (minus gene)
               + "GGGGGGGGGG"                        # 21..30 (spacer)
               + "AAAAA" + "GTGGAG" + "TTTTTTTTT")    # 31..50 (plus gene)

MINI_GFF = """##gff-version 3
chr1\ttest\tgene\t1\t20\t.\t-\t.\tID=geneM;gene_biotype=protein_coding
chr1\ttest\tmRNA\t1\t20\t.\t-\t.\tID=mRNAM;Parent=geneM
chr1\ttest\texon\t1\t5\t.\t-\t.\tID=exM1;Parent=mRNAM
chr1\ttest\texon\t12\t20\t.\t-\t.\tID=exM2;Parent=mRNAM
chr1\ttest\tgene\t31\t50\t.\t+\t.\tID=geneP;gene_biotype=protein_coding
chr1\ttest\tmRNA\t31\t50\t.\t+\t.\tID=mRNAP;Parent=geneP
chr1\ttest\texon\t31\t35\t.\t+\t.\tID=exP1;Parent=mRNAP
chr1\ttest\texon\t42\t50\t.\t+\t.\tID=exP2;Parent=mRNAP
"""


def build_mini_db_and_seqdict(tmp_dir):
    """Return (gffutils DB, seq_dict) for the mini two-gene annotation above."""
    import gffutils
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    gff_path = os.path.join(str(tmp_dir), "mini.gff")
    with open(gff_path, "w") as fh:
        fh.write(MINI_GFF)
    db = gffutils.create_db(gff_path, ":memory:", force=True, keep_order=True,
                            merge_strategy="merge")
    seq_dict = {"chr1": SeqRecord(Seq(MINI_GENOME), id="chr1")}
    return db, seq_dict


# --- tiny train/transfer datasets ---------------------------------------------------

def write_train_datasets(dirpath, n_windows=40):
    """Create dataset_{train,validation,test}.h5 sized for a real 1-epoch train/transfer run.

    The model batch size for the 80nt config is 36 (18 * N_GPUS=2) and the DataLoader uses
    drop_last=True, so each shard needs >= 36 windows to yield at least one batch. Returns
    {name: path}.
    """
    paths = {}
    for i, name in enumerate(["train", "validation", "test"]):
        p = os.path.join(str(dirpath), f"dataset_{name}.h5")
        write_dataset_h5(p, n_windows=n_windows, seed=100 + i)
        paths[name] = p
    return paths


# --- mini genome + GFF on disk for the full create-data CLI --------------------------

def _rand_seq(length, rng):
    return "".join(rng.choice(list("ACGT"), size=length))


def write_mini_genome_and_gff(dirpath, seed=0):
    """Write a small ``mini.fa`` + ``mini.gff`` (two chromosomes, several multi-exon genes on
    both strands, one multi-transcript gene, one non-coding gene) that drives the full
    ``create-data`` CLI to real ``dataset_*.h5`` files. Labels are placed at every exon junction
    (motifs only matter under ``--canonical-only``). Returns (fasta_path, gff_path).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    chroms = {"chr1": _rand_seq(3000, rng), "chr2": _rand_seq(3000, rng)}

    fasta_path = os.path.join(str(dirpath), "mini.fa")
    with open(fasta_path, "w") as fh:
        for name, seq in chroms.items():
            fh.write(f">{name}\n{seq}\n")

    # GFF3: gene -> mRNA(s) -> exon(s). gene_biotype gates --biotype filtering.
    lines = ["##gff-version 3"]

    def gene(chrom, gid, start, end, strand, biotype, transcripts):
        lines.append(f"{chrom}\tt\tgene\t{start}\t{end}\t.\t{strand}\t.\tID={gid};gene_biotype={biotype}")
        feat = "mRNA" if biotype == "protein_coding" else "lnc_RNA"
        for tid, exons in transcripts:
            lines.append(f"{chrom}\tt\t{feat}\t{start}\t{end}\t.\t{strand}\t.\tID={tid};Parent={gid}")
            for j, (es, ee) in enumerate(exons):
                lines.append(f"{chrom}\tt\texon\t{es}\t{ee}\t.\t{strand}\t.\tID={tid}.e{j};Parent={tid}")

    gene("chr1", "g1", 100, 800, "+", "protein_coding", [("g1.t1", [(100, 300), (500, 800)])])
    gene("chr1", "g2", 1000, 1800, "-", "protein_coding", [("g2.t1", [(1000, 1300), (1500, 1800)])])
    # multi-transcript gene (exercises parse_type all_isoforms vs canonical longest)
    gene("chr1", "g3", 2000, 2800, "+", "protein_coding",
         [("g3.t1", [(2000, 2300), (2500, 2800)]), ("g3.t2", [(2000, 2200), (2400, 2800)])])
    gene("chr2", "g4", 100, 900, "+", "protein_coding", [("g4.t1", [(100, 400), (600, 900)])])
    gene("chr2", "g5", 1100, 1900, "-", "protein_coding", [("g5.t1", [(1100, 1400), (1600, 1900)])])
    # non-coding gene (excluded under --biotype protein-coding)
    gene("chr2", "g6", 2100, 2900, "+", "lncRNA", [("g6.t1", [(2100, 2400), (2600, 2900)])])

    gff_path = os.path.join(str(dirpath), "mini.gff")
    with open(gff_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return fasta_path, gff_path


# --- variant inputs (reference + annotation + VCF) -----------------------------------

def write_variant_inputs(dirpath, seed=0):
    """Write ``ref.fa`` (~12kb so even the 10000nt keras window fits), a custom annotation TSV,
    and ``variants.vcf`` containing an SNV, an insertion, a deletion and a multi-allelic record,
    all near position 6000 (inside the gene and far from the chromosome ends). The VCF REF
    alleles are read back from the generated reference so they match. Returns
    (ref_path, ann_path, vcf_path).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ref = list(_rand_seq(12000, rng))  # mutable so we can read exact bases
    seq = "".join(ref)

    ref_path = os.path.join(str(dirpath), "ref.fa")
    with open(ref_path, "w") as fh:
        fh.write(">chr_test\n" + seq + "\n")

    # one + strand gene spanning the variant region; 2 exons => an internal intron boundary
    ann_path = os.path.join(str(dirpath), "annotation.tsv")
    with open(ann_path, "w") as fh:
        fh.write("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n")
        # TX_START is 0-based in the table (code adds +1); exons as comma-lists with trailing comma
        fh.write("GENE1\tchr_test\t+\t999\t9000\t999,5999,\t5000,9000,\n")

    def base(pos1):  # 1-based -> base char
        return seq[pos1 - 1]

    p_snv, p_del, p_ins, p_multi = 6000, 6100, 6200, 6300
    alt_snv = "A" if base(p_snv) != "A" else "C"
    _others = [b for b in "ACGT" if b != base(p_multi)]
    alt_multi = f"{_others[0]},{_others[1]}"
    vcf_path = os.path.join(str(dirpath), "variants.vcf")
    with open(vcf_path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr_test,length=12000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write(f"chr_test\t{p_snv}\t.\t{base(p_snv)}\t{alt_snv}\t.\t.\t.\n")                  # SNV
        fh.write(f"chr_test\t{p_del}\t.\t{base(p_del)}{base(p_del + 1)}\t{base(p_del)}\t.\t.\t.\n")  # deletion
        fh.write(f"chr_test\t{p_ins}\t.\t{base(p_ins)}\t{base(p_ins)}T\t.\t.\t.\n")              # insertion
        fh.write(f"chr_test\t{p_multi}\t.\t{base(p_multi)}\t{alt_multi}\t.\t.\t.\n")             # multi-allelic
    return ref_path, ann_path, vcf_path
