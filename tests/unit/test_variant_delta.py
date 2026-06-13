"""Unit tests for openspliceai/variant/utils.py delta-score machinery.

These run the *real* PyTorch ``Annotator`` (built once from a packaged 80nt state_dict on
CPU) over the synthetic ``variant_inputs`` fixture (~12kb ref + custom TSV + SNV / deletion /
insertion / multi-allelic VCF near pos 6000). Every expected value here was grounded by first
running the function on the fixture and observing the actual output (number of scores, field
counts, which mask fields zero out).
"""
import os

import pytest


# ---------------------------------------------------------------------------------------
# A module-scoped PyTorch Annotator built once over the variant_inputs fixture.
# Building it per-test would reload the checkpoint every time; the fixture is deterministic
# so a single shared instance is safe. We re-derive the fixture inside the (session) tmp so
# we don't depend on the function-scoped ``variant_inputs`` fixture for the shared object.
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pytorch_annotator(tmp_path_factory, repo_root):
    from tests.fixtures.synthetic import write_variant_inputs
    from openspliceai.variant.utils import Annotator

    state = repo_root / "models" / "openspliceai-honeybee" / "80nt" / "model_80nt_rs10.pt"
    if not state.exists():
        pytest.skip(f"packaged 80nt checkpoint not found: {state}")

    d = tmp_path_factory.mktemp("variant_ann")
    ref, ann, vcf = write_variant_inputs(d)
    annotator = Annotator(ref, ann, model_path=str(state), model_type="pytorch", CL=80)
    return annotator, ref, ann, vcf


def _records(vcf_path):
    import pysam
    return list(pysam.VariantFile(vcf_path))


def test_annotator_loads_single_pytorch_model(pytorch_annotator):
    annotator, *_ = pytorch_annotator
    assert annotator.keras is False
    assert len(annotator.models) == 1


def test_snv_yields_one_ten_field_score(pytorch_annotator):
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    snv = _records(vcf)[0]                     # chr_test 6000 C -> A
    assert snv.ref == "C" and snv.alts == ("A",)

    scores = get_delta_scores(snv, annotator, dist_var=50, mask=0, flanking_size=80)
    assert len(scores) == 1
    fields = scores[0].split("|")
    assert len(fields) == 10
    assert fields[0] == "A"        # ALLELE
    assert fields[1] == "GENE1"    # SYMBOL


def test_deletion_yields_one_score(pytorch_annotator):
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    deletion = _records(vcf)[1]               # chr_test 6100 GC -> G
    assert len(deletion.ref) == 2 and len(deletion.alts[0]) == 1

    scores = get_delta_scores(deletion, annotator, dist_var=50, mask=0, flanking_size=80)
    assert len(scores) == 1
    assert len(scores[0].split("|")) == 10


def test_insertion_yields_one_score(pytorch_annotator):
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    insertion = _records(vcf)[2]              # chr_test 6200 G -> GT
    assert len(insertion.ref) == 1 and len(insertion.alts[0]) == 2

    scores = get_delta_scores(insertion, annotator, dist_var=50, mask=0, flanking_size=80)
    assert len(scores) == 1
    assert len(scores[0].split("|")) == 10


def test_multiallelic_yields_one_score_per_alt(pytorch_annotator):
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    multi = _records(vcf)[3]                  # chr_test 6300 C -> A,G
    assert len(multi.alts) == 2

    scores = get_delta_scores(multi, annotator, dist_var=50, mask=0, flanking_size=80)
    assert len(scores) == 2
    # one score per alt, allele field carried through in order
    assert scores[0].split("|")[0] == "A"
    assert scores[1].split("|")[0] == "G"
    for s in scores:
        assert len(s.split("|")) == 10


def test_mask_zeroes_donor_loss_for_insertion(pytorch_annotator):
    """mask=1 zeroes the unannotated donor-loss (DS_DL) field that is nonzero unmasked.

    Grounded: for the insertion record the unmasked DS_DL is 0.35 and becomes 0.00 with mask=1,
    while the delta-position fields are unchanged.
    """
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    insertion = _records(vcf)[2]

    unmasked = get_delta_scores(insertion, annotator, dist_var=50, mask=0, flanking_size=80)
    masked = get_delta_scores(insertion, annotator, dist_var=50, mask=1, flanking_size=80)
    assert len(unmasked) == 1 and len(masked) == 1

    uf = unmasked[0].split("|")
    mf = masked[0].split("|")
    # DS_DL is field index 5 (ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|...)
    assert float(uf[5]) > 0.0           # nonzero unmasked donor-loss
    assert float(mf[5]) == 0.0          # masked away
    # delta positions (indices 6..9) are unaffected by masking
    assert uf[6:10] == mf[6:10]


def test_mask_runs_and_keeps_ten_fields_all_records(pytorch_annotator):
    """mask=1 still produces well-formed 10-field scores for every record."""
    from openspliceai.variant.utils import get_delta_scores
    annotator, ref, ann, vcf = pytorch_annotator
    total = 0
    for rec in _records(vcf):
        scores = get_delta_scores(rec, annotator, dist_var=50, mask=1, flanking_size=80)
        for s in scores:
            assert len(s.split("|")) == 10
            total += 1
    # SNV(1) + del(1) + ins(1) + multiallelic(2) == 5 scores total
    assert total == 5


def test_ref_allele_longer_than_window_is_skipped(pytorch_annotator, tmp_path):
    """A record whose REF allele exceeds 2*dist_var is skipped -> [] (grounded)."""
    import pysam
    from pyfaidx import Fasta
    from openspliceai.variant.utils import get_delta_scores

    annotator, ref, ann, vcf = pytorch_annotator

    # Build a REF allele of length 120 (> 2*dist_var=100) that matches the reference at pos 6000.
    fa = Fasta(ref, sequence_always_upper=True)
    pos = 6000
    ref_allele = fa["chr_test"][pos - 1: pos - 1 + 120].seq
    assert len(ref_allele) == 120

    big_vcf = tmp_path / "big_ref.vcf"
    big_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr_test,length=12000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        f"chr_test\t{pos}\t.\t{ref_allele}\t{ref_allele[0]}\t.\t.\t.\n"
    )
    rec = next(iter(pysam.VariantFile(str(big_vcf))))
    assert len(rec.ref) > 2 * 50
    scores = get_delta_scores(rec, annotator, dist_var=50, mask=0, flanking_size=80)
    assert scores == []


def test_get_name_and_strand_finds_overlapping_gene(pytorch_annotator):
    """GENE1 (chr_test, +) overlaps pos 6000; a far-off position returns no genes."""
    annotator, *_ = pytorch_annotator
    genes, strands, idxs = annotator.get_name_and_strand("chr_test", 6000)
    assert list(genes) == ["GENE1"]
    assert list(strands) == ["+"]
    assert len(idxs) == 1

    # pos 1 is upstream of TX_START (1000 after +1) -> no overlap
    g2, s2, i2 = annotator.get_name_and_strand("chr_test", 1)
    assert len(g2) == 0 and len(s2) == 0 and len(i2) == 0


def test_get_pos_data_distances(pytorch_annotator):
    """get_pos_data returns (dist_tx_start, dist_tx_end, dist_exon_bdry).

    Grounded: tx_starts[idx]=1000 (table value 999 + 1), tx_ends[idx]=9000, and pos 6000 sits
    exactly on the exon boundary at 6000 (table EXON_START 5999 + 1).
    """
    annotator, *_ = pytorch_annotator
    _, _, idxs = annotator.get_name_and_strand("chr_test", 6000)
    idx = idxs[0]
    dist_tx_start, dist_tx_end, dist_exon_bdry = annotator.get_pos_data(idx, 6000)

    assert annotator.tx_starts[idx] == 1000
    assert annotator.tx_ends[idx] == 9000
    assert dist_tx_start == 1000 - 6000          # -5000
    assert dist_tx_end == 9000 - 6000            # 3000
    assert dist_exon_bdry == 0                   # pos 6000 is on the exon boundary


def test_resolve_builtin_annotation_exists():
    from openspliceai.variant.utils import _resolve_builtin_annotation
    for name in ("grch37", "grch38"):
        path = _resolve_builtin_annotation(name)
        assert os.path.exists(path), f"packaged annotation missing: {path}"


def test_annotator_accepts_custom_tsv(pytorch_annotator):
    """Constructing the Annotator from the custom TSV fixture parses the gene table fields."""
    annotator, ref, ann, vcf = pytorch_annotator
    # the custom TSV defined exactly one gene, GENE1 on chr_test
    assert "GENE1" in list(annotator.genes)
    assert "chr_test" in list(annotator.chroms)
    assert "+" in list(annotator.strands)
