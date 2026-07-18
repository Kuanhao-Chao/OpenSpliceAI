"""Regression lock for the exact ref-length boundary of the MNV / indel reshape guard.

``get_delta_scores`` skips records with ``len(REF) > dist_var + 1``, because the ``y_alt``
reshape only stays cov-length (``cov = 2*dist_var + 1``, so ``cov // 2 == dist_var``) when
``ref_len <= dist_var + 1``. One position further, an equal-length MNV overruns ``cov`` (the
final ``concatenate`` shape-mismatches) and a deletion empties the ``np.max`` slice -- which is
exactly why the guard exists.

bpow's PR #18 tests the 60-mer gap-skip (well past the edge) but not the boundary itself. This
pins it precisely: with ``dist_var = 50``, a **51-mer** REF MNV (``ref_len == dist_var + 1``) is
scored into a full 10-field numeric record, and a **52-mer** REF MNV (``ref_len == dist_var + 2``,
the first length that overruns) returns ``[]`` -- skipped, never crashed.
"""
import os

import pytest

_NXT = {"A": "C", "C": "G", "G": "T", "T": "A", "N": "N"}


def _mut(s):
    """Every base changed to a guaranteed-different one (so the whole span is an MNV)."""
    return "".join(_NXT[b] for b in s)


@pytest.fixture(scope="module")
def boundary_annotator(tmp_path_factory, repo_root):
    """A real PyTorch Annotator (packaged 80nt checkpoint, CPU) over the variant_inputs fixture.

    Mirrors the shared fixture in tests/unit/test_variant_delta.py; kept local so this
    regression file is self-contained.
    """
    from openspliceai.variant.utils import Annotator
    from tests.fixtures.synthetic import write_variant_inputs

    state = repo_root / "models" / "openspliceai-honeybee" / "80nt" / "model_80nt_rs10.pt"
    if not state.exists():
        pytest.skip(f"packaged 80nt checkpoint not found: {state}")

    d = tmp_path_factory.mktemp("variant_boundary_ann")
    ref, ann, _vcf = write_variant_inputs(d)
    return Annotator(ref, ann, model_path=str(state), model_type="pytorch", CL=80)


def _write_single_record_vcf(path, chrom, pos, ref, alt):
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##contig=<ID={chrom},length=12000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")


@pytest.mark.parametrize("ref_len,expect_scored", [(51, True), (52, False)])
def test_mnv_reshape_guard_boundary(boundary_annotator, tmp_path, ref_len, expect_scored):
    import pysam

    from openspliceai.variant.utils import get_delta_scores

    dist_var = 50
    assert ref_len in (dist_var + 1, dist_var + 2)

    pos = 7000  # inside GENE1 (chr_test:1000-9000), far from the chromosome ends
    ref_allele = boundary_annotator.ref_fasta["chr_test"][pos - 1: pos - 1 + ref_len].seq.upper()
    alt_allele = _mut(ref_allele)  # equal-length MNV: every base changed
    assert len(ref_allele) == ref_len and len(alt_allele) == ref_len

    vpath = os.path.join(str(tmp_path), f"boundary_{ref_len}.vcf")
    _write_single_record_vcf(vpath, "chr_test", pos, ref_allele, alt_allele)
    rec = next(iter(pysam.VariantFile(vpath)))

    scores = get_delta_scores(rec, boundary_annotator, dist_var=dist_var, mask=0, flanking_size=80)

    if expect_scored:
        # ref_len == dist_var + 1: reshape stays cov-length -> a real, fully numeric score.
        assert len(scores) == 1
        fields = scores[0].split("|")
        assert len(fields) == 10
        assert fields[0] == alt_allele and fields[1] == "GENE1"
        for f in fields[2:6]:  # DS_AG/DS_AL/DS_DG/DS_DL parse as floats (not '.')
            assert f != "."
            float(f)
        for f in fields[6:10]:  # DP_AG/DP_AL/DP_DG/DP_DL parse as ints
            int(f)
    else:
        # ref_len == dist_var + 2: overruns the reshape -> skipped, not crashed.
        assert scores == []
