"""End-to-end smoke test for the PyTorch variant delta-score path.

Builds a tiny custom reference + annotation + VCF (so no GPU / large genome needed) and runs
get_delta_scores through the real Annotator + PyTorch model ensemble. Exercises one_hot_encode,
the coordinate cropping, and the ensemble-mean inference.
"""
import pytest


def _write_variant_fixtures(tmp_path):
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr_test\n" + ("A" * 400) + "\n")     # ref base at pos 200 is 'A'

    ann = tmp_path / "ann.txt"
    ann.write_text(
        "#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n"
        "GENE1\tchr_test\t+\t49\t350\t49,199,\t150,350,\n"
    )

    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr_test,length=400>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr_test\t200\t.\tA\tC\t.\t.\t.\n"
    )
    return str(ref), str(ann), str(vcf)


@pytest.mark.integration
@pytest.mark.slow
def test_variant_pytorch_delta_scores(tmp_path, packaged_80nt_dir):
    import pysam
    from openspliceai.variant.utils import Annotator, get_delta_scores

    ref, ann, vcf_path = _write_variant_fixtures(tmp_path)
    annotator = Annotator(ref, ann, model_path=packaged_80nt_dir, model_type="pytorch", CL=80)

    record = next(iter(pysam.VariantFile(vcf_path)))
    scores = get_delta_scores(record, annotator, dist_var=50, mask=0, flanking_size=80)

    assert len(scores) == 1
    fields = scores[0].split("|")
    assert fields[0] == "C" and fields[1] == "GENE1"   # ALLELE | SYMBOL | ...
    assert len(fields) == 10                            # 4 delta scores + 4 delta positions
