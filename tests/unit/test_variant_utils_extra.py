"""Additional coverage for openspliceai/variant/utils.py.

Complements test_variant_delta.py (which uses a + strand gene) with: the minus-strand
reverse-complement path in get_delta_scores / get_delta_scores_batched, batched==sequential
on the minus strand, the multi-nucleotide-variant and bad-record short circuits,
_ensemble_forward averaging, and the model-loader error/exit paths (pytorch + keras)."""
import os
import types

import pytest
import torch

import openspliceai.variant.utils as vu

MINUS_TSV = ("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n"
             "GENEM\tchr_test\t-\t999\t9000\t999,5999,\t5000,9000,\n")


@pytest.fixture(scope="module")
def minus_annotator(tmp_path_factory, repo_root):
    """A PyTorch Annotator whose single gene is on the MINUS strand (same coords as the
    + gene in test_variant_delta), built over the variant_inputs reference."""
    from tests.fixtures.synthetic import write_variant_inputs
    state = repo_root / "models" / "openspliceai-honeybee" / "80nt" / "model_80nt_rs10.pt"
    if not state.exists():
        pytest.skip(f"packaged 80nt checkpoint not found: {state}")
    d = tmp_path_factory.mktemp("variant_minus")
    ref, _ann, vcf = write_variant_inputs(d)
    minus_ann = d / "annotation_minus.tsv"
    minus_ann.write_text(MINUS_TSV)
    annotator = vu.Annotator(ref, str(minus_ann), model_path=str(state), model_type="pytorch", CL=80)
    return annotator, vcf


def _records(vcf_path):
    import pysam
    return list(pysam.VariantFile(vcf_path))


# --- minus-strand path ---------------------------------------------------------------

def test_minus_strand_get_delta_scores_yields_score(minus_annotator):
    annotator, vcf = minus_annotator
    snv = _records(vcf)[0]
    scores = vu.get_delta_scores(snv, annotator, dist_var=50, mask=0, flanking_size=80)
    assert len(scores) == 1
    fields = scores[0].split("|")
    assert len(fields) == 10 and fields[1] == "GENEM"


@pytest.mark.parametrize("mask", [0, 1])
def test_batched_equals_sequential_on_minus_strand(minus_annotator, mask):
    """Locks batched==sequential for a minus-strand gene (extends the equivalence lock)."""
    os.environ["OSAI_TF32"] = "0"   # full fp32 -> bit-reproducible on any device
    annotator, vcf = minus_annotator
    for rec in _records(vcf):
        seq = vu.get_delta_scores(rec, annotator, 50, mask, flanking_size=80)
        bat = vu.get_delta_scores_batched([rec], annotator, 50, mask, flanking_size=80, batch_size=8)
        assert bat[0] == seq


# --- MNV + bad record short circuits -------------------------------------------------

def test_multinucleotide_variant_emits_dotted_score(minus_annotator):
    """A REF>1 & ALT>1 record short-circuits to a '.'-filled score (no model run)."""
    import pysam
    annotator, _vcf = minus_annotator
    pos = 6100
    ref2 = annotator.ref_fasta["chr_test"][pos - 1: pos + 1].seq   # 2 real reference bases
    alt2 = ("AT" if ref2 != "AT" else "GC")
    d = os.path.dirname(_vcf)
    mnv = os.path.join(d, "mnv.vcf")
    with open(mnv, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr_test,length=12000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write(f"chr_test\t{pos}\t.\t{ref2}\t{alt2}\t.\t.\t.\n")
    rec = next(iter(pysam.VariantFile(mnv)))
    scores = vu.get_delta_scores(rec, annotator, 50, 0, flanking_size=80)
    assert len(scores) == 1
    assert scores[0] == f"{alt2}|GENEM|.|.|.|.|.|.|.|."


def test_bad_record_returns_empty(minus_annotator):
    annotator, _vcf = minus_annotator
    bad = types.SimpleNamespace(chrom="chr_test", pos=6000, ref="C", alts=None)  # len(None) -> TypeError
    assert vu.get_delta_scores(bad, annotator, 50, 0, flanking_size=80) == []
    assert vu.get_delta_scores_batched([bad], annotator, 50, 0, flanking_size=80) == [[]]


# --- _ensemble_forward ---------------------------------------------------------------

def test_ensemble_forward_is_mean(model_80nt):
    import copy
    m2 = copy.deepcopy(model_80nt)
    for p in m2.parameters():
        p.data.add_(0.02)
    xb = torch.zeros(2, 4, 5080)
    xb[:, 0, :] = 1.0
    out = vu._ensemble_forward([model_80nt, m2], xb)
    with torch.no_grad():
        manual = torch.mean(torch.stack([model_80nt(xb), m2(xb)]), axis=0).permute(0, 2, 1)
    assert out.shape == manual.shape
    assert torch.allclose(out, manual, atol=1e-6)


# --- model loader error/exit paths ---------------------------------------------------

def test_resolve_default_spliceai_models_returns_five_paths():
    paths = vu._resolve_default_spliceai_models()
    assert len(paths) == 5 and all(p.endswith(".h5") for p in paths)


@pytest.mark.parametrize("cl", [400, 2000, 10000])
def test_load_pytorch_models_table_branch_then_exit(packaged_80nt_state, cl):
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(packaged_80nt_state, cl)


def test_load_pytorch_models_empty_dir_exits(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(str(d), 80)


def test_load_pytorch_models_corrupt_dir_and_file_exit(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "bad.pt").write_text("nope")
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(str(d), 80)
    bad = tmp_path / "x.pt"
    bad.write_text("nope")
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(str(bad), 80)


def test_load_pytorch_models_invalid_path_exits(tmp_path):
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(str(tmp_path / "missing.pt"), 80)


def test_load_pytorch_models_size_mismatch_branch(tmp_path, packaged_80nt_state):
    state = torch.load(packaged_80nt_state, map_location="cpu")
    key = next(iter(state))
    state[key] = torch.zeros(state[key].shape[0] + 1, *state[key].shape[1:])   # wrong shape
    bad = str(tmp_path / "bad.pt")
    torch.save(state, bad)
    with pytest.raises(SystemExit):
        vu.load_pytorch_models(bad, 80)


def test_annotator_unsupported_model_type_exits(tmp_path):
    from tests.fixtures.synthetic import write_variant_inputs
    ref, ann, _vcf = write_variant_inputs(tmp_path)
    with pytest.raises(SystemExit):
        vu.Annotator(ref, ann, model_path="x", model_type="bogus", CL=80)


def test_annotator_malformed_annotation_exits(tmp_path):
    from tests.fixtures.synthetic import write_variant_inputs
    ref, _ann, _vcf = write_variant_inputs(tmp_path)
    bad = tmp_path / "bad.tsv"
    bad.write_text("not\ta\tvalid\theader\n1\t2\t3\t4\n")   # no #NAME column -> KeyError
    with pytest.raises(SystemExit):
        vu.Annotator(ref, str(bad), model_path="x", model_type="pytorch", CL=80)


@pytest.mark.keras
def test_load_keras_models_error_paths(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        vu.load_keras_models(str(empty))             # no .h5 in dir
    with pytest.raises(SystemExit):
        vu.load_keras_models(str(tmp_path / "nope")) # invalid path
