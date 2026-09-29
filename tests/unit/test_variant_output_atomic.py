"""Unit coverage for safe publication and non-annotation contig pass-through."""

import os
import signal
import types

import pysam
import pytest


class _FakeAnnotator:
    """Small scorer stand-in; only fields used by ``variant.variant`` are needed."""

    chroms = ("chr1",)
    ref_fasta = {
        "chr1": "A" * 20,
        "chrAlt": "C" * 20,
    }


def _write_vcf(path):
    # Deliberately omit chrAlt from the header. This is the production failure mode:
    # htslib accepts the input text but cannot write its alt-contig record unless the
    # destination header is repaired first.
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=20>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\trs1\tA\tC\t.\tPASS\t.\n"
        "chrAlt\t2\trsAlt\tC\tT\t.\tPASS\t.\n"
    )


def _args(input_vcf, output_vcf, batch_size=1):
    return types.SimpleNamespace(
        input_vcf=str(input_vcf),
        output_vcf=str(output_vcf),
        ref_genome="unused.fa",
        annotation="unused.tsv",
        model="unused.pt",
        flanking_size=80,
        distance=50,
        mask=1,
        model_type="pytorch",
        precision=5,
        batch_size=batch_size,
    )


def _temporary_files(output_path):
    return list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_missing_reference_contig_is_declared_and_passed_through(
    tmp_path, monkeypatch, batch_size
):
    from openspliceai.variant import variant as variant_mod

    source = tmp_path / "input.vcf"
    output = tmp_path / "result.vcf"
    _write_vcf(source)

    monkeypatch.setattr(variant_mod, "Annotator", lambda *_args: _FakeAnnotator())
    scored_chroms = []

    def score_one(record, *_args):
        scored_chroms.append(record.chrom)
        return ["C|GENE1|0.10000|0|0|0|1|0|0|0"]

    def score_batch(records, *_args):
        scored_chroms.extend(record.chrom for record in records)
        return [["C|GENE1|0.10000|0|0|0|1|0|0|0"] for _ in records]

    monkeypatch.setattr(variant_mod, "get_delta_scores", score_one)
    monkeypatch.setattr(variant_mod, "get_delta_scores_batched", score_batch)

    variant_mod.variant(_args(source, output, batch_size=batch_size))

    assert scored_chroms == ["chr1"]
    assert _temporary_files(output) == []
    with pysam.VariantFile(output) as result:
        assert result.header.contigs["chrAlt"].length == 20
        records = list(result)
    assert len(records) == 2
    assert records[0].info["OpenSpliceAI"] == (
        "C|GENE1|0.10000|0|0|0|1|0|0|0",
    )
    assert records[1].chrom == "chrAlt"
    assert records[1].pos == 2
    assert records[1].id == "rsAlt"
    assert records[1].ref == "C"
    assert records[1].alts == ("T",)
    assert "OpenSpliceAI" not in records[1].info


def test_success_uses_same_directory_atomic_replace(tmp_path, monkeypatch):
    from openspliceai.variant import variant as variant_mod

    source = tmp_path / "input.vcf"
    output = tmp_path / "nested" / "result.vcf"
    _write_vcf(source)
    monkeypatch.setattr(variant_mod, "Annotator", lambda *_args: _FakeAnnotator())
    monkeypatch.setattr(variant_mod, "get_delta_scores", lambda *_args: [])

    real_replace = os.replace
    replace_calls = []

    def recording_replace(source_path, destination_path):
        replace_calls.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(variant_mod.os, "replace", recording_replace)

    previous_umask = os.umask(0o022)
    try:
        variant_mod.variant(_args(source, output))
    finally:
        os.umask(previous_umask)

    assert len(replace_calls) == 1
    temporary, destination = replace_calls[0]
    assert os.path.dirname(temporary) == str(output.parent)
    assert destination == str(output)
    assert output.exists()
    assert output.stat().st_mode & 0o777 == 0o644
    assert _temporary_files(output) == []


def test_scoring_failure_preserves_existing_final_and_cleans_temp(tmp_path, monkeypatch):
    from openspliceai.variant import variant as variant_mod

    source = tmp_path / "input.vcf"
    output = tmp_path / "result.vcf"
    _write_vcf(source)
    original = b"existing completed output\n"
    output.write_bytes(original)
    monkeypatch.setattr(variant_mod, "Annotator", lambda *_args: _FakeAnnotator())

    def fail_scoring(*_args):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(variant_mod, "get_delta_scores", fail_scoring)

    with pytest.raises(RuntimeError, match="simulated model failure"):
        variant_mod.variant(_args(source, output))

    assert output.read_bytes() == original
    assert _temporary_files(output) == []


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_termination_signal_preserves_existing_final_and_cleans_temp(tmp_path, monkeypatch):
    from openspliceai.variant import variant as variant_mod

    source = tmp_path / "input.vcf"
    output = tmp_path / "result.vcf"
    _write_vcf(source)
    original = b"existing completed output\n"
    output.write_bytes(original)
    monkeypatch.setattr(variant_mod, "Annotator", lambda *_args: _FakeAnnotator())

    def terminate_during_scoring(*_args):
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(variant_mod, "get_delta_scores", terminate_during_scoring)

    with pytest.raises(InterruptedError, match="interrupted by signal"):
        variant_mod.variant(_args(source, output))

    assert output.read_bytes() == original
    assert _temporary_files(output) == []


def test_validation_failure_preserves_existing_final_and_cleans_temp(tmp_path, monkeypatch):
    from openspliceai.variant import variant as variant_mod

    source = tmp_path / "input.vcf"
    output = tmp_path / "result.vcf"
    _write_vcf(source)
    original = b"existing completed output\n"
    output.write_bytes(original)
    monkeypatch.setattr(variant_mod, "Annotator", lambda *_args: _FakeAnnotator())
    monkeypatch.setattr(variant_mod, "get_delta_scores", lambda *_args: [])

    def fail_validation(*_args):
        raise RuntimeError("simulated validation failure")

    monkeypatch.setattr(variant_mod, "_validate_output", fail_validation)

    with pytest.raises(RuntimeError, match="simulated validation failure"):
        variant_mod.variant(_args(source, output))

    assert output.read_bytes() == original
    assert _temporary_files(output) == []


def test_structural_validation_rejects_wrong_count_and_incomplete_tail(tmp_path):
    from openspliceai.variant.variant import _validate_output

    output = tmp_path / "candidate.vcf"
    output.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=20>\n"
        "##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description=\"scores\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tC\t.\tPASS\tOpenSpliceAI=C|G|0|0|0|0|0|0|0|0\n"
    )

    _validate_output(output, expected_records=1)
    with pytest.raises(RuntimeError, match="record count"):
        _validate_output(output, expected_records=2)

    output.write_bytes(output.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="newline-terminated"):
        _validate_output(output, expected_records=1)
