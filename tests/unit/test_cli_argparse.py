"""Argparse-level tests for the CLI fixes (flanking choices, transfer unfreeze flag)."""
import pytest

from openspliceai.openspliceai import parse_args

_TRANSFER = ["transfer", "-o", "o", "-p", "p", "-m", "m", "-train", "t", "-test", "t"]
_CALIBRATE = ["calibrate", "-o", "o", "-p", "p", "-m", "m", "-train", "t", "-test", "t"]


def test_transfer_unfreeze_all_defaults_false():
    args = parse_args(_TRANSFER)
    assert args.unfreeze_all is False


def test_transfer_unfreeze_all_flag_sets_true():
    args = parse_args(_TRANSFER + ["--unfreeze-all"])
    assert args.unfreeze_all is True


@pytest.mark.parametrize("size", [80, 400, 2000, 10000])
def test_predict_flanking_size_accepts_supported(size):
    args = parse_args(["predict", "-i", "x", "-m", "m", "-f", str(size)])
    assert args.flanking_size == size


def test_predict_flanking_size_rejects_unsupported():
    with pytest.raises(SystemExit):
        parse_args(["predict", "-i", "x", "-m", "m", "-f", "123"])


def test_predict_gene_flank_defaults_to_auto():
    """--gene-flank defaults to -1 (auto = flanking_size//2 of real genomic context)."""
    args = parse_args(["predict", "-i", "x", "-m", "m", "-f", "10000"])
    assert args.gene_flank == -1


def test_predict_gene_flank_override():
    args = parse_args(["predict", "-i", "x", "-m", "m", "-f", "10000", "--gene-flank", "0"])
    assert args.gene_flank == 0


def test_variant_flanking_size_rejects_unsupported():
    with pytest.raises(SystemExit):
        parse_args(["variant", "-R", "r", "-A", "grch38", "-f", "999"])


def test_variant_flanking_size_accepts_supported():
    args = parse_args(["variant", "-R", "r", "-A", "grch38", "-f", "400"])
    assert args.flanking_size == 400


def test_calibrate_flanking_size_choices_enforced():
    assert parse_args(_CALIBRATE + ["-f", "80"]).flanking_size == 80
    with pytest.raises(SystemExit):
        parse_args(_CALIBRATE + ["-f", "123"])
