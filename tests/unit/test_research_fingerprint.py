"""Portable source identity checks for the research map/reduce launcher."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from validation.full_snv_concordance.slurm.fingerprint import fingerprint


EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("contents, expected", [(b"", EMPTY_SHA), (b"abc", ABC_SHA)])
def test_input_file_known_sha256(tmp_path, contents, expected):
    path = tmp_path / "pairs with spaces.tsv"
    path.write_bytes(contents)
    assert fingerprint(path) == expected


def test_tree_preserves_gnu_records_and_excludes_caches_and_symlinks(tmp_path):
    (tmp_path / "A.txt").write_bytes(b"abc")
    (tmp_path / "b with spaces.txt").write_bytes(b"")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored bytecode")
    (tmp_path / "link.txt").symlink_to(tmp_path / "A.txt")
    # This independent manifest fixes record formatting and bytewise ordering.
    records = (
        f"{ABC_SHA}  {tmp_path}/A.txt\n"
        f"{EMPTY_SHA}  {tmp_path}/b with spaces.txt\n"
    ).encode()
    expected = hashlib.sha256(records).hexdigest()
    assert fingerprint(tmp_path) == expected
    (cache / "ignored.pyc").write_bytes(b"changed cache")
    assert fingerprint(tmp_path) == expected
    (tmp_path / "A.txt").write_bytes(b"changed source")
    assert fingerprint(tmp_path) != expected


def test_tree_escapes_newlines_and_backslashes_like_gnu(tmp_path):
    (tmp_path / "a\nb.txt").write_bytes(b"abc")
    (tmp_path / "c\\d.txt").write_bytes(b"")
    records = (
        f"\\{ABC_SHA}  {tmp_path}/a\\nb.txt\n"
        f"\\{EMPTY_SHA}  {tmp_path}/c\\\\d.txt\n"
    ).encode()
    assert fingerprint(tmp_path) == hashlib.sha256(records).hexdigest()


def test_empty_tree_and_root_symlink_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="empty source tree"):
        fingerprint(tmp_path)
    source = tmp_path / "source.py"
    source.write_bytes(b"abc")
    link = tmp_path / "link.py"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular file or directory"):
        fingerprint(link)
    with pytest.raises(FileNotFoundError):
        fingerprint(tmp_path / "missing")


def test_shell_wrapper_uses_selected_interpreter(tmp_path):
    source = tmp_path / "pairs.tsv"
    source.write_bytes(b"abc")
    wrapper = Path("validation/full_snv_concordance/slurm/fingerprint.sh")
    completed = subprocess.run(
        [str(wrapper), str(source), sys.executable],
        text=True, capture_output=True, check=True,
    )
    assert completed.stdout.strip() == ABC_SHA

