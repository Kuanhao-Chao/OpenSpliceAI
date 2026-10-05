"""Portable SHA-256 fingerprints for immutable research inputs and source trees."""

import hashlib
import os
from pathlib import Path
import stat
import sys


def file_sha256(path):
    """Hash a file in bounded chunks, including empty files."""
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    """Hash a file, or the sorted GNU sha256sum records of a nonempty tree.

    Tree records retain full input paths, C-locale byte ordering and GNU filename
    escaping. Symlinks and bytecode caches are excluded, matching the historical
    Linux launcher. The same path and contents therefore produce the same digest
    on submission and compute hosts, independent of their locale or GNU tools.
    """
    path = os.fsencode(path)
    mode = os.stat(path, follow_symlinks=False).st_mode
    if stat.S_ISREG(mode):
        return file_sha256(path)
    if not stat.S_ISDIR(mode):
        raise ValueError("Fingerprint input must be a regular file or directory")
    files = []
    for directory, directories, names in os.walk(path):
        directories[:] = [name for name in directories if name != b"__pycache__"]
        for name in names:
            candidate = os.path.join(directory, name)
            if stat.S_ISREG(os.stat(candidate, follow_symlinks=False).st_mode):
                files.append(candidate)
    if not files:
        raise ValueError("Cannot fingerprint an empty source tree")
    result = hashlib.sha256()
    for name in sorted(files):
        escaped = b"\\" in name or b"\n" in name
        display_name = name.replace(b"\\", b"\\\\").replace(b"\n", b"\\n")
        result.update(
            (b"\\" if escaped else b"")
            + file_sha256(name).encode("ascii") + b"  " + display_name + b"\n"
        )
    return result.hexdigest()


if __name__ == "__main__":
    print(fingerprint(Path(sys.argv[1])))
