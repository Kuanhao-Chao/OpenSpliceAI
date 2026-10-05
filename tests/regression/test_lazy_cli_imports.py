"""Regression lock: the ``openspliceai`` CLI must not import the heavy dependency
stack at module load, so that ``openspliceai`` (no args) and ``openspliceai --help``
work even when a transitive dependency has an import-time failure.

Background (GitHub issue #19): a transitive dependency read numpy's ``np.long``
(removed in numpy 1.24, re-added in 2.0); on numpy 1.26.4 that raises
``AttributeError`` at import. Because ``openspliceai/openspliceai.py`` used to
import every subcommand package -- and therefore torch / pandas /
scikit-learn / scipy / biopython / pysam -- at the top level, that single bad
dependency broke even a no-argument invocation and ``--help``. The subcommand
imports are now deferred into ``main()``'s dispatch. This test pins that: loading
the CLI, running ``--help``, and a bare no-args call must not pull in torch /
pandas / scipy / sklearn.

Each assertion runs in a subprocess so it sees a clean ``sys.modules`` (the rest
of the suite imports torch/pandas long before this test runs). The subprocess
imports the CLI from this repo (this file's package root), independent of any
separately installed copy.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("torch", "pandas", "scipy", "sklearn")


def _run(snippet):
    code = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\nHEAVY = {HEAVY!r}\n" + snippet
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_importing_cli_does_not_load_heavy_deps():
    r = _run(
        "import openspliceai.openspliceai\n"
        "bad = sorted(set(HEAVY) & set(sys.modules))\n"
        "assert not bad, 'heavy deps imported at CLI load: ' + repr(bad)\n"
    )
    assert r.returncode == 0, r.stderr


def test_help_does_not_load_heavy_deps():
    r = _run(
        "import openspliceai.openspliceai as m\n"
        "try:\n"
        "    m.main(['--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0, e.code\n"
        "bad = sorted(set(HEAVY) & set(sys.modules))\n"
        "assert not bad, 'heavy deps imported by --help: ' + repr(bad)\n"
    )
    assert r.returncode == 0, r.stderr


def test_no_args_errors_without_heavy_deps():
    r = _run(
        "import openspliceai.openspliceai as m\n"
        "try:\n"
        "    m.main([])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 2, e.code\n"  # argparse: missing required subcommand
        "bad = sorted(set(HEAVY) & set(sys.modules))\n"
        "assert not bad, 'heavy deps imported on no-args: ' + repr(bad)\n"
    )
    assert r.returncode == 0, r.stderr
