"""Cross-run synthesis for the full-SNV concordance study.

This package is deliberately a *sibling* of ``validation.full_snv_concordance``
rather than a module inside it: every queued map, reduce and render job verifies
a recursive SHA-256 fingerprint of that package directory, so adding a file there
while jobs are queued would fail them closed. Nothing here is executed by those
jobs; this package only reads the reduced summaries they produce.
"""

from .loading import Run, load_run, load_study  # noqa: F401

__all__ = ["Run", "load_run", "load_study"]
