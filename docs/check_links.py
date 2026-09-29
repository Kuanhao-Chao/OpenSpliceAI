#!/usr/bin/env python3
"""Check that every local asset and page referenced by the built docs actually exists.

Sphinx validates ``.. image::`` paths, but not URLs written by hand inside ``.. raw:: html``
blocks or in ``_templates/*.html``. The OpenSpliceAI docs use both heavily (the JHU/CCB logo
swappers), so a warning-free ``sphinx-build`` is not on its own evidence that the site renders.
This walks the built HTML and resolves every local href/src/srcset against the output tree.

Usage:  python docs/check_links.py [build/html]
Exits non-zero if anything is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

# href/src/srcset values, single or double quoted. `content` is deliberately excluded --
# it is <meta> payload ("width=device-width, initial-scale=1"), not a URL.
ATTR_RE = re.compile(r"""\b(?:href|src|srcset)\s*=\s*(["'])(.*?)\1""", re.I | re.S)
# Bare quoted paths inside inline <script> logo-swapping code, e.g. '../_static/JHU_ccb-dark.png'.
SCRIPT_ASSET_RE = re.compile(r"""(["'])((?:\.{1,2}/)*_(?:static|images)/[^"']+?)\1""")

SKIP_SCHEMES = {"http", "https", "mailto", "javascript", "data", "ftp", "tel"}


def candidates(html: str) -> set[str]:
    found = {m.group(2).strip() for m in ATTR_RE.finditer(html)}
    found |= {m.group(2).strip() for m in SCRIPT_ASSET_RE.finditer(html)}
    return found


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "build/html").resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory -- build the docs first", file=sys.stderr)
        return 2

    pages = sorted(root.rglob("*.html"))
    missing: list[tuple[Path, str, Path]] = []
    checked = 0

    for page in pages:
        html = page.read_text(encoding="utf-8", errors="replace")
        for raw in candidates(html):
            for ref in (r.strip() for r in raw.split(",")):  # srcset is comma-separated
                ref = ref.split()[0] if ref and " " in ref else ref
                if not ref or ref.startswith("#"):
                    continue
                parsed = urlparse(ref)
                if parsed.scheme.lower() in SKIP_SCHEMES or ref.startswith("//"):
                    continue
                path = unquote(parsed.path)
                if not path:
                    continue
                base = root if path.startswith("/") else page.parent
                target = (base / path.lstrip("/")).resolve()
                checked += 1
                # Stay inside the output tree; a reference escaping it is broken by definition.
                if not target.is_file() and not (target.is_dir() and (target / "index.html").is_file()):
                    missing.append((page.relative_to(root), ref, target))

    print(f"scanned {len(pages)} pages, resolved {checked} local references")
    if missing:
        print(f"\n{len(missing)} broken local reference(s):\n")
        for page, ref, target in sorted(set(missing)):
            print(f"  {page}\n      -> {ref}\n         (no such file: {target})")
        return 1
    print("all local references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
