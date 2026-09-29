# OpenSpliceAI documentation

The full user manual is published at
**<https://khchao.com/OpenSpliceAI/>** 📒

The site is built with [Sphinx](https://www.sphinx-doc.org/) from the reStructuredText sources in
`source/`, and deployed to GitHub Pages by `.github/workflows/docs.yml` on every push to `main`.

## Building locally

Sphinx 8.2.3 requires Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make html                      # output in build/html
make html SPHINXOPTS="-W"      # what CI runs: warnings are errors
python check_links.py build/html
```

Open `build/html/index.html` in a browser to preview.

## Layout

| path | contents |
| --- | --- |
| `source/conf.py` | Sphinx configuration. The `extensions` list must stay in sync with `requirements.txt`. |
| `source/index.rst` | Landing page and the root toctree. |
| `source/content/` | All documentation pages. |
| `source/_static/` | CSS and the JHU/CCB logos. |
| `source/_images/` | Figures and the JHU footer logos. |
| `source/_templates/` | Sidebar overrides for the [furo](https://pradyunsg.me/furo/) theme. |
| `check_links.py` | Verifies every local link and asset in the built site resolves. |

## Notes for contributors

- `make html SPHINXOPTS="-W"` must pass before pushing — CI treats warnings as errors.
- Also run `check_links.py`: Sphinx does not validate URLs written by hand inside `.. raw:: html`
  blocks or in `_templates/*.html`, and this site uses both heavily.
- In templates, reference assets with `{{ pathto('_static/…', 1) }}` rather than hand-written `./`
  or `../` prefixes, so they resolve at every page depth.
- `.gitignore` excludes `*.png` repo-wide; `docs/source/_images/` and `docs/source/_static/` are
  explicitly re-included, so new figures there commit normally.
