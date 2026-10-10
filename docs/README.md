# OpenSpliceAI documentation

The full user manual is published at
**<https://khchao.com/OpenSpliceAI/>** 📒

The site is built with [Sphinx](https://www.sphinx-doc.org/) from the reStructuredText sources in
`source/`, and deployed to GitHub Pages by `.github/workflows/docs.yml` on every push to `main`.

## Building locally

Use Python ≥ 3.11 and Node.js ≥ 22.12. The HTML build includes the genome
browser, so install both sets of dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm ci --prefix genome-browser

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
| `genome-browser/` | Human SNV browser, real-data review subset and browser tests. |

## Notes for contributors

- `make html SPHINXOPTS="-W"` must pass before pushing — CI treats warnings as errors.
- Also run `check_links.py`: Sphinx does not validate URLs written by hand inside `.. raw:: html`
  blocks or in `_templates/*.html`, and this site uses both heavily.
- In templates, reference assets with `{{ pathto('_static/…', 1) }}` rather than hand-written `./`
  or `../` prefixes, so they resolve at every page depth.
- `.gitignore` excludes `*.png` repo-wide; `docs/source/_images/` and `docs/source/_static/` are
  explicitly re-included, so new figures there commit normally.
## Human genome browser

The HTML build includes the standalone browser at `genome/`. From this
directory, run the frontend unit tests or start the development server:

```bash
cd genome-browser
npm test
npm run dev
```

Open
`http://127.0.0.1:4173/OpenSpliceAI/genome/`. `npm run audit` runs the browser
interaction and export checks. Python preparation tests run with
`python -m unittest discover -s tools/genome_browser/tests -v`.

The bundled dataset is a clearly labelled real-data review subset. Genome-wide
data use an institutional HTTPS range/CORS host configured through
`docs/genome-browser/public/settings.json`; see
[`tools/genome_browser/README.md`](../tools/genome_browser/README.md).
