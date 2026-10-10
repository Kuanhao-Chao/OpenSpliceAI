Human genome browser
====================

`Open the browser <https://khchao.com/OpenSpliceAI/genome/>`_ to explore
precomputed GRCh38 SNV delta scores alongside the scoring annotation's genes,
exons and strand. The browser uses the interaction style of the
`Shorkie yeast browser <https://khchao.com/shorkie-lab/genome/>`_.

The bundled review dataset contains **real scored loci**, including a
minus-strand gene and alternate-contig reference mismatches. It is explicitly
labelled as a subset. Genome-wide publication requires the prepared data on
the configured institutional HTTPS host. Complete r10 is the default model;
r13 remains a frozen preview with coverage gaps until its final audit passes.

Navigate and inspect
--------------------

Enter a gene symbol, a 1-based inclusive region such as
``chr1:925851-926130``, or an exact allele such as ``chr1:69091:A>G``.
Use the chromosome overview, minimap, gene navigation, zoom buttons, or drag
to move through the genome. Shift-drag marks a region of interest. The share
button captures the dataset snapshot and every view setting.
Shared links also pin the immutable manifest URL, so changing the default
catalog does not redirect an existing shared view to a different snapshot.

Track settings control order, height, density, score threshold, ALT and gene
filters. Six themes and mobile track controls follow the existing yeast
browser. The keyboard-accessible variant selector provides the same exact
values as clicking the tracks. Sequence search accepts DNA IUPAC symbols and
both strands, with visible-region, gene, chromosome and whole-reference
scopes. Broad searches require the corresponding static sequence indexes;
an unavailable index is reported explicitly.

Interpret the measurements
--------------------------

* **DS_AG / DS_AL:** acceptor gain / loss variant delta scores.
* **DS_DG / DS_DL:** donor gain / loss variant delta scores.
* **DP:** original signed offset. The affected genomic coordinate is
  ``VCF POS + DP`` on both gene strands. A zero DS does not establish an
  affected site, even though its original DP is preserved in the inspector.
* **ALT heatmap:** maximum delta across the displayed gene annotations for
  each alternate base. Inspect an allele for every original annotation.
* **r13 coverage:** accepted source occurrences, including absent predictions.
  Pending work, absent predictions, measured zero and positions outside the
  source collection are distinct states.
* **r13 − r10:** difference in maximum delta for a matched allele/gene.
  Conflicting annotations, partial preview coverage and REF mismatches are
  excluded from ordinary comparisons.

r10 and r13 are **individual model seeds**, with 10,000 nt flanking size,
event search distance 50 and M1 masking. They are not a five-model ensemble.
The original masked SpliceAI annotation has two-decimal precision and its
gene set differs from the MANE scoring annotation. The browser preserves
these differences; seed disagreement is not calibrated uncertainty.

At close zoom, exact OpenSpliceAI values retain all five decimal places.
Wide views show fixed-bin annotation maxima, never fabricated exact alleles.
Summary counts refer to source occurrences or gene annotations and are not
labelled unique variants. The original collection does not contain every
possible SNV at every reference base. The methods panel reports reference,
annotation, model and snapshot hashes.

Export results
--------------

PNG and vector SVG figures use the same renderer as the tracks. Exact CSV
contains every original annotation, signed offsets, affected positions,
reference-match flags and source identities. Missing values remain blank;
measured zeros remain numeric zero. Summary CSV is a separate export of bin
statistics. CSV includes explicitly named 0-based half-open and 1-based
coordinate columns. Exports require a successfully loaded, stable view.

Run locally
-----------

The browser has no model inference dependency. Install Node.js 22.12 or later:

.. code-block:: bash

   cd docs/genome-browser
   npm ci
   npm test
   npm run dev

Open ``http://127.0.0.1:4173/OpenSpliceAI/genome/``. To use an independently
prepared snapshot, start its range server from the repository root:

.. code-block:: bash

   python -m tools.genome_browser serve results/genome_browser/full

Then open the browser with
``?manifest=http://127.0.0.1:8765/manifest.json``. The production documentation
build runs ``npm run build`` and copies the application under ``genome/``.

Prepare and publish genome-wide data
------------------------------------

Preparation reads accepted, content-hashed VCF audit manifests without changing
the scoring files or jobs. It runs on CPUs and creates packed, independently
compressed blocks, avoiding millions of tile files. Shards resume only after
their configuration and stored content hashes agree.

.. code-block:: bash

   python -m tools.genome_browser prepare CONFIG.json SNAPSHOT_DIR
   # Optional native suffix-array dependency, in an isolated environment:
   pip install -r tools/genome_browser/requirements.txt
   OMP_NUM_THREADS=2 python -m tools.genome_browser reference CONFIG.json SNAPSHOT_DIR --search
   python -m tools.genome_browser finalize CONFIG.json SNAPSHOT_DIR
   python -m tools.genome_browser attach-reference SNAPSHOT_DIR
   python -m tools.genome_browser verify SNAPSHOT_DIR

The institutional data host must provide HTTPS, HTTP 206 byte ranges,
cross-origin access for ``https://khchao.com`` and exposed ``Content-Range``.
HTTP content compression is disabled for indexed files; blocks are already
compressed internally. Upload immutable data files first, verify them, and
publish the manifest/catalog last. Set ``docs/genome-browser/public/settings.json``
to the public catalog URL. Validate the deployed endpoint with:

.. code-block:: bash

   python -m tools.genome_browser probe https://HOST/SNAPSHOT/manifest.json

The repository's ``tools/genome_browser/README.md`` documents configuration,
resumption, expected resources and publication. ``FORMAT.md`` describes the
lossless binary format; ``hosting/`` provides Apache and Nginx examples.
The small application belongs on GitHub Pages; the large data stay on the
institutional host. No paid hosting or GPU inference is needed for the browser.
The bundled review subset authenticates small same-origin files before slicing
them, which accommodates GitHub Pages recompression. This exception is limited
to files at most 1 MiB; genome-wide data always require unchanged range bytes.
