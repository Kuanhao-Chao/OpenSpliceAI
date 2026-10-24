Human genome browser
====================

`Open the browser <https://khchao.com/OpenSpliceAI/genome/>`_ to explore
precomputed GRCh38 SNV delta scores with native IGV.js navigation and an
OpenSpliceAI exact-allele inspector. The updated application is being reviewed
locally; the public website continues to serve its previously published snapshot
until publication is approved.

The bundled demonstration contains six real scored regions from the final r10
and r13 collections. Its reference excerpts, gene features and scores are
self-contained. Regions outside the demonstration are explicitly unavailable.
The completed source collections and their genome-wide browser packaging are
separate deliverables; full online data require a configured indexed-data host.

Navigate and inspect
--------------------

Enter a scoring-gene symbol, a 1-based inclusive region such as
``chr1:925851-926130``, or an exact allele such as ``chr1:69091:A>G``.
Drag a native track to pan continuously. Drag the ruler to zoom to a region;
hold Shift while dragging the ruler to mark a region of interest. Native track
menus control presentation, scaling and height; drag handles reorder tracks.
The controls also support chromosome navigation, zoom, previous/next genes,
multiple genomic panes, and explicit selection of the inspector's active pane.

The exact table displays 100 alleles per page. Pagination does not truncate
statistics or CSV exports. Select an allele to inspect all original gene
annotations, source occurrences, five-decimal OpenSpliceAI values and signed
offsets. Native overview tracks show maxima across REF-matching alternate
alleles and genes; ALT, gene and threshold filters apply to the exact panels.

Save and restore a session to preserve the immutable manifest hash, model,
view settings, loci, track presentation and marked regions. Shared links pin
the manifest URL and content hash. BED regions can be imported and exported.
Additional local or URL tracks use IGV's supported formats and corresponding
indexes. Local files remain on the user's computer and must be reselected
when restoring a session on another page or device.

Interpret the measurements
--------------------------

* **DS_AG / DS_AL:** acceptor gain / loss variant delta scores.
* **DS_DG / DS_DL:** donor gain / loss variant delta scores.
* **DP:** original signed offset. The affected genomic coordinate is
  ``VCF POS + DP`` on both gene strands. A zero DS does not identify an
  affected site; its original DP remains available in the inspector and CSV.
* **Coverage:** REF-matching input and predicted ALT counts. Absent predictions,
  measured zero, REF mismatches, positions outside the input collection and
  regions outside the demonstration remain distinct.
* **r13 minus r10:** matched allele/gene comparisons. Conflicting annotations
  and REF mismatches are excluded from ordinary comparison statistics.

Both final source collections contain 3,433,384,833 source occurrences.
Of 3,339,752,304 distinct canonical input SNVs, 3,271,090,620 have predictions
per model (approximately 97.94%). The remaining 68,661,684 do not have a
prediction annotation. Completing processing does not turn absent predictions
into zeros. The source collection does not contain every possible human SNV.

r10 and r13 are individual MANE-trained model seeds, with 10,000 nt flanking
size, event search distance 50 and M1 masking. They are not a model ensemble.
The scoring genes come from the pinned GENCODE v24 canonical annotation.
The original masked SpliceAI baseline has two-decimal source precision and a
different annotation history. Model disagreement is not calibrated uncertainty.

Wide views use indexed annotation maxima. Exact inspection and CSV retain the
lossless source scores, offsets, source identities and duplicate occurrences.
The evidence panel distinguishes legacy data from receipt-bound provenance;
a model configuration label does not establish receipt-bound identity for
historical files. Reference, annotation, model and dataset hashes are available
in the methods panel.

Export and sequence search
--------------------------

PNG and vector SVG export native IGV figures, including whole-genome overviews.
Exact CSV includes all matching original annotations and explicitly named
0-based half-open and 1-based coordinate columns. Missing values are blank;
recorded zeros remain numeric zero. Summary CSV is a separate export of bins.
Exports require stable current-view data.

IUPAC sequence search supports both strands and cancellation. Visible-region
search uses published reference excerpts; chromosome and whole-reference
search require the corresponding static search indexes. The demonstration
includes the complete mitochondrial reference and its search index. Unpublished
indexes produce an explicit unavailable result rather than zero matches.

Run and review locally
----------------------

The browser performs no model inference. With Node.js 22.12 or later:

.. code-block:: bash

   cd docs/genome-browser
   npm ci
   npm test
   npm run build
   npm run dev

Open ``http://127.0.0.1:4173/OpenSpliceAI/genome/``. To serve a verified full
snapshot and the built application from one local origin, run from the repo root:

.. code-block:: bash

   python -m tools.genome_browser serve SNAPSHOT_DIR \
       --browser docs/genome-browser/dist/genome --port 8765

Then open ``http://127.0.0.1:8765/``. The data server serves the snapshot's
manifest-listed files; preparation databases and private configuration are
excluded. The legacy viewer preserves version-1 snapshot links.

Prepare and host full data
--------------------------

CPU-only preparation reads the accepted, content-hashed source VCFs. Packed
blocks, native BigWigs, indexed genes and reference/search artifacts support
bounded regional access without millions of tile files. Configurations and
builder code are frozen for scheduled runs. Resume authenticates completed
packs and their source identities; full publication requires a final verified
release, including source census reconciliation and reference content proof.

The large data stay outside Git and GitHub Pages. A future HTTPS data endpoint
must provide correct HTTP 206 byte ranges, expose Content-Range through CORS
when cross-origin, and preserve indexed bytes without HTTP content compression.
Export a verified file inventory and validate the actual endpoint before
activating its catalog. Publishing is a separate action from local review.
See ``tools/genome_browser/README.md`` for preparation and hosting commands.
