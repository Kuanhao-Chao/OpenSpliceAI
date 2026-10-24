# IGV browser implementation and local review

This work remains uncommitted and unpublished. The source is isolated from the
scoring checkout. The public website still serves the earlier review snapshot.

The current application uses pinned IGV.js 3.8.9 for native genomic navigation,
track rendering, multiple panes and user tracks. Exact source records remain in
lossless OSGB1 packs. Filtering, ranking, statistics and complete exact CSV
exports run in a worker; tables return 100-row pages and compact plot buffers.

Version-2 sessions pin the manifest URL and SHA-256 and preserve loci, active
pane, track presentation and regions. Verified demonstration Files serialize as
logical snapshot descriptors. Local user tracks serialize as reselection
placeholders. A rebuilt legacy viewer preserves version-1 snapshots.

The default demonstration contains 44,593 source occurrences across six regions,
complete mitochondrial sequence/search, indexed score/coverage tracks and
published-region gene features. Public export is an authenticated allowlist;
private configuration, SQLite state and internal receipts remain outside it.
See public/catalog.json for its exact immutable identity.

Scientific processing completion, prediction availability and packaging status
are separate. Full r10/r13 source processing is complete; each model has
3,271,090,620 predictions among 3,339,752,304 distinct canonical inputs. The
68,661,684 absent predictions remain absent. Source occurrence and annotation
counts are never relabeled unique variants. GENCODE v24 is the scoring annotation;
MANE describes model training. Legacy provenance is retained explicitly.

Full packaging continues in frozen Slurm job 32361767. Dependent CPU job 32417755
will authenticate its checkpoints, finish conversion if needed, rebuild native
tracks/reference evidence, reconcile the final census, and export a local full
snapshot. It stops for source-job scientific failures and does not publish.
Actual completion requires all-data-verified.json and review-ready.json; a queued
job or an intermediate manifest does not establish completion.

Local evidence and remaining checks are recorded in validation/local-review.json
and LOCAL_REVIEW.md. A failed test remains a release blocker until corrected.
There has been no new commit, push, merge or deployment for this review.
