# OSGB1 static genome data

All file paths in a published manifest are relative to its public `baseUrl`
(or the manifest directory). Private source paths stay in local preparation
configuration. Each snapshot has an immutable ID and source/audit hashes.

`scores-NNNN.pack` concatenates independently gzip-compressed logical blocks.
`index-CONTIG-WINDOW.json.gz` lists every overlapping block fragment's byte
offset, compressed length, raw length, SHA-256 and 0-based genomic interval.
An index window spans 16,777,216 reference bases. A logical score block is
16,384 bp; it can have several fragments at manifest-shard boundaries. Clients
read **all** fragments. Their occurrence IDs prevent accidental record loss.

Uncompressed score blocks begin with eight bytes `OSGB1\0\0\0`, a little-endian
uint32 JSON-header length, UTF-8 JSON, then column arrays. The header describes
each column's name, type, count, byte offset and byte length. Arrays use little
endian uint32 (`I`), uint8 (`B`), or signed int16 (`h`). Row columns are `pos`,
`ref`, `alt`, `chunk`, `ordinal`, `flags`. Allele codes index
`ACGTNRYWSKMBDHV.`. Flag bit 0 marks a reference mismatch and bit 1 marks a
source occurrence with accepted r13 evidence. POS is zero based.

For each of `r10`, `r13`, `baseline`, annotations have `row`, `gene`, four
`DS_EVENT` and four `DP_EVENT` columns. Gene IDs index the block dictionary.
DS is an integer scaled by 100000. A zero is a real recorded value; an empty
annotation list is missing. DP is the original signed genomic-forward offset.
Preserve distinct DS/DP annotations even when their gene or allele is repeated.
Identical annotations may be displayed together with every occurrence ID.
An allele with some unaccepted source occurrences has partial r13 coverage.

Summary bins are 1,024 bp. Each bin stores source occurrence counts, accepted
r13 occurrence counts, REF mismatches, and annotation counts, zero counts,
channel maxima and channel sums for each model. These are **not unique-variant
counts**. Partial bins are not prorated when clipped to a viewport. Contig
overview bins span 1,048,576 bp. Wide-view filters cannot recover allele/gene-specific
statistics; the UI should require exact zoom for those filters.

Reference and FM search files use fixed 44-byte page-directory entries:
uint64 byte offset, uint32 compressed length, 32 raw SHA-256 bytes. Independent
gzip pages contain reference ASCII (16,384 bp) or FM-index data (4,096 BWT rows).
FM pages contain uint32 row count, one uint32 absolute rank per alphabet symbol,
ASCII BWT, and one uint32 SA sample per row. Unsampled rows use `0xffffffff`.
SA samples occur when the reference position is divisible by 32. LF locate
takes at most 31 steps. Reference `ranges`, when present, explicitly limit
sequence availability in a review subset. Whole-genome search requires indexes
for **every** requested reference contig, including patches and alt contigs.

The global `genes.json.gz` contains search locations without exon geometry.
Each annotated contig has a separate, lazily fetched gene-feature descriptor.
The independent overview descriptor prevents loading regional score indexes at
startup. Compressed JSON descriptors include both stored `bytes`/`sha256` and
`decodedBytes`/`decodedSha256`, so transparent HTTP metadata decompression can
be authenticated. Indexed pack and page-directory bytes must stay unchanged.

File SHA-256 verifies complete publication; per-block/page SHA-256 verifies
browser range responses. HTTP gzip transformations are disabled: compression
is internal to the file and offsets refer to stored bytes. The host returns
206, exact Content-Range, CORS and exposed Content-Range. Missing/invalid bytes
raise an error and disable stable exports; they never become empty data.

The bundled `review-subset` on the application's own origin uses whole-file
reads for indexed artifacts at most 1 MiB each. GitHub Pages may recompress
ranged representations for Firefox. These bounded reads authenticate the
complete stored-file SHA-256 before slicing and checking the block/page bytes.
Genome-wide datasets and external origins always retain the strict range
contract; they never fall back to downloading a complete packed file.
