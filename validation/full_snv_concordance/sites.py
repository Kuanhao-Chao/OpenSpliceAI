"""Annotated splice-site coordinates, derived the way the scorer derives them.

The point of this module is self-consistency. The delta scores being analysed were
produced by `openspliceai variant -M 1 -A data/grch38_chr.txt`, and that masking
decision is made against a specific set of coordinates read out of that specific
table. If this module derived sites from a different annotation -- the RefSeq GFF
in `results/all_ss/splice_sites.bed`, say -- the distance strata would measure the
disagreement between two annotations as much as anything about the predictors.

Two conventions are copied deliberately from `openspliceai/variant/utils.py`:

* **Coordinates** (`variant/utils.py:282-292`). In this table `EXON_START` is
  0-based and `EXON_END` is 1-based-inclusive, so the scorer loads
  ``exon_starts + 1`` and leaves ``exon_ends`` alone; both are then directly
  comparable to a 1-based VCF ``POS``. The upstream SpliceAI ``Annotator`` does
  the same, so this is the format's convention rather than a local quirk.
* **Strand** . Exon arrays are in ascending genomic order whatever the strand. On
  ``+`` the internal exon ends are donors and the internal exon starts are
  acceptors; on ``-`` those roles swap.

One deliberate divergence: the scorer masks against
``np.union1d(exon_starts, exon_ends)``, which *includes* the first exon's start
and the last exon's end. Those are transcript ends, not splice sites, so they are
excluded here -- a gain call at a TSS is not a splice-site call, and counting it
as one would flatter both predictors equally but describe neither. The reports
state this difference where it matters.

`Annotator` is not reused directly even though it owns the same parsing, because
its constructor also loads a reference FASTA and the models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

ACCEPTOR = "acceptor"
DONOR = "donor"
AMBIGUOUS = "ambiguous"

#: Upper bound (inclusive) of each distance bin, in bases, and its label. The
#: first two bins isolate the boundary itself and a symmetric 1–2 bp neighbourhood.
#: The latter is a proximity bin, not a strand-aware essential-dinucleotide label.
DISTANCE_BINS: Tuple[Tuple[Optional[int], str], ...] = (
    (0, "at_site"),
    (2, "1-2"),
    (10, "3-10"),
    (50, "11-50"),
    (500, "51-500"),
    (None, ">500"),
)
NO_SITE = "no_site"


def distance_bin(distance: Optional[int]) -> str:
    """Label the distance to the nearest annotated splice site."""
    if distance is None:
        return NO_SITE
    distance = abs(int(distance))
    for upper, label in DISTANCE_BINS:
        if upper is None or distance <= upper:
            return label
    return DISTANCE_BINS[-1][1]


@dataclass(frozen=True)
class SiteIndex:
    """Sorted splice-site positions per contig, with the type of each site.

    Positions are 1-based to match VCF ``POS``. Lookup is a binary search against
    a sorted array per contig, which keeps the whole structure a few megabytes and
    the per-variant cost to a couple of comparisons -- it runs once per paired
    annotation over billions of rows.
    """

    positions: Mapping[str, np.ndarray]
    types: Mapping[str, np.ndarray]
    digest: str
    site_count: int
    gene_count: int

    def nearest(self, chrom: str, pos: int) -> Tuple[Optional[int], str]:
        """Signed distance to the nearest site and that site's type.

        The distance is ``site - pos``, so a negative value means the nearest site
        lies before the variant. Callers that only bin by magnitude can ignore the
        sign; it is kept because direction matters when reading a gain call.
        """
        array = self.positions.get(chrom)
        if array is None or array.size == 0:
            return None, NO_SITE
        index = int(np.searchsorted(array, pos))
        best: Optional[int] = None
        best_index = -1
        nearest_types = set()
        for candidate in (index - 1, index):
            if candidate < 0 or candidate >= array.size:
                continue
            delta = int(array[candidate]) - int(pos)
            if best is None or abs(delta) < abs(best):
                best, best_index = delta, candidate
                nearest_types = {str(self.types[chrom][candidate])}
            elif abs(delta) == abs(best):
                nearest_types.add(str(self.types[chrom][candidate]))
        if best is None:
            return None, NO_SITE
        kind = str(self.types[chrom][best_index]) if len(nearest_types) == 1 else AMBIGUOUS
        return best, kind

    def stratum_key(self, chrom: str, pos: int) -> str:
        """``"<distance bin>:<site type>"`` -- the key used by the site_distance stratum."""
        distance, site_type = self.nearest(chrom, pos)
        return f"{distance_bin(distance)}:{site_type}"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_positions(field: str) -> np.ndarray:
    return np.asarray([int(value) for value in str(field).split(",") if value], dtype=np.int64)


def load_site_index(path: str | Path) -> SiteIndex:
    """Derive the splice-site index from a SpliceAI-format annotation table."""
    path = Path(path)
    per_chrom: Dict[str, list] = {}
    gene_count = 0

    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            columns = {name: header.index(name) for name in
                       ("#NAME", "CHROM", "STRAND", "EXON_START", "EXON_END")}
        except ValueError as exc:  # pragma: no cover - malformed input guard
            raise ValueError(f"{path}: annotation is missing a required column ({exc})") from exc

        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[columns["CHROM"]]
            strand = fields[columns["STRAND"]]
            # `+ 1` on starts only: EXON_START is 0-based, EXON_END already 1-based
            # inclusive. This mirrors variant/utils.py:282-292 exactly.
            starts = _split_positions(fields[columns["EXON_START"]]) + 1
            ends = _split_positions(fields[columns["EXON_END"]])
            if starts.size != ends.size or starts.size == 0:
                continue
            gene_count += 1
            if starts.size < 2:
                # A single-exon transcript has no internal boundary, so no splice site.
                continue
            order = np.argsort(starts)
            starts, ends = starts[order], ends[order]
            # Internal boundaries only: the first start and the last end are
            # transcript ends, not splice sites.
            internal_starts, internal_ends = starts[1:], ends[:-1]
            if strand == "-":
                donors, acceptors = internal_starts, internal_ends
            else:
                donors, acceptors = internal_ends, internal_starts
            bucket = per_chrom.setdefault(chrom, [])
            bucket.append((donors, DONOR))
            bucket.append((acceptors, ACCEPTOR))

    positions: Dict[str, np.ndarray] = {}
    types: Dict[str, np.ndarray] = {}
    total = 0
    for chrom, entries in per_chrom.items():
        all_positions = np.concatenate([values for values, _ in entries])
        all_types = np.concatenate([
            np.full(values.size, label, dtype=object) for values, label in entries
        ])
        order = np.argsort(all_positions, kind="stable")
        all_positions, all_types = all_positions[order], all_types[order]
        # Collapse shared coordinates without letting annotation row order pick
        # the biological type when overlapping genes assign opposite roles.
        keep = np.ones(all_positions.size, dtype=bool)
        keep[1:] = all_positions[1:] != all_positions[:-1]
        starts = np.flatnonzero(keep)
        for lo, hi in zip(starts, np.r_[starts[1:], all_positions.size]):
            if len(set(all_types[lo:hi])) > 1:
                all_types[lo] = AMBIGUOUS
        positions[chrom] = all_positions[keep]
        types[chrom] = all_types[keep]
        total += int(keep.sum())

    return SiteIndex(
        positions=positions,
        types=types,
        digest=_digest_file(path),
        site_count=total,
        gene_count=gene_count,
    )


__all__ = ["SiteIndex", "load_site_index", "distance_bin", "DISTANCE_BINS",
           "ACCEPTOR", "DONOR", "NO_SITE"]
