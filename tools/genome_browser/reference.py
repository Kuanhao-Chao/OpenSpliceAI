"""Read the pinned FASTA through its existing .fai, without loading the genome."""
from __future__ import annotations

from pathlib import Path


class Reference:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open("rb")
        self.contigs = {}
        for line in Path(str(path) + ".fai").read_text().splitlines():
            name, length, offset, bases, width, *_ = line.split("\t")
            values = tuple(map(int, (length, offset, bases, width)))
            if name in self.contigs or values[0] < 1 or values[2] < 1 or values[3] < values[2]:
                raise ValueError("invalid or duplicate FASTA index entry")
            begin = max(0, values[1] - 65536)
            self.handle.seek(begin)
            preceding = self.handle.read(values[1] - begin).splitlines()
            if not preceding or not preceding[-1].startswith(b">" + name.encode() + b" ") and preceding[-1] != b">" + name.encode():
                raise ValueError(f"FASTA index offset does not follow {name}'s header")
            self.contigs[name] = values

    def read(self, chrom: str, start: int, end: int) -> str:
        length, offset, bases, width = self.contigs[chrom]
        if not 0 <= start <= end <= length:
            raise ValueError(f"reference range outside {chrom}: {start}-{end}")
        if start == end:
            return ""
        first = offset + start // bases * width + start % bases
        last = offset + (end - 1) // bases * width + (end - 1) % bases + 1
        self.handle.seek(first)
        result = self.handle.read(last - first).replace(b"\n", b"").replace(b"\r", b"")
        if len(result) != end - start:
            raise ValueError("FASTA index does not agree with reference")
        return result.decode("ascii").upper()

    def close(self):
        self.handle.close()


def genes_from_tsv(path):
    genes = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, chrom, strand, start, end, starts, ends = line.split("\t")[:7]
        exon_starts = [int(x) for x in starts.split(",") if x]
        exon_ends = [int(x) for x in ends.split(",") if x]
        if len(exon_starts) != len(exon_ends):
            raise ValueError(f"unequal exon start/end counts for {name}")
        exons = list(zip(exon_starts, exon_ends))
        start, end = int(start), int(end)
        if strand not in ("+", "-") or not exons or any(not start <= a < b <= end for a, b in exons):
            raise ValueError(f"invalid annotation for {name}")
        if any(exons[i][0] < exons[i-1][1] for i in range(1, len(exons))):
            raise ValueError(f"unsorted or overlapping exons for {name}")
        genes.append({"name": name, "chrom": chrom, "strand": strand,
                      "start": start, "end": end, "exons": exons})
    return genes
