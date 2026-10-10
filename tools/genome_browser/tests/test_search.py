import gzip
import importlib.util
import struct
import unittest

from tools.genome_browser.search_index import fm_pages


@unittest.skipUnless(importlib.util.find_spec('pydivsufsort'), 'optional native preparation dependency')
class SearchIndexTests(unittest.TestCase):
    def test_native_fm_rank_search_and_sampled_locate(self):
        sequence = ('ACGTTGCAAN' * 850) + 'GATTACA'
        pages = fm_pages(sequence)
        meta = next(pages)
        decoded = []
        for compressed in pages:
            raw = gzip.decompress(compressed)
            n = struct.unpack_from('<I', raw)[0]
            count = len(meta['alphabet'])
            ranks = struct.unpack_from('<' + 'I' * count, raw, 4)
            bwt = raw[4 + 4 * count:4 + 4 * count + n]
            samples = struct.unpack_from('<' + 'I' * n, raw, 4 + 4 * count + n)
            decoded.append((ranks, bwt, samples))

        def rank(char, end):
            if end == meta['rows']: return meta['counts'][char]
            page, local = divmod(end, meta['pageBp'])
            ranks, bwt, _ = decoded[page]
            return ranks[meta['alphabet'].index(char)] + bwt[:local].count(ord(char))

        def locate(row):
            for step in range(meta['sampleRate']):
                page, local = divmod(row, meta['pageBp'])
                _, bwt, samples = decoded[page]
                if samples[local] != 0xffffffff: return (samples[local] + step) % meta['rows']
                char = chr(bwt[local])
                row = meta['cumulative'][char] + rank(char, row)
            self.fail('locate exceeded sampled bound')

        for query in ('ACGT', 'AN', 'GATTACA', 'ZZZ', 'CA', 'A'):
            lo, hi = 0, meta['rows']
            for char in reversed(query):
                if char not in meta['cumulative']: lo, hi = 0, 0; break
                lo, hi = meta['cumulative'][char] + rank(char, lo), meta['cumulative'][char] + rank(char, hi)
            observed = sorted(locate(i) for i in range(lo, hi))
            expected = [i for i in range(len(sequence)) if sequence.startswith(query, i)]
            self.assertEqual(observed, expected, query)


if __name__ == '__main__':
    unittest.main()
