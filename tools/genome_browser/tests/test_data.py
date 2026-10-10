import csv
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.genome_browser.build import atomic, file_sha, finalize, prepare, prepare_parallel
from tools.genome_browser.format import (canonical_json, decode_block, ds_integer,
                                        encode_block, merge_summaries, parse_vcf, summarize)
from tools.genome_browser.hosting import verify
from tools.genome_browser.reference import Reference


def row(pos=0, ref="A", alt="C", flags=2, model=None, ordinal=1):
    return {"pos": pos, "ref": ref, "alt": alt, "flags": flags, "chunk": 1, "ordinal": ordinal,
            "models": {"r10": model or [], "r13": [], "baseline": []}}


class FormatTests(unittest.TestCase):
    def test_lossless_five_decimals_and_signed_offsets(self):
        entries = [{"gene": "GENE,\"minus", "ds": [1, 0, 100000, 12456], "dp": [-50, 0, 50, -2]}]
        raw = encode_block("chr1", 0, [row(model=entries)])
        header, cols = decode_block(raw)
        self.assertEqual(header["genes"], ['GENE,"minus'])
        self.assertEqual(cols["r10.DS_AG"], [1])
        self.assertEqual(cols["r10.DP_AG"], [-50])
        self.assertEqual(cols["r10.DS_DG"], [100000])
        self.assertEqual(cols["r13.row"], [])

    def test_vcf_repeated_gene_and_multiallelic_entries(self):
        r = parse_vcf('chr1\t2\t.\tA\tC,G\t.\t.\tOpenSpliceAI=C|G|0|0|0|0|-50|0|0|50,C|G|0|0|0|0|50|0|0|-50,G|H|0.00001|0|0|0|1|0|0|0\n', 2, 3)
        self.assertEqual(len(r['annotations']['OpenSpliceAI']), 3)
        self.assertEqual(r['alts'], ['C', 'G'])
        self.assertEqual(r['pos'], 1)

    def test_missing_whole_annotation_is_absent(self):
        r = parse_vcf('chr1\t1\t.\tA\tC\t.\t.\tOpenSpliceAI=C|G|.|.|.|.|.|.|.|.\n', 1, 1)
        self.assertEqual(r['annotations']['OpenSpliceAI'], [])

    def test_bad_scores_fail_without_rounding(self):
        for s in ['NaN', '-0.1', '1.1', '0.000001', '.', 'inf']:
            with self.subTest(s=s), self.assertRaises(ValueError):
                ds_integer(s)
        self.assertEqual(ds_integer('0.00001'), 1)

    def test_truncated_block_fails(self):
        with self.assertRaises((ValueError, struct.error)):
            decode_block(encode_block('chr1', 0, [row()])[:-1])

    def test_summaries_preserve_zero_and_pending_denominators(self):
        a = row(model=[{"gene": "G", "ds": [0]*4, "dp": [2]*4}])
        b = row(1, flags=0, ordinal=2)
        summaries = merge_summaries([summarize([a], 0), summarize([b], 0)])
        self.assertEqual(summaries[0]['rows'], 2)
        self.assertEqual(summaries[0]['acceptedR13'], 1)
        self.assertEqual(summaries[0]['models']['r10']['zero'], 1)
        self.assertNotIn('r13', summaries[0]['models'])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref = self.root / 'ref.fna'
        self.ref.write_text('>chr1\nACGTACGT\n')
        Path(str(self.ref) + '.fai').write_text('chr1\t8\t6\t8\t9\n')
        self.annotation = self.root / 'genes.tsv'
        self.annotation.write_text('#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\nG\tchr1\t-\t0\t8\t0,4,\t2,8,\n')
        self.vcf = self.root / 'r10.vcf'
        self.vcf.write_text('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t1\t.\tA\tC\t.\t.\tOpenSpliceAI=C|G|0.00001|0|0|0|-1|0|0|0\nchr1\t1\t.\tG\tC\t.\t.\t.\n')
        self.manifest = self.root / 'r10.tsv'
        self.write_manifest(self.manifest, self.vcf)
        self.config = {'id': 'test', 'label': 'test', 'reference': str(self.ref), 'annotation': str(self.annotation),
                       'r10_manifest': str(self.manifest), 'shard_size': 1, 'models': {'r10': {'precision': 5}}}
        self.config_path = self.root / 'config.json'
        self.config_path.write_bytes(canonical_json(self.config))
        self.output = self.root / 'output'

    def write_manifest(self, path, vcf, state='valid'):
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, delimiter='\t', fieldnames=['chunk_id', 'state', 'error', 'output_path', 'output_file_sha256', 'output_records'])
            writer.writeheader()
            writer.writerow({'chunk_id': 1, 'state': state, 'error': '', 'output_path': str(vcf), 'output_file_sha256': file_sha(vcf), 'output_records': 2})

    def tearDown(self):
        self.temp.cleanup()

    def test_reference_first_last_and_boundaries(self):
        ref = Reference(self.ref)
        self.assertEqual(ref.read('chr1', 0, 8), 'ACGTACGT')
        self.assertEqual(ref.read('chr1', 7, 8), 'T')
        with self.assertRaises(ValueError): ref.read('chr1', -1, 1)
        ref.close()

    def test_prepare_roundtrip_mismatch_resume_and_verify(self):
        prepare(self.config_path, self.output)
        header, cols = decode_block(gzip.decompress((self.output / 'scores-0000.pack').read_bytes()))
        self.assertEqual(cols['flags'], [0, 1])
        self.assertEqual(cols['r10.DS_AG'], [1])
        prepare(self.config_path, self.output)
        manifest = finalize(self.config_path, self.output)
        self.assertEqual(manifest['sourceOccurrences'], 2)
        self.assertEqual(manifest['acceptedR13Occurrences'], 0)
        self.assertTrue(verify(self.output)['passed'])

    def test_content_modified_after_audit_rejected(self):
        self.vcf.write_text(self.vcf.read_text().replace('0.00001', '0.00002'))
        with self.assertRaisesRegex(ValueError, 'content hash'):
            prepare(self.config_path, self.output)
        self.assertFalse((self.output / 'manifest.json').exists())

    def test_two_workers_commit_disjoint_shards_and_resume(self):
        second = self.root / 'r10-second.vcf'
        second.write_text(self.vcf.read_text().replace('chr1\t1\t', 'chr1\t2\t'))
        with self.manifest.open('a') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow([2, 'valid', '', str(second), file_sha(second), 2])
        prepare_parallel(self.config_path, self.output, workers=2)
        first = finalize(self.config_path, self.output)
        self.assertEqual(first['preparedShards'], 2)
        self.assertEqual(first['sourceOccurrences'], 4)
        self.assertTrue(verify(self.output)['passed'])
        prepare_parallel(self.config_path, self.output, workers=2)
        resumed = finalize(self.config_path, self.output)
        self.assertEqual([f['sha256'] for f in first['files']], [f['sha256'] for f in resumed['files']])

    def test_unaccepted_source_rejected(self):
        self.write_manifest(self.manifest, self.vcf, 'invalid')
        with self.assertRaisesRegex(ValueError, 'accepted'):
            prepare(self.config_path, self.output)

    def test_r13_wrong_source_keys_rejected(self):
        r13 = self.root / 'r13.vcf'
        r13.write_text(self.vcf.read_text().replace('\tA\tC\t', '\tA\tT\t').replace('=C|', '=T|'))
        m13 = self.root / 'r13.tsv'; self.write_manifest(m13, r13)
        self.config['r13_manifest'] = str(m13)
        self.config_path.write_bytes(canonical_json(self.config))
        with self.assertRaisesRegex(ValueError, 'source keys'):
            prepare(self.config_path, self.output)

    def test_modified_checkpoint_detected(self):
        prepare(self.config_path, self.output)
        path = self.output / 'scores-0000.pack'; data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'resume verification'):
            prepare(self.config_path, self.output)

    def test_final_r13_promotion_requires_final_audit(self):
        self.config['r13_final'] = True
        self.config_path.write_bytes(canonical_json(self.config))
        prepare(self.config_path, self.output)
        with self.assertRaisesRegex(ValueError, 'promotion'):
            finalize(self.config_path, self.output)


if __name__ == '__main__':
    unittest.main()
