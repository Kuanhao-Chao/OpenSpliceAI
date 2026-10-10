import json
import tempfile
import unittest
from pathlib import Path

from tools.genome_browser.build import file_sha
from tools.genome_browser.hosting import verify
from tools.genome_browser.publication import publication_plan


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot = self.root / 'snapshot'; self.snapshot.mkdir()
        data = self.snapshot / 'scores-0000.pack'; data.write_bytes(b'verified bytes')
        self.manifest = {'id': 'immutable-test', 'label': 'test snapshot', 'files': [
            {'path': data.name, 'bytes': data.stat().st_size, 'sha256': file_sha(data)}]}
        self.save()

    def save(self):
        (self.snapshot / 'manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_excludes_private_state_and_publishes_manifest_last(self):
        (self.snapshot / 'private-config.json').write_text('/private/source.vcf')
        output = self.root / 'plan'
        plan = publication_plan(self.snapshot, 'https://institution.example/data/immutable-test/', output)
        self.assertTrue(plan['verification']['passed'])
        self.assertEqual((output / 'data-files.txt').read_text(), 'scores-0000.pack\n')
        self.assertIn('manifest.json', (output / 'checksums.sha256').read_text())
        self.assertEqual(plan['uploadOrder'][-2:], ['manifest.json', 'catalog.json'])
        self.assertEqual(json.loads((output / 'catalog.json').read_text())['datasets'][0]['manifest'],
                         'https://institution.example/data/immutable-test/manifest.json')

    def test_corrupt_data_and_unsafe_urls_prevent_publication(self):
        output = self.root / 'plan'
        for url in ('http://institution.example/data', 'https://institution.example/data?token=x',
                    'https://institution.example/data#fragment', 'relative/path'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                publication_plan(self.snapshot, url, output)
        self.assertFalse(output.exists())
        (self.snapshot / 'scores-0000.pack').write_bytes(b'changed bytes!')
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            publication_plan(self.snapshot, 'https://institution.example/data', output)
        self.assertFalse(output.exists())

    def test_absolute_traversing_and_duplicate_manifest_paths_rejected(self):
        original = dict(self.manifest['files'][0])
        for name in (str(self.snapshot / original['path']), '../snapshot/' + original['path'], 'x\\y', 'x\ny'):
            self.manifest['files'] = [{**original, 'path': name}]; self.save()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'safe relative'):
                verify(self.snapshot)
        self.manifest['files'] = [original, original]; self.save()
        with self.assertRaisesRegex(ValueError, 'safe relative'):
            verify(self.snapshot)
