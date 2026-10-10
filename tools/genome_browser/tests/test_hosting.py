import functools
import http.server
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.genome_browser.hosting import RangeHandler


class LinkTests(unittest.TestCase):
    def test_project_prefix_and_paths_escaping_the_output(self):
        path = Path(__file__).resolve().parents[3] / 'docs/check_links.py'
        spec = importlib.util.spec_from_file_location('browser_link_check', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'genome/assets').mkdir(parents=True)
            (root / 'genome/assets/app.js').write_text('')
            (root / 'genome/index.html').write_text('<script src="/OpenSpliceAI/genome/assets/app.js"></script>')
            with patch.object(sys, 'argv', ['check_links', str(root)]), redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)
            (root / 'genome/index.html').write_text('<script src="../../escaped.js"></script>')
            with patch.object(sys, 'argv', ['check_links', str(root)]), redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 1)


class HostingTests(unittest.TestCase):
    def test_range_cors_and_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, 'file.pack').write_bytes(b'0123456789')
            try:
                server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(RangeHandler, directory=temp))
            except PermissionError:
                self.skipTest('local socket binding requires unsandboxed verification')
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}/file.pack'
                with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=2-5'})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), b'2345')
                    self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
                    self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=20-25'}))
                self.assertEqual(caught.exception.code, 416)
            finally:
                server.shutdown(); server.server_close(); worker.join(timeout=2)
