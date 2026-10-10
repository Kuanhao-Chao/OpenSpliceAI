"""Local range server and institutional-host publication checks."""
from __future__ import annotations

import hashlib
import http.server
import json
import re
import urllib.request
from functools import partial
from pathlib import Path


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, ETag, Accept-Ranges")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def send_head(self):
        path = Path(self.translate_path(self.path))
        byte_range = self.headers.get("Range")
        if not byte_range or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
        size = path.stat().st_size
        if not match:
            self.send_error(416, "one explicit byte range is required")
            return None
        start = int(match[1])
        end = min(int(match[2]) if match[2] else size - 1, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        handle = path.open("rb")
        handle.seek(start)
        self._range_remaining = end - start + 1
        return handle

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        del self._range_remaining
        while remaining:
            data = source.read(min(65536, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def serve(directory, port=8765):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=str(Path(directory).resolve())))
    print(f"Browser data: http://127.0.0.1:{port}/manifest.json", flush=True)
    server.serve_forever()


def probe(url, origin="https://khchao.com"):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Origin": origin}), timeout=30) as response:
        manifest = json.load(response)
        cors = response.headers.get("Access-Control-Allow-Origin")
        if cors not in ("*", origin):
            raise ValueError("manifest does not allow cross-origin browser reads")
    base = manifest.get("baseUrl") or url.rsplit("/", 1)[0] + "/"
    descriptor = next(f for f in manifest["files"] if f["path"].endswith(".pack") and f["bytes"] >= 64)
    req = urllib.request.Request(urllib.parse.urljoin(base, descriptor["path"]),
                                 headers={"Origin": origin, "Range": "bytes=0-63", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-63/{descriptor['bytes']}":
            raise ValueError("host does not deliver correct byte ranges")
        if response.headers.get("Access-Control-Allow-Origin") not in ("*", origin):
            raise ValueError("packed data is missing CORS")
        exposed = response.headers.get("Access-Control-Expose-Headers", "").lower()
        if "content-range" not in exposed and exposed.strip() != "*":
            raise ValueError("Content-Range must be exposed to browser JavaScript")
        if response.headers.get("Content-Encoding", "identity") != "identity" or len(response.read(65)) != 64:
            raise ValueError("host changed packed byte representation")
    return {"passed": True, "manifest": url, "origin": origin, "rangeFile": descriptor["path"]}


def verify(directory):
    from .build import file_sha
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    for file in manifest["files"]:
        relative = Path(file['path'])
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts or
                '\\' in file['path'] or '\n' in file['path'] or '\r' in file['path'] or file['path'] in seen):
            raise ValueError('manifest requires unique, safe relative file paths')
        seen.add(file['path'])
        path = root / file["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("manifest path escapes snapshot")
        if path.stat().st_size != file["bytes"] or file_sha(path) != file["sha256"]:
            raise ValueError(f"snapshot file content mismatch: {file['path']}")
    return {"passed": True, "files": len(manifest["files"]),
            "bytes": sum(f["bytes"] for f in manifest["files"]), "dataset": manifest["id"]}
