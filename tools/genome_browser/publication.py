"""Concrete publication inventory; uploading waits for an institutional target."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .build import atomic, file_sha
from .format import canonical_json
from .hosting import verify


def export_snapshot(directory, output):
    """Export an authenticated allowlist, never a preparation directory tree."""
    directory, output = Path(directory), Path(output)
    proof = verify(directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError('snapshot export destination must be empty')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('scope')=='source-collection' and not manifest.get('releaseVerified'):
        raise ValueError('full snapshot has not passed its release verification gate')
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        source, target = directory / entry['path'], output / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic(output / 'manifest.json', (directory / 'manifest.json').read_bytes())
    verify(output)
    return proof


def publication_plan(directory, public_url, output):
    directory, output = Path(directory), Path(output)
    verification = verify(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('scope')=='source-collection' and not manifest.get('releaseVerified'):
        raise ValueError('full snapshot has not passed its release verification gate')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', manifest['id']):
        raise ValueError('snapshot ID is not safe for a public directory')
    url = urlparse(public_url)
    if url.scheme != 'https' or not url.netloc or url.query or url.fragment:
        raise ValueError('publication needs a concrete institutional HTTPS snapshot URL')
    public_url = public_url.rstrip('/') + '/'
    catalog = {'datasets': [{'id': manifest['id'], 'label': manifest['label'],
                             'manifest': public_url + 'manifest.json'}]}
    files = [f['path'] for f in manifest['files']]
    if len(set(files)) != len(files) or any('\n' in f for f in files):
        raise ValueError('duplicate or unsafe publication path')
    atomic(output / 'data-files.txt', ('\n'.join(files) + '\n').encode())
    atomic(output / 'catalog.json', canonical_json(catalog))
    atomic(output / 'checksums.sha256', ('\n'.join(f"{f['sha256']}  {f['path']}" for f in manifest['files']) +
                                        f"\n{file_sha(directory / 'manifest.json')}  manifest.json\n").encode())
    plan = {'dataset': manifest['id'], 'publicUrl': public_url,
            'verification': verification, 'manifestSha256': file_sha(directory / 'manifest.json'),
            'uploadOrder': ['data-files.txt inventory', 'manifest.json', 'catalog.json'],
            'checks': ['local content hashes', 'uploaded content hashes', 'HTTPS range/CORS probe', 'browser acceptance audit'],
            'applicationSettings': {'catalogUrl': public_url + 'catalog.json'},
            'note': 'No remote files were uploaded. Publish an immutable snapshot and its manifest before the catalog.'}
    atomic(output / 'publication.json', canonical_json(plan))
    return plan
