"""Public, paged source evidence with no local paths or inferred model identity."""
import json
from pathlib import Path
from .build import ManifestLookup, atomic, compressed_json
from .format import canonical_json


def build_provenance(config_path, output):
    root=Path(output); config=json.loads(Path(config_path).read_text())
    manifest=json.loads((root/'manifest.json').read_text()); lookups={m:ManifestLookup(config[m+'_manifest']) for m in ('r10','r13')}
    ids=config.get('chunks') or list(lookups['r10']); pages={}
    try:
        for chunk in ids:
            entry={}
            for model,lookup in lookups.items():
                row=lookup.get(chunk) or {}
                entry[model]={'state':row.get('provenance_state') or 'not_recorded','outputSha256':row.get('output_file_sha256',''),'receiptSha256':row.get('receipt_sha256',''),'sourceOccurrences':int(row.get('output_records') or 0)}
            pages.setdefault(str((chunk-1)//1000),{})[str(chunk)]=entry
    finally:
        for lookup in lookups.values():lookup.close()
    files={f['path']:f for f in manifest['files']}; descriptors={}
    for page,entries in pages.items():
        name=f'provenance-{int(page):04d}.json.gz';d=compressed_json(root/name,entries);d['path']=name;descriptors[page]=d;files[name]=d
    manifest['provenance']=descriptors;manifest['files']=list(files.values())
    atomic(root/'manifest.json',canonical_json(manifest));return descriptors
