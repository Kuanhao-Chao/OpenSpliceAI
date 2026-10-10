"""Frozen CPU preparation jobs, separate from the production scoring controller."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .build import accepted, atomic, file_sha, finalize, load_manifest, prepare_parallel
from .format import canonical_json, sha256
from .hosting import verify
from .review import make_config
from .search_index import attach_reference, build_reference


def code_identity(directory):
    return sha256(canonical_json({p.name: file_sha(p) for p in sorted(Path(directory).glob('*.py'))}))


def freeze(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if (destination / 'config.json').exists():
        raise ValueError('use a new frozen preparation directory')
    destination.mkdir(parents=True, exist_ok=True)
    config = make_config(root)
    for key in ('chunks', 'regions', 'reference_review_regions'):
        config.pop(key, None)
    for key in ('r10_manifest', 'r13_manifest'):
        target = destination / f'{key}.tsv'
        shutil.copyfile(config[key], target)
        config[key] = str(target)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    config.update({'id': f'grch38-r10-r13preview-{suffix}', 'label': 'GRCh38 r10 complete / r13 audited preview',
                   'scope': 'source-collection', 'shard_size': 1000, 'snapshot_directory': str(destination / 'data'),
                   'scoring_campaign': str(root / 'results/full_snv_concordance/hybrid_finish_20261010_gpu4'),
                   'r13_evidence': 'Frozen legacy/receipt-bound audit plus freshly content-verified worker passes. Preview acceptance is not the final r13 completion gate.'})
    candidates = set()
    for campaign in ('hybrid_finish_20261009', 'hybrid_finish_20261010_gpu4'):
        for path in (root / 'results/full_snv_concordance' / campaign / 'state').glob('worker_*.json'):
            worker = json.loads(path.read_text())
            candidates.update(int(n) for n in worker.get('passed_chunks', []))
    old = load_manifest(config['r13_manifest'])
    extra = sorted(n for n in candidates if n in old and not accepted(old[n]))
    atomic(destination / 'candidate_chunks.json', canonical_json(extra))
    auditor = root / 'results/full_snv_concordance/finish_20261001/code/audit_vcfs.py'
    shutil.copyfile(auditor, destination / 'auditor.py')
    config['auditor_sha256'] = file_sha(auditor)
    for model in ('r10', 'r13'):
        seed = config['models'][model]['seed']
        path = root / f'models/openspliceai-mane/10000nt/model_10000nt_rs{seed}.pt'
        actual = file_sha(path)
        if actual != config['models'][model]['sha256']:
            raise ValueError(f'model identity differs from reviewed configuration: {model}')
    code = destination / 'code/tools/genome_browser'
    code.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, code / path.name)
    config['builder_sha256'] = code_identity(code)
    source_census = root / 'results/full_snv_concordance/cpu_recovery_20261008/verification/source_keys.json'
    census = json.loads(source_census.read_text())
    config['source_census'] = {k: census[k] for k in ('passed', 'records', 'allele_records', 'unique_allele_keys',
           'unique_snvs', 'duplicate_allele_entries', 'positions', 'canonical_three_alleles_every_position') if k in census}
    config['source_census']['sha256'] = file_sha(source_census)
    atomic(destination / 'config.json', canonical_json(config))
    atomic(destination / 'freeze.json', canonical_json({'created': suffix, 'candidateWorkerPasses': len(extra),
           'r10ManifestSha256': file_sha(config['r10_manifest']), 'r13InitialManifestSha256': file_sha(config['r13_manifest']),
           'builderSha256': config['builder_sha256'], 'modelIdentitiesChecked': True}))
    return config


def extend_preview(root, config, workers):
    if (root / 'preview-frozen.json').exists():
        return
    auditor_path = root / 'auditor.py'
    if file_sha(auditor_path) != config['auditor_sha256']:
        raise ValueError('frozen auditor changed')
    spec = importlib.util.spec_from_file_location('snapshot_auditor', auditor_path)
    auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
    primary = load_manifest(config['r10_manifest'])
    secondary = load_manifest(config['r13_manifest'])
    candidates = json.loads((root / 'candidate_chunks.json').read_text())
    supported = {line.split('\t')[1].encode() for line in Path(config['annotation']).read_text().splitlines() if line and not line.startswith('#')}
    def audit(n):
        return auditor.inspect_pair(n, primary[n]['input_path'], secondary[n]['output_path'], supported, 50, expected_seed='rs13')
    additional = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(audit, candidates):
            if accepted(row) and row['provenance_state'] == 'receipt_bound' and row['input_file_sha256'] == primary[int(row['chunk_id'])]['input_file_sha256']:
                secondary[int(row['chunk_id'])] = row; additional.append(int(row['chunk_id']))
    # This update occurs before any score packs; the resulting TSV is then frozen.
    if (root / 'data/configuration.sha256').exists():
        raise ValueError('cannot change a preview after score preparation starts')
    path = Path(config['r13_manifest']); temp = path.with_suffix('.partial')
    with temp.open('w') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=auditor.FIELDNAMES, lineterminator='\n')
        writer.writeheader(); writer.writerows(secondary[n] for n in sorted(secondary))
    temp.replace(path)
    atomic(root / 'preview-frozen.json', canonical_json({'acceptedChunks': sum(accepted(r) for r in secondary.values()),
           'freshlyAcceptedChunks': additional, 'candidateChunks': len(candidates), 'manifestSha256': file_sha(path),
           'classification': 'immutable preview, not final completion'}))


def parse_shared_quota(raw):
    """Read GPFS root group limits, including its optional table separator."""
    rows = [line.replace('|', ' ').split() for line in raw.splitlines()]
    matches = [row for row in rows if row[:3] == ['data', 'root', 'GRP']]
    if len(matches) != 1 or len(matches[0]) < 13:
        raise ValueError('shared quota could not be established')
    fields = matches[0]
    def size(value):
        number = re.fullmatch(r'([\d.]+)([KMGTPE]?)', value)
        if not number: raise ValueError('unknown quota byte unit')
        result = float(number[1]) * 1024 ** ('KMGTPE'.find(number[2]) + 1 if number[2] else 0)
        if not math.isfinite(result): raise ValueError('invalid quota byte value')
        return result
    used, limit, doubt = map(size, (fields[3], fields[5], fields[6]))
    files, file_limit, in_doubt = (int(fields[i]) for i in (8, 10, 11))
    return {'freeFiles': file_limit - files - in_doubt, 'freeBytes': limit - used - doubt}


def quota_guard(config, preparation_files=4000):
    raw = subprocess.check_output(['/usr/lpp/mmfs/bin/mmlsquota', '-g', 'ssalzbe1', '--block-size', 'auto', 'data'], text=True, timeout=30)
    quota = parse_shared_quota(raw)
    status = Path(config['scoring_campaign']) / 'CURRENT_STATUS.md'
    remaining = re.search(r'remaining: \*\*([\d,]+)\*\*', status.read_text())
    if not remaining: raise ValueError('current scoring reserve could not be established')
    n = int(remaining[1].replace(',', ''))
    output = Path(config['snapshot_directory'])
    existing = sum(p.is_file() for p in output.glob('*')) if output.exists() else 0
    pending_files = max(0, preparation_files - existing)
    if quota['freeFiles'] < pending_files + 10000 + 3 * n or quota['freeBytes'] < 100 * 1024**3 + 6_000_000 * n:
        raise ValueError('insufficient quota after preserving scoring output and safety reserves')
    return {**quota, 'reservedScoringChunks': n, 'preparationFileBudget': preparation_files,
            'existingPreparationFiles': existing, 'pendingPreparationFiles': pending_files, 'retainedFileMargin': 10000}


def run(root, role, workers):
    root = Path(root).resolve(); config = json.loads((root / 'config.json').read_text()); output = Path(config['snapshot_directory'])
    if code_identity(Path(__file__).parent) != config['builder_sha256']:
        raise ValueError('frozen builder source changed')
    marker = root / f'{role}-complete.json'
    if marker.exists(): return
    if (root / f'{role}-FAILED.json').exists(): raise ValueError('a scientific preparation failure requires review before retry')
    try:
        reserve = quota_guard(config, 16 if role == 'finalize' else 4000)
        print(json.dumps({'role': role, 'quota': reserve, 'workers': workers, 'GPUs': 0}), flush=True)
        if role == 'scores':
            extend_preview(root, config, workers)
            prepare_parallel(root / 'config.json', output, workers=workers)
            manifest = finalize(root / 'config.json', output)
            proof = verify(output)
            atomic(marker, canonical_json(proof))
        elif role == 'reference':
            build_reference(root / 'config.json', output, with_search=True)
            atomic(marker, canonical_json({'passed': True, 'role': role, 'GPUs': 0}))
        else:
            if not (root / 'scores-complete.json').exists() or not (root / 'reference-complete.json').exists():
                raise ValueError('both independent preparation jobs must complete before final publication verification')
            attach_reference(output)
            proof = verify(output)
            atomic(root / 'all-data-verified.json', canonical_json(proof))
            atomic(marker, canonical_json(proof))
        print(json.dumps({'role': role, 'state': 'complete'}), flush=True)
    except Exception as error:
        atomic(root / f'{role}-FAILED.json', canonical_json({'error': str(error), 'time': datetime.now(timezone.utc).isoformat()}))
        raise


def main():
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('freeze'); p.add_argument('root'); p.add_argument('destination')
    p = subs.add_parser('run'); p.add_argument('directory'); p.add_argument('role', choices=['scores', 'reference', 'finalize']); p.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'freeze': print(json.dumps({'id': freeze(args.root, args.destination)['id']}))
    else: run(args.directory, args.role, args.workers)


if __name__ == '__main__': main()
