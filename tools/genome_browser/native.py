"""Streaming native IGV artifacts derived from the lossless OSGB1 store.

One 16kb tile is reduced at a time, including all shard fragments at its
boundaries. Zeros are stored; missing predictions have no BigWig entry.
"""
from __future__ import annotations
import gzip
import itertools
import json
import sqlite3
import struct
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pyBigWig
import pysam
from .build import atomic, file_sha
from .format import EVENTS, canonical_json, sha256
from .reference import Reference, genes_from_tsv


def arrays(raw):
    if raw[:8] != b'OSGB1\0\0\0': raise ValueError('invalid packed block magic')
    size = struct.unpack_from('<I', raw, 8)[0]
    header = json.loads(raw[12:12+size]); payload = memoryview(raw)[12+size:]
    kinds = {'I': '<u4', 'B': 'u1', 'h': '<i2'}
    columns = {}
    for c in header['columns']:
        if c['offset'] + c['bytes'] > len(payload): raise ValueError('truncated score column')
        a = np.frombuffer(payload, dtype=kinds[c['type']], count=c['count'], offset=c['offset'])
        if a.nbytes != c['bytes']: raise ValueError('invalid column size')
        columns[c['name']] = a
    return header, columns


def tiles(output):
    """Ordered tile fragments; use the private preparation index, never all rows."""
    root = Path(output); db = sqlite3.connect(f'file:{root / "preparation.sqlite"}?mode=ro', uri=True)
    files = OrderedDict()
    try:
        cursor = db.execute('SELECT chrom,start,shard,offset,length,sha FROM blocks ORDER BY chrom,start,shard,offset')
        for (chrom, start), group in itertools.groupby(cursor, lambda r: (r[0], r[1])):
            fragments = []
            for _, _, shard, offset, length, digest in group:
                name = f'scores-{shard:04d}.pack'
                handle = files.pop(name, None)
                if handle is None: handle = (root / name).open('rb')
                files[name] = handle
                while len(files) > 8: files.popitem(last=False)[1].close()
                handle.seek(offset); compressed = handle.read(length)
                if sha256(compressed) != digest: raise ValueError(f'block hash differs: {name}:{offset}')
                fragments.append(arrays(gzip.decompress(compressed)))
            yield chrom, start, fragments
    finally:
        for f in files.values(): f.close()
        db.close()


def descriptor(path, root):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}


class Runs:
    """Run length encoding keeps long scored-zero intervals compact."""
    def __init__(self, writer): self.writer, self.pending = writer, None
    def add(self, chrom, positions, values):
        if not len(positions): return
        changes = np.r_[True, (np.diff(positions) != 1) | (values[1:] != values[:-1])]
        starts = np.flatnonzero(changes); ends = np.r_[starts[1:]-1, len(positions)-1]
        begin=positions[starts].astype(np.int64).copy(); finish=positions[ends].astype(np.int64)+1; scores=values[starts].astype(np.float64)
        if self.pending and self.pending[0]==chrom and self.pending[2]==int(begin[0]) and self.pending[3]==float(scores[0]):
            begin[0]=self.pending[1]; self.pending=None
        else: self.flush()
        if len(begin)>1:
            self.writer.addEntries([chrom]*(len(begin)-1),begin[:-1].tolist(),ends=finish[:-1].tolist(),values=scores[:-1].tolist())
        self.pending=(chrom,int(begin[-1]),int(finish[-1]),float(scores[-1]))
    def flush(self):
        if self.pending:
            chrom, a, b, value = self.pending
            self.writer.addEntries([chrom], [a], ends=[b], values=[value])
            self.pending = None


def build_tracks(config_path, output):
    root = Path(output); config = json.loads(Path(config_path).read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    identity = sha256(canonical_json({'config': manifest['configSha256'], 'schema': 2, 'generator': file_sha(__file__)}))
    receipt = root/'native-tracks-receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['identity'] != identity: raise ValueError('native track configuration changed; use a new snapshot')
        if all((root/f['path']).stat().st_size == f['bytes'] and file_sha(root/f['path']) == f['sha256'] for f in old['files']):
            manifest['igv'] = old['igv']; manifest['coverage'] = old['coverage']
            manifest.update({'manifestVersion':2,'publishedIntervals':config.get('regions'),
                'processing':{'r10':{'complete':True},'r13':{'complete':bool(config.get('r13_final'))}},'bookmarks':config.get('bookmarks',[])})
            merge_files(manifest, old['files']); atomic(root/'manifest.json', canonical_json(manifest)); return old
        raise ValueError('native track resume verification failed')
    sizes = sorted((c['name'], c['length']) for c in manifest['contigs'])
    writers, runs, tracks, coverage_tracks = {}, {}, [], []
    for model in ('r10','r13','baseline'):
        for event in EVENTS:
            key = f'{model}-{event}'; path = root/f'{key}.bw.partial'
            bw = pyBigWig.open(str(path), 'w'); bw.addHeader(sizes, maxZooms=10)
            writers[key] = bw; runs[key] = Runs(bw)
            tracks.append({'model': model, 'event': event, 'path': f'{key}.bw', 'aggregation': 'max over represented ALT/gene annotations; REF matches only'})
    for key in ('input','r10','r13'):
        name = f'coverage-{key}'; bw = pyBigWig.open(str(root/f'{name}.bw.partial'), 'w'); bw.addHeader(sizes, maxZooms=10)
        writers[name] = bw; runs[name] = Runs(bw); coverage_tracks.append({'model':key,'path':f'{name}.bw'})
    counts = {'sourceOccurrences':0,'uniqueCanonicalInput':0,'models':{m:{'predictionOccurrences':0,'annotationEntries':0,'uniqueCanonicalPredicted':0} for m in ('r10','r13','baseline')}}
    try:
        for number, (chrom, start, fragments) in enumerate(tiles(root)):
            cols = [c for _,c in fragments]
            pos = np.concatenate([c['pos'] for c in cols]); ref = np.concatenate([c['ref'] for c in cols]); alt = np.concatenate([c['alt'] for c in cols])
            flags = np.concatenate([c['flags'] for c in cols]); canonical = (ref<4)&(alt<4)&(ref!=alt)
            keys = (pos.astype(np.uint64)<<8) | (ref.astype(np.uint64)<<4) | alt
            unique = np.unique(keys[canonical]); counts['sourceOccurrences'] += len(pos); counts['uniqueCanonicalInput'] += len(unique)
            eligible = canonical & ((flags & 1)==0)
            input_keys = np.unique(keys[eligible]); cp, nc = np.unique(input_keys>>8, return_counts=True)
            runs['coverage-input'].add(chrom, cp, nc)
            offsets = np.cumsum([0]+[len(c['pos']) for c in cols[:-1]])
            positions, inverse = np.unique(pos, return_inverse=True)
            for model in ('r10','r13','baseline'):
                rows = np.concatenate([c[model+'.row'].astype(np.int64)+o for c,o in zip(cols,offsets)])
                predicted = np.zeros(len(pos), dtype=bool); predicted[rows] = True
                stat = counts['models'][model]; stat['predictionOccurrences'] += int(predicted.sum()); stat['annotationEntries'] += len(rows)
                stat['uniqueCanonicalPredicted'] += len(np.unique(keys[canonical & predicted]))
                if model != 'baseline':
                    pk = np.unique(keys[eligible & predicted]); pp, n = np.unique(pk>>8, return_counts=True)
                    # Explicit zero predictions where an input allele is present.
                    full = np.zeros(len(cp), dtype=np.float64); full[np.searchsorted(cp, pp)] = n
                    runs['coverage-'+model].add(chrom, cp, full)
                valid = eligible[rows]
                for event in EVENTS:
                    values = np.concatenate([c[model+'.DS_'+event] for c in cols])
                    maxima = np.full(len(positions), -1, dtype=np.int32)
                    np.maximum.at(maxima, inverse[rows[valid]], values[valid].astype(np.int32))
                    present = maxima>=0; runs[f'{model}-{event}'].add(chrom, positions[present], maxima[present]/100000)
            if number % 200 == 0: print(json.dumps({'stage':'native','tile':number,'chrom':chrom,'start':start,'sourceOccurrences':counts['sourceOccurrences']}),flush=True)
        for run in runs.values(): run.flush()
    finally:
        for bw in writers.values(): bw.close()
    expected = config.get('global_census')
    if manifest['scope'] == 'source-collection' and expected:
        if counts['sourceOccurrences'] != expected['sourceOccurrences'] or counts['uniqueCanonicalInput'] != expected['uniqueCanonicalInput']:
            raise ValueError('native conversion input census differs from final audit')
        for m in ('r10','r13'):
            for k in ('uniqueCanonicalPredicted','predictionOccurrences','annotationEntries'):
                if counts['models'][m][k] != expected['models'][m][k]: raise ValueError(f'native {m} {k} differs from final audit')
    files=[]
    for key in writers:
        path=root/f'{key}.bw'; (root/f'{key}.bw.partial').replace(path); files.append(descriptor(path,root))
    for track in tracks+coverage_tracks: track.update(next(f for f in files if f['path']==track['path']))
    annotation = root/'scoring-genes.bed'
    genes = genes_from_tsv(config['annotation'])
    if manifest['scope'] == 'demonstration' and config.get('regions'):
        genes = [g for g in genes if any(r['chrom']==g['chrom'] and r['start']<g['end'] and g['start']<r['end'] for r in config['regions'])]
    order = {c:i for i,(c,_) in enumerate(sizes)}
    with annotation.open('w') as f:
        for g in sorted(genes,key=lambda g:(order.get(g['chrom'],999999),g['start'],g['end'])):
            exons=g['exons']; f.write('\t'.join(map(str,[g['chrom'],g['start'],g['end'],g['name'],0,g['strand'],g['start'],g['start'],0,len(exons),','.join(str(b-a) for a,b in exons)+',',','.join(str(a-g['start']) for a,b in exons)+',']))+'\n')
    pysam.tabix_compress(str(annotation),str(annotation)+'.gz',force=True)
    pysam.tabix_index(str(annotation)+'.gz',preset='bed',force=True); annotation.unlink()
    # .gz is transparently decompressed by some static hosts. Indexed BGZF
    # bytes must retain their original representation, so publish as .bgz.
    (root/'scoring-genes.bed.gz').replace(root/'scoring-genes.bed.bgz')
    (root/'scoring-genes.bed.gz.tbi').replace(root/'scoring-genes.bed.bgz.tbi')
    gene=descriptor(root/'scoring-genes.bed.bgz',root); gi=descriptor(root/'scoring-genes.bed.bgz.tbi',root); files.extend([gene,gi])
    coverage={'published':counts,'global':expected or counts}
    igv={'tracks':tracks,'coverageTracks':coverage_tracks,'genes':{**gene,'index':gi},'reference':config.get('igv_reference',{})}
    result={'identity':identity,'files':files,'coverage':coverage,'igv':igv}
    atomic(receipt,canonical_json(result)); manifest.update({'manifestVersion':2,'igv':igv,'coverage':coverage,
        'publishedIntervals':config.get('regions'), 'processing':{'r10':{'complete':True},'r13':{'complete':bool(config.get('r13_final'))}},
        'bookmarks':config.get('bookmarks',[])})
    merge_files(manifest,files); atomic(root/'manifest.json',canonical_json(manifest)); return result


def merge_files(manifest, files):
    bypath={f['path']:f for f in manifest['files']}
    bypath.update({f['path']:f for f in files}); manifest['files']=list(bypath.values())


def build_native_reference(config_path, output, *, excerpts=False):
    from contextlib import closing
    config=json.loads(Path(config_path).read_text())
    with closing(Reference(config['reference'])) as ref:
        return _write_native_reference(config_path,output,ref,excerpts=excerpts)


def _write_native_reference(config_path, output, ref, *, excerpts=False):
    root=Path(output);config=json.loads(Path(config_path).read_text());files=[]
    if excerpts:
        path=root/'reference-excerpts.fa'; ranges={}
        for r in config['reference_review_regions']: ranges.setdefault(r['chrom'],[]).append((r['start'],r['end']))
        # A real first base provides chromosome metadata for the full assembly;
        # every other unavailable base stays unavailable, never padded with Ns.
        with path.open('w') as f:
            for chrom,(length,*_) in ref.contigs.items():
                intervals=sorted([(0,1)]+ranges.get(chrom,[])); merged=[]
                for a,b in intervals:
                    if merged and a<=merged[-1][1]: merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                    else: merged.append((a,b))
                for a,b in merged:
                    b=min(b,length); f.write(f'>{chrom}:{a+1}-{b} @len={length}\n{ref.read(chrom,a,b)}\n')
        files.append(descriptor(path,root)); native={'kind':'excerpts','fasta':files[-1]}
    else:
        import hashlib
        path=root/'reference.fa.bgz'
        expected=config.get('reference_sha256') or file_sha(config['reference'])
        receipt=root/'native-reference-receipt.json'
        identity=sha256(canonical_json({'reference':expected,'generator':file_sha(__file__)}))
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get('identity') != identity or any(not (root/f['path']).is_file() or file_sha(root/f['path'])!=f['sha256'] for f in old['files']):
                raise ValueError('native reference resume verification failed')
        else:
            if not path.exists():
                digest=hashlib.sha256()
                with Path(config['reference']).open('rb') as src, pysam.BGZFile(str(path)+'.partial','wb') as dst:
                    for data in iter(lambda:src.read(8*1048576),b''): digest.update(data); dst.write(data)
                if digest.hexdigest()!=expected: raise ValueError('native reference source hash differs')
                Path(str(path)+'.partial').replace(path)
            # Existing bytes require proof of decompressed identity, even on resume.
            digest=hashlib.sha256()
            with pysam.BGZFile(str(path),'rb') as source:
                for data in iter(lambda:source.read(8*1048576),b''): digest.update(data)
            if digest.hexdigest()!=expected: raise ValueError('native reference decompressed content differs')
            pysam.faidx(str(path))
            observed={r.split('\t')[0]:int(r.split('\t')[1]) for r in Path(str(path)+'.fai').read_text().splitlines()}
            if observed!={name:values[0] for name,values in ref.contigs.items()}: raise ValueError('native reference contig lengths differ')
            files=[descriptor(Path(str(path)+suffix),root) for suffix in ('','.fai','.gzi')]
            atomic(receipt,canonical_json({'identity':identity,'sourceSha256':expected,'decompressedSha256':digest.hexdigest(),'contigs':len(observed),'files':files}))
        files=[descriptor(Path(str(path)+suffix),root) for suffix in ('','.fai','.gzi')]
        native={'kind':'indexed','fasta':files[0],'index':files[1],'compressedIndex':files[2]}
    manifest=json.loads((root/'manifest.json').read_text()); manifest['igv']['reference']=native
    merge_files(manifest,files); atomic(root/'manifest.json',canonical_json(manifest)); return native
