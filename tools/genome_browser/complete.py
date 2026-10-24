"""Final audited snapshots, verified adoption and a checkpointed CPU campaign."""
from __future__ import annotations
import argparse
import csv
import gzip
import itertools
import concurrent.futures
import json
import os
import shutil
import sqlite3
import subprocess
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from .build import ManifestLookup, accepted, atomic, file_sha, finalize, init_db, prepare_parallel, vcf_rows
from .format import canonical_json, sha256
from .hosting import verify
from .integrity import decoded_rows, semantic_digest
from .native import build_tracks, build_native_reference
from .provenance import build_provenance
from .reference import Reference
from .review import make_config, review_reference
from .search_index import attach_reference
from .demo_recipe import recipe_settings


def builder_identity(directory=None):
    directory=Path(directory) if directory else Path(__file__).parent
    return sha256(canonical_json({p.name:file_sha(p) for p in sorted(directory.glob('*.py'))}))


def freeze(repo, destination):
    repo=Path(repo).resolve(); destination=Path(destination).resolve()
    if (destination/'config.json').exists(): return json.loads((destination/'config.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    config=make_config(repo)
    for field in ('chunks','regions','reference_review_regions'): config.pop(field,None)
    source=repo/'results/full_snv_concordance/hybrid_finish_20261018_gpu5'
    config['r13_manifest']=str(source/'manifests/audit_final_001/accepted.tsv')
    for key in ('r10_manifest','r13_manifest'):
        target=destination/f'{key}.tsv'; shutil.copyfile(config[key],target); config[key]=str(target)
    for name,original in [('audit.json',source/'state/inventory.json'),('census.json',source/'verification/scored_keys.json')]:
        shutil.copyfile(original,destination/name)
    audit=json.loads((destination/'audit.json').read_text()); census=json.loads((destination/'census.json').read_text())
    if not (audit.get('completion_gate_passed') and audit.get('full_content') and census.get('passed')): raise ValueError('final scientific audit has not passed')
    global_census={'sourceOccurrences':census['results'][0]['output_records'],'uniqueCanonicalInput':census['results'][0]['unique_output_canonical_snvs'],'models':{}}
    for model,result in zip(('r10','r13'),census['results']):
        if result['manifest_sha256'] != file_sha(config[model+'_manifest']): raise ValueError('census is not bound to the frozen manifest')
        lookup=ManifestLookup(config[model+'_manifest'])
        if len(lookup)!=100000 or not all(accepted(lookup.get(i)) for i in range(1,100001)): raise ValueError('incomplete final accepted manifest')
        provenance={}
        for i in lookup:
            state=lookup[i].get('provenance_state','unknown'); provenance[state]=provenance.get(state,0)+1
        lookup.close()
        global_census['models'][model]={'uniqueCanonicalPredicted':result['unique_predicted_canonical_snvs'],
            'predictionOccurrences':result['prediction_records'],'annotationEntries':audit['openspliceai_entries'],
            'missingCanonical':global_census['uniqueCanonicalInput']-result['unique_predicted_canonical_snvs'],'provenance':provenance}
        config['models'][model]['label']=f'OpenSpliceAI MANE {model} · single model'
        config['models'][model]['annotation']='GENCODE v24 canonical scoring table; MANE-trained model'
    config.update({'id':f'grch38-r10-r13-final-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}',
        'label':'GRCh38 r10 / r13 · final audited source collection','scope':'source-collection','shard_size':1000,
        'r13_final':True,'r13_final_audit':{'path':str(destination/'audit.json'),'sha256':file_sha(destination/'audit.json')},
        'r13_distinct_scored_census':{'path':str(destination/'census.json'),'sha256':file_sha(destination/'census.json')},
        'r13_evidence':'Final full-content audit; historical and receipt-bound provenance retained. Processing completion is separate from prediction availability.',
        'global_census':global_census,'snapshot_directory':str(destination/'data'),
        'adopt_from':str(repo/'results/genome_browser/full_20261010_v3/data'),
        'reference_from':str(repo/'results/genome_browser/full_20261010_v3/data'),
        'repository':str(repo),'builder_sha256':builder_identity()})
    for key in ('reference','annotation'):
        if file_sha(config[key])!=config[key+'_sha256']: raise ValueError(f'{key} identity differs')
    code=destination/'code/tools/genome_browser'; shutil.copytree(Path(__file__).parent,code,ignore=shutil.ignore_patterns('__pycache__','tests'),dirs_exist_ok=True)
    atomic(destination/'config.json',canonical_json(config)); return config


def expected_rows(config, chunk_ids):
    a=ManifestLookup(config['r10_manifest']); b=ManifestLookup(config['r13_manifest']); ref=Reference(config['reference']); cached=None; sequence=''
    try:
        for chunk in chunk_ids:
            m10,m13=a[chunk],b[chunk]
            first=vcf_rows(m10['output_path'],chunk,m10['output_file_sha256']); second=vcf_rows(m13['output_path'],chunk,m13['output_file_sha256'])
            sentinel=object(); count=0
            for r,s in itertools.zip_longest(first,second,fillvalue=sentinel):
                if r is sentinel or s is sentinel or any(r[k]!=s[k] for k in ('chrom','pos','ref','alts')): raise ValueError('adoption source alignment differs')
                if r['annotations'].get('SpliceAI',[])!=s['annotations'].get('SpliceAI',[]): raise ValueError('baseline changed between models')
                count+=1; chrom,pos=r['chrom'],r['pos']; key=(chrom,pos//16384*16384)
                if cached!=key: sequence=ref.read(chrom,key[1],min(key[1]+16384,ref.contigs[chrom][0])); cached=key
                flags=2|int(sequence[pos-key[1]]!=r['ref'])
                for alt in r['alts']:
                    yield {'chrom':chrom,'pos':pos,'ref':r['ref'],'alt':alt,'chunk':chunk,'ordinal':r['ordinal'],'flags':flags,
                        'models':{'r10':[e for e in r['annotations'].get('OpenSpliceAI',[]) if e['alt']==alt],
                            'r13':[e for e in s['annotations'].get('OpenSpliceAI',[]) if e['alt']==alt],
                            'baseline':[e for e in r['annotations'].get('SpliceAI',[]) if e['alt']==alt]}}
            if count!=int(m10['output_records']) or count!=int(m13['output_records']): raise ValueError('adoption source count differs')
    finally: a.close(); b.close(); ref.close()


def adopt(config_path, step=1, worker_offset=0):
    config=json.loads(Path(config_path).read_text()); root=Path(config['snapshot_directory']); old=Path(config['adopt_from'])
    if not (old/'preparation.sqlite').exists(): return
    root.mkdir(exist_ok=True); identity=sha256(canonical_json(config)); lock=root/'configuration.sha256'
    if lock.exists() and lock.read_text().strip()!=identity: raise ValueError('snapshot identity changed')
    if not lock.exists(): atomic(lock,(identity+'\n').encode())
    db=init_db(root/'preparation.sqlite'); prior=sqlite3.connect(f'file:{old/"preparation.sqlite"}?mode=ro',uri=True)
    m10=ManifestLookup(config['r10_manifest']); m13=ManifestLookup(config['r13_manifest'])
    try:
        for shard,_,name,digest,size,records,r13_records in prior.execute('SELECT * FROM shards ORDER BY id'):
            if shard % step != worker_offset: continue
            if db.execute('SELECT 1 FROM shards WHERE id=?',(shard,)).fetchone(): continue
            path=old/name
            if path.stat().st_size!=size or file_sha(path)!=digest: raise ValueError('adopted pack hash differs')
            ids=list(range(shard*1000+1,(shard+1)*1000+1)); stream=expected_rows(config,ids); n=0
            with path.open('rb') as handle:
                for row in prior.execute('SELECT * FROM blocks WHERE shard=? ORDER BY offset',(shard,)):
                    _,chrom,_,block_offset,length,raw_bytes,block_sha,_=row
                    handle.seek(block_offset); compressed=handle.read(length)
                    if sha256(compressed)!=block_sha: raise ValueError('adoption block checksum differs')
                    raw=gzip.decompress(compressed)
                    if len(raw)!=raw_bytes: raise ValueError('adoption decoded size differs')
                    header,decoded=decoded_rows(raw); expected=list(itertools.islice(stream,len(decoded)))
                    if len(expected)!=len(decoded) or header['chrom']!=chrom or any(r['chrom']!=chrom for r in expected) or semantic_digest(decoded)!=semantic_digest(expected):
                        raise ValueError(f'adoption semantic verification failed: shard {shard} offset {block_offset}')
                    n+=len(decoded)
            if next(stream,None) is not None or n!=records or r13_records!=records: raise ValueError('adoption source occurrences differ')
            target=root/name
            if target.exists():
                if file_sha(target)!=digest: raise ValueError('existing adoption target differs')
            else: os.link(path,target)
            shard_identity=sha256(canonical_json({'config':identity,'chunks':ids,
                'r10':[m10[i]['output_file_sha256'] for i in ids],'r13':[m13[i]['output_file_sha256'] for i in ids]}))
            with db:
                db.executemany('INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?)',prior.execute('SELECT * FROM blocks WHERE shard=?',(shard,)))
                db.execute('INSERT INTO shards VALUES (?,?,?,?,?,?,?)',(shard,shard_identity,name,digest,size,records,r13_records))
            receipt={'shard':shard,'passed':True,'method':'every decoded source occurrence, DS, DP and gene compared with final audited VCFs','records':n,'sha256':digest}
            atomic(Path(config_path).parent/f'adopt-{shard:04d}.json',canonical_json(receipt)); print(json.dumps(receipt),flush=True)
    finally: db.close(); prior.close(); m10.close(); m13.close()


def adopt_checkpoint(config_path, prior_config_path):
    """Rebind closed, content-verified packs from an authenticated frozen campaign."""
    cfg=json.loads(Path(config_path).read_text()); prior_cfg=json.loads(Path(prior_config_path).read_text())
    code=Path(prior_config_path).parent/'code/tools/genome_browser'
    if not code.is_dir() or builder_identity(code)!=prior_cfg.get('builder_sha256'): raise ValueError('checkpoint frozen builder differs')
    for key in ('reference_sha256','annotation_sha256','models','global_census','chunks','regions','shard_size'):
        if cfg.get(key)!=prior_cfg.get(key): raise ValueError(f'checkpoint scientific configuration differs: {key}')
    for model in ('r10','r13'):
        if file_sha(cfg[model+'_manifest'])!=file_sha(prior_cfg[model+'_manifest']): raise ValueError('checkpoint source manifests differ')
    old=Path(prior_cfg['snapshot_directory']); root=Path(cfg['snapshot_directory']); root.mkdir(parents=True,exist_ok=True)
    prior_identity=sha256(canonical_json(prior_cfg)); identity=sha256(canonical_json(cfg))
    if (old/'configuration.sha256').read_text().strip()!=prior_identity: raise ValueError('checkpoint configuration identity differs')
    lock=root/'configuration.sha256'
    if lock.exists() and lock.read_text().strip()!=identity: raise ValueError('target checkpoint identity differs')
    if not lock.exists(): atomic(lock,(identity+'\n').encode())
    source=sqlite3.connect(f'file:{old/"preparation.sqlite"}?mode=ro',uri=True); db=init_db(root/'preparation.sqlite')
    lookups={m:ManifestLookup(cfg[m+'_manifest']) for m in ('r10','r13')}
    ids=cfg.get('chunks') or list(lookups['r10']); size=cfg.get('shard_size',1000); imported=0
    try:
        for shard,old_identity,name,digest,bytes_,records,r13_records in source.execute('SELECT * FROM shards ORDER BY id'):
            chunk_ids=ids[shard*size:(shard+1)*size]
            bindings={'chunks':chunk_ids,'r10':[lookups['r10'][i]['output_file_sha256'] for i in chunk_ids],'r13':[lookups['r13'][i]['output_file_sha256'] for i in chunk_ids]}
            if old_identity!=sha256(canonical_json({'config':prior_identity,**bindings})): raise ValueError('checkpoint shard source identity differs')
            existing=db.execute('SELECT sha FROM shards WHERE id=?',(shard,)).fetchone()
            if existing:
                if existing[0]!=digest: raise ValueError('target checkpoint content differs')
                continue
            path=old/name
            if path.stat().st_size!=bytes_ or file_sha(path)!=digest: raise ValueError('checkpoint pack content differs')
            target=root/name
            if target.exists():
                if file_sha(target)!=digest: raise ValueError('target pack content differs')
            else: os.link(path,target)
            with db:
                db.executemany('INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?)',source.execute('SELECT * FROM blocks WHERE shard=?',(shard,)))
                db.execute('INSERT INTO shards VALUES (?,?,?,?,?,?,?)',(shard,sha256(canonical_json({'config':identity,**bindings})),name,digest,bytes_,records,r13_records))
            imported+=1
            print(json.dumps({'stage':'checkpoint-adoption','shard':shard,'sourceOccurrences':records,'sha256':digest}),flush=True)
    finally:
        db.close();source.close()
        for lookup in lookups.values():lookup.close()
    return imported


def run(directory,workers=2):
    root=Path(directory).resolve(); cfg=root/'config.json'; config=json.loads(cfg.read_text()); output=Path(config['snapshot_directory']); start=time.monotonic()
    if config.get('builder_sha256') != builder_identity(): raise ValueError('frozen builder identity changed')
    from .jobs import parse_shared_quota
    atomic(root/'status.json',canonical_json({'state':'running','dataset':config['id'],'workers':workers,'gpus':0}))
    quota=parse_shared_quota(subprocess.check_output(['/usr/lpp/mmfs/bin/mmlsquota','-g','ssalzbe1','--block-size','auto','data'],text=True))
    if quota['freeBytes']<100*1024**3 or quota['freeFiles']<15000: raise ValueError('browser preparation safety reserve unavailable')
    print(json.dumps({'stage':'preflight','quota':quota,'workers':workers,'gpus':0}),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'configuration.sha256'
    identity=sha256(canonical_json(config))
    if lock.exists() and lock.read_text().strip()!=identity: raise ValueError('snapshot identity changed')
    if not lock.exists(): atomic(lock,(identity+'\n').encode())
    init_db(output/'preparation.sqlite').close()
    if config.get('adopt_checkpoint'):
        adopt_checkpoint(cfg,config['adopt_checkpoint'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(adopt,cfg,workers,i) for i in range(workers)]: future.result()
    prepare_parallel(cfg,output,workers=workers)
    if not (root/'all-data-verified.json').exists():
        finalize(cfg,output); build_tracks(cfg,output)
        old=Path(config['reference_from']); manifest=json.loads((output/'manifest.json').read_text())
        for receipt_path in sorted(old.glob('reference-*.receipt.json')):
            receipt=json.loads(receipt_path.read_text())
            if receipt['sourceIdentity']!=manifest['referenceSha256']: raise ValueError('reference source identity differs')
            for entry in receipt.get('files',[]):
                path=old/entry['path']
                if file_sha(path)!=entry['sha256']: raise ValueError('reference adoption checksum differs')
                if not (output/path.name).exists(): os.link(path,output/path.name)
            if not (output/receipt_path.name).exists(): shutil.copyfile(receipt_path,output/receipt_path.name)
        # Older reference receipts use one object per artifact. Attach only
        # descriptors authenticated by the reference builder below.
        from .search_index import build_reference
        build_reference(cfg,output,with_search=True)
        attach_reference(output); build_native_reference(cfg,output)
        build_provenance(cfg,output)
        proof=verify(output)
        manifest=json.loads((output/'manifest.json').read_text());manifest['releaseVerified']=True
        atomic(output/'manifest.json',canonical_json(manifest))
        proof['manifestSha256']=file_sha(output/'manifest.json'); proof['elapsedSeconds']=time.monotonic()-start
        atomic(root/'all-data-verified.json',canonical_json(proof))
    atomic(root/'status.json',canonical_json({'state':'complete','dataset':config['id'],'verification':str(root/'all-data-verified.json')}))


def demo(config_path, output, recipe_path=None):
    parent=json.loads(Path(config_path).read_text()); config=dict(parent); original=make_config(parent['repository'])
    config.update({k:original[k] for k in ('chunks','regions','reference_review_regions','default_view')})
    config.update({'id':parent['id']+'-demo-v3','label':'GRCh38 r10 / r13 · demonstration subset','scope':'demonstration','shard_size':2})
    # Derive two additional windows from audited records outside the four
    # existing examples. Both choices are deterministic and REF-matching.
    strong=None; different=None
    for r in (() if recipe_path else expected_rows(parent,config['chunks'])):
        if r['flags']&1 or r['ref'] not in 'ACGT' or r['alt'] not in 'ACGT': continue
        if any(a['chrom']==r['chrom'] and a['start']<=r['pos']<a['end'] for a in config['regions']): continue
        bymodel={m:{e['gene']:e for e in r['models'][m] if sum(e['gene']==x['gene'] for x in r['models'][m])==1} for m in ('r10','r13')}
        key=(r['chrom'],r['pos'],r['ref'],r['alt'])
        for gene in sorted(bymodel['r10'].keys() & bymodel['r13'].keys()):
            a,b=bymodel['r10'][gene],bymodel['r13'][gene]
            score=max(a['ds']+b['ds']); delta=max(abs(x-y) for x,y in zip(a['ds'],b['ds']))
            if strong is None or score>strong[0]: strong=(score,key)
            if different is None or delta>different[0]: different=(delta,key)
    for label,candidate in [('Strong delta score',strong),('Matched seed difference',different)]:
        if candidate:
            chrom,pos,_,_=candidate[1]; region={'chrom':chrom,'start':max(0,pos-512),'end':pos+513}
            if not any(a['chrom']==chrom and a['start']<region['end'] and a['end']>region['start'] for a in config['regions']): config['regions'].append(region)
    for r in config['regions']:
        a=r['start']//16384*16384; b=(r['end']+16383)//16384*16384
        if not any(x['chrom']==r['chrom'] and x['start']<=a and x['end']>=b for x in config['reference_review_regions']): config['reference_review_regions'].append({'chrom':r['chrom'],'start':a,'end':b})
    config['reference_review_regions'].append({'chrom':'chrM','start':0,'end':16569})
    config['bookmarks']=[{**r,'label':f'Example {i+1} · {r["chrom"]}:{r["start"]+1:,}–{r["end"]:,}'} for i,r in enumerate(config['regions'])]
    if recipe_path:
        if not parent.get('r13_final'):
            raise ValueError('expanded demonstration requires final audited source models')
        recipe=json.loads(Path(recipe_path).read_text())
        with closing(Reference(parent['reference'])) as ref:
            config.update(recipe_settings(recipe,{name:entry[0] for name,entry in ref.contigs.items()}))
        config.update({'id':parent['id']+'-demo-'+recipe['name'],
                       'demo_recipe_sha256':file_sha(recipe_path),'builder_sha256':builder_identity()})
        for model in ('r10','r13'):
            lookup=ManifestLookup(config[model+'_manifest'])
            try:
                if not all(accepted(lookup.get(i)) for i in config['chunks']):
                    raise ValueError(f'demo selection lacks accepted {model} audit evidence')
            finally: lookup.close()
    config['snapshot_directory']=str(Path(output).resolve()); target=Path(config_path).parent/'demo-config.json'; atomic(target,canonical_json(config))
    prepare_parallel(target,output,workers=1); finalize(target,output); review_reference(config,Path(output)); build_tracks(target,output)
    from .search_index import build_reference
    build_reference(target,output,with_search=True,contig_names=['chrM']); attach_reference(output)
    build_native_reference(target,output,excerpts=True)
    build_provenance(target,output)
    manifest=json.loads((Path(output)/'manifest.json').read_text())
    expected=config.get('expected_source_occurrences')
    if expected is not None and manifest['sourceOccurrences']!=expected:
        raise ValueError('demo source census differs from its frozen recipe')
    proof=verify(output)
    if proof['bytes']>20*1048576: raise ValueError('demo exceeds 20 MiB data budget')
    atomic(Path(output)/'verification.json',canonical_json(proof)); return proof


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('freeze'); a.add_argument('repository'); a.add_argument('destination')
    a=s.add_parser('run'); a.add_argument('directory'); a.add_argument('--workers',type=int,default=2)
    a=s.add_parser('demo'); a.add_argument('config'); a.add_argument('output'); a.add_argument('--recipe',help='JSON selection of audited chunks and labelled 0-based intervals')
    args=p.parse_args()
    if args.action=='freeze': print(json.dumps({'id':freeze(args.repository,args.destination)['id']}))
    elif args.action=='run': run(args.directory,args.workers)
    else: print(json.dumps(demo(args.config,args.output,args.recipe)))

if __name__=='__main__': main()
