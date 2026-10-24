"""Read-only campaign progress and Slurm accounting; never submit or alter jobs."""
from __future__ import annotations
import csv
import json
import math
import sqlite3
import subprocess
from datetime import datetime,timezone
from pathlib import Path


def parse_accounting(text):
    fields=('jobId','state','elapsedSeconds','allocatedCpus','requestedMemory','allocatedTRES','account','partition')
    rows=[dict(zip(fields,line.strip().split('|'))) for line in text.splitlines() if line.strip()]
    if not rows:return None
    row=rows[0]
    for key in ('elapsedSeconds','allocatedCpus'):row[key]=int(row[key] or 0)
    tres=dict(part.split('=',1) for part in row['allocatedTRES'].split(',') if '=' in part)
    row['billedUnitHours']=round(row['elapsedSeconds']*float(tres.get('billing',row['allocatedCpus']))/3600,3)
    row['allocatedGpus']=int(tres.get('gres/gpu',0));return row


def campaign_status(directory,job=None):
    root=Path(directory);config=json.loads((root/'config.json').read_text());data=Path(config['snapshot_directory'])
    def marker(name):
        path=root/name
        if not path.exists():return False
        value=json.loads(path.read_text());proof=value.get('verification',value)
        return bool(proof.get('passed') and proof.get('dataset')==config['id'])
    result={'checkedAtUTC':datetime.now(timezone.utc).isoformat(),'dataset':config['id'],'sourceScoringComplete':bool(config.get('r13_final')),'releaseReady':marker('review-ready.json'),'campaignVerification':marker('all-data-verified.json')}
    chunks=config.get('chunks')
    if chunks is None:
        with Path(config['r10_manifest']).open() as f:count=sum(1 for _ in csv.DictReader(f,delimiter='\t'))
    else:count=len(chunks)
    expected=math.ceil(count/config.get('shard_size',1000));completed=records=bytes_=0
    dbpath=data/'preparation.sqlite'
    if dbpath.exists():
        with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
            completed,records,bytes_=db.execute('SELECT count(*),coalesce(sum(records),0),coalesce(sum(bytes),0) FROM shards').fetchone()
    total=config.get('global_census',{}).get('sourceOccurrences')
    result['scorePacks']={'completed':completed,'expected':expected,'sourceOccurrences':records,'expectedSourceOccurrences':total,'storedBytes':bytes_,'percentOfPacks':round(100*completed/expected,3) if expected else None}
    result['phase']='ready-for-local-review' if result['releaseReady'] else 'campaign-verified' if result['campaignVerification'] else 'native-reference-or-verification' if completed==expected else 'score-pack-conversion'
    result['etaNote']='Pack conversion, native tracks, reference and final verification have different costs; pack percentage is not total-runtime percentage.'
    if job:
        try:
            raw=subprocess.check_output(['/cm/shared/apps/slurm/current/bin/sacct','-n','-X','-P','-j',str(job),'--format=JobID,State,ElapsedRaw,AllocCPUS,ReqMem,AllocTRES,Account,Partition'],text=True,stderr=subprocess.PIPE,timeout=20)
            result['job']=parse_accounting(raw)
        except (subprocess.SubprocessError,OSError) as error:result['accountingUnavailable']=str(error)
    return result
