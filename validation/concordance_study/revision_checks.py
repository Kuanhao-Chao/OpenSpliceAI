"""Independent arithmetic checks and a source ledger for the report revision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


from . import report
from .concise_figures import digest
from .concise_release import TEMPLATES, template_context
from .concise_supplement import read_table, write_table
from .loading import EVENTS


def check_close(actual, expected, name):
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(name)
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f'{name}: {actual} != {expected}')


def validate(facts, publication):
    primary = facts['primary']
    n = primary['coverage']['paired_annotations']
    tables = publication/'tables/A_rs10_genomewide'
    checks = []
    def check(actual, expected, name):
        check_close(actual, expected, name)
        checks.append(dict(check=name,actual=actual,expected=expected,status='passed'))
    thresholds = read_table(tables/'thresholds.csv')
    for r in thresholds:
        e,t=r['label'],r['threshold']
        both,left,right,negative = (int(r[k]) for k in ('both_positive','left_only','right_only','both_negative'))
        check(both+left+right+negative,n,f'{e}/{t}: denominator')
        check(r['jaccard'],both/(both+left+right),f'{e}/{t}: Jaccard')
        check(r['call_rate_ratio_right_over_left'],(both+right)/(both+left),f'{e}/{t}: call ratio')
        for key in ('jaccard','call_rate_ratio_right_over_left'):
            check(r[key],primary['thresholds'][e][t][key],f'{e}/{t}: facts {key}')
    for r in read_table(tables/'agreement.csv'):
        e=r['label']
        check(r['n'],n,f'{e}: score denominator')
        check(r['bias'],float(r['mean_right'])-float(r['mean_left']),f'{e}: signed mean difference')
        for key in ('pearson_r','mae','bias'):
            check(r[key],primary['agreement'][e][key],f'{e}: facts {key}')
    site = read_table(tables/'site_dp.csv')
    for r in site:
        count,within=int(r['eligible']),int(r['within'])
        if not 0 <= within <= count:
            raise ValueError('invalid context position counts')
        check(float(r['rate']) if r['rate'] else None,within/count if count else None,
              '/'.join(r[k] for k in ('event','threshold','distance','site_type','tolerance_bp')))
    for e in EVENTS:
        for t,row in primary['dp'][e].items():
            for tolerance in (0,1,2,5,10):
                key=f'within_{tolerance}bp'
                check(row[key],row[key+'_n']/row['eligible'],f'{e}/{t}: {key} exact fraction')
                groups=[r for r in site if r['event']==e and float(r['threshold'])==float(t) and int(r['tolerance_bp'])==tolerance]
                check(sum(int(r['eligible']) for r in groups),row['eligible'],f'{e}/{t}/{tolerance}: context eligible sum')
                check(sum(int(r['within']) for r in groups),row[key+'_n'],f'{e}/{t}/{tolerance}: context within sum')
    for r in read_table(tables/'quantization.csv'):
        check(r['share_of_mae_beyond_quantization'],float(r['quantization_adjusted_mae'])/float(r['raw_mae']),r['label']+': residual-error ratio')
    matrix=primary['dominant']['matrix']
    labels=[r['left_dominant'] for r in matrix]
    check(sum(r['right_'+e] for r in matrix for e in labels),n,'dominant matrix denominator')
    for r in matrix:
        total=sum(r['right_'+e] for e in labels)
        for e in labels:
            check(primary['dominant']['row_normalized'][r['left_dominant']][e],r['right_'+e]/total,'dominant '+r['left_dominant']+'/'+e)
    # Per-gene ratios: verify arithmetic before any eligibility filter.
    seed_rows=read_table(publication/'tables/seed_versus_method_strata.csv')
    for r in seed_rows:
        if r['mae_ratio']:
            check(r['mae_ratio'],float(r['method_mae'])/float(r['seed_mae']),f"seed {r['dimension']}/{r['stratum']}/{r['event']}/{r['model_arm']}")
    for e in EVENTS:
        event=facts['seed_versus_model']['events'][e]
        for arm in ('C_rs10_matched','D_rs13_matched'):
            check(event['ratios'][arm]['mae'],event['models'][arm]['mae']/event['seed']['mae'],f'{e}/{arm}: global MAE ratio')
    for arm in ('C_rs10_matched','D_rs13_matched'):
        check(facts['arms'][arm]['coverage']['paired_annotations'],facts['arms']['B_seeds_rs10_rs13']['coverage']['paired_annotations'],arm+': common population')
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--revision',type=Path,required=True)
    args=parser.parse_args(argv)
    frozen=args.study/'publication'
    facts=json.loads((frozen/'study_facts.json').read_text())
    checks=validate(facts,frozen)
    out=args.revision/'verification'
    out.mkdir(parents=True,exist_ok=True)
    write_table(out/'numerical-checks.csv',checks)
    snapshot=json.loads((args.revision/'progress_snapshot.json').read_text())
    context=template_context(facts,snapshot,'20260910-r2','2026-09-10')
    ledger=[]
    for slug in ('full-snv-scoring-technical-report','full-snv-scoring-supplement','openspliceai-technical-report'):
        text=(TEMPLATES/(slug+'.mdx')).read_text()
        report.resolve(text,context)
        for match in report.PLACEHOLDER.finditer(text):
            path,spec=match.group(1),match.group(2)
            value=report.resolve(match.group(0),context)
            ledger.append(dict(document=slug,line=text[:match.start()].count('\n')+1,source=path,format=spec or '',display=value))
    write_table(out/'claim-sources.csv',ledger)
    # Export exact exception context even if source-record investigation is unresolved.
    rows=read_table(frozen/'tables/A_rs10_genomewide/site_dp.csv')
    exceptions=[r for r in rows if r['event'] in ('AL','DL') and float(r['threshold'])==.5 and int(r['tolerance_bp'])==0 and int(r['within'])<int(r['eligible'])]
    write_table(out/'position-exception-context.csv',exceptions)
    result=dict(status='passed',checks=len(checks),claim_sources=len(ledger),
                frozen_facts_sha256=digest(frozen/'study_facts.json'),
                scope='Arithmetic and cross-table consistency of retained aggregate evidence; no experimental accuracy inference')
    (out/'scientific-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
