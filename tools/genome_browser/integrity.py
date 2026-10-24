"""Semantic round-trip checks independent of compression and gene dictionaries."""
from .format import BASES, EVENTS, canonical_json, decode_block, sha256


def decoded_rows(raw):
    header, cols = decode_block(raw)
    rows = [{**{k: cols[k][i] for k in ('pos', 'chunk', 'ordinal', 'flags')},
             'ref': BASES[cols['ref'][i]], 'alt': BASES[cols['alt'][i]],
             'models': {m: [] for m in ('r10', 'r13', 'baseline')}} for i in range(header['rows'])]
    for model in ('r10', 'r13', 'baseline'):
        for j, i in enumerate(cols[model + '.row']):
            rows[i]['models'][model].append({'gene': header['genes'][cols[model + '.gene'][j]],
                'ds': [cols[model + '.DS_' + e][j] for e in EVENTS],
                'dp': [cols[model + '.DP_' + e][j] for e in EVENTS]})
    return header, rows


def semantic_digest(rows):
    return sha256(canonical_json([{**{k: r[k] for k in ('pos','ref','alt','chunk','ordinal','flags')},
        'models': {m: [{k: e[k] for k in ('gene','ds','dp')} for e in r['models'].get(m, [])]
                   for m in ('r10','r13','baseline')}} for r in rows]))


def verify_encoding(raw, expected):
    _, rows = decoded_rows(raw)
    if semantic_digest(rows) != semantic_digest(expected):
        raise ValueError('lossless score/offset/source identity round-trip failed')
