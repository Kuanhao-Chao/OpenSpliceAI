"""Validate a reproducible, small demonstration selection before conversion."""
from __future__ import annotations

import re


def recipe_settings(recipe, contigs):
    name = recipe.get('name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name):
        raise ValueError('demo recipe needs a safe immutable name')
    chunks = recipe.get('chunks')
    if not isinstance(chunks, list) or not chunks or any(type(i) is not int or i < 1 for i in chunks):
        raise ValueError('demo chunks must be positive integers')
    if chunks != sorted(set(chunks)):
        raise ValueError('demo chunks must be unique and in source order')

    def interval(value):
        if not isinstance(value, dict):
            raise ValueError('demo interval must be an object')
        chrom, start, end = (value.get(k) for k in ('chrom', 'start', 'end'))
        if not isinstance(chrom, str) or chrom not in contigs:
            raise ValueError('demo interval has an unknown reference contig')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= contigs[chrom]:
            raise ValueError('demo interval must use valid 0-based half-open coordinates')
        return {'chrom': chrom, 'start': start, 'end': end}

    entries = recipe.get('regions')
    if not isinstance(entries, list) or not entries:
        raise ValueError('demo recipe needs scored regions')
    regions, bookmarks, seen = [], [], set()
    for entry in entries:
        view = interval(entry)
        key = tuple(view.values())
        if key in seen:
            raise ValueError('duplicate demo interval')
        seen.add(key)
        label = entry.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('demo regions need descriptive bookmark labels')
        regions.append(view)
        bookmarks.append({**view, 'label': label.strip()})
    default = interval(recipe.get('default_view'))
    if not any(r['chrom'] == default['chrom'] and r['start'] <= default['start'] and r['end'] >= default['end'] for r in regions):
        raise ValueError('demo default view must be inside a published interval')
    expected = recipe.get('expected_source_occurrences')
    if expected is not None and (type(expected) is not int or expected < 1):
        raise ValueError('expected demo source occurrences must be a positive integer')

    ranges = {}
    for r in regions:
        start = r['start'] // 16384 * 16384
        end = min(contigs[r['chrom']], (r['end'] + 16383) // 16384 * 16384)
        ranges.setdefault(r['chrom'], []).append((start, end))
    if 'chrM' in contigs:
        ranges.setdefault('chrM', []).append((0, contigs['chrM']))
    reference = []
    for chrom, intervals in sorted(ranges.items()):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        reference.extend({'chrom': chrom, 'start': start, 'end': end} for start, end in merged)
    return {'chunks': list(chunks), 'regions': regions, 'bookmarks': bookmarks,
            'default_view': default, 'reference_review_regions': reference,
            'expected_source_occurrences': expected}
