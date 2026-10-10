"""Measure representative preparation throughput before a full CPU allocation."""
import argparse
import json
import resource
import time
from pathlib import Path

from .build import atomic, prepare
from .format import canonical_json
from .review import make_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/genome_browser/benchmark')
    args = parser.parse_args()
    root = Path(args.output)
    config = make_config('.')
    config.update({'id': 'preparation-benchmark', 'label': 'Preparation benchmark',
                   'scope': 'preparation-benchmark', 'shard_size': 12,
                   'chunks': [1, 2, 1000, 10000, 20000, 40000, 60000, 80000, 90000, 95000, 99999, 100000]})
    config.pop('regions', None)
    config.pop('reference_review_regions', None)
    atomic(root / 'config.json', canonical_json(config))
    start = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    prepare(root / 'config.json', root / 'data')
    after = resource.getrusage(resource.RUSAGE_SELF)
    import sqlite3
    with sqlite3.connect(root / 'data/preparation.sqlite') as db:
        size, rows = db.execute('SELECT sum(bytes),sum(records) FROM shards').fetchone()
    report = {'chunks': 12, 'sourceOccurrences': rows, 'packedBytes': size,
              'wallSeconds': time.monotonic() - start, 'cpuSeconds': after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime,
              'maxRssKiB': after.ru_maxrss, 'classification': 'representative small-sample estimate, not a measured whole-genome total'}
    report['extrapolatedWallHours100000Chunks'] = report['wallSeconds'] / 12 * 100000 / 3600
    report['extrapolatedPackedBytes100000Chunks'] = size / 12 * 100000
    atomic(root / 'report.json', canonical_json(report))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
