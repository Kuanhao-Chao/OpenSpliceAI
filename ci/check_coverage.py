"""Enforce separate statement and branch thresholds from coverage JSON."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--statements', type=float, default=95.)
    parser.add_argument('--branches', type=float, default=90.)
    args = parser.parse_args()
    totals = json.loads(args.report.read_text())['totals']
    statements = 100*totals['covered_lines']/totals['num_statements']
    if not totals.get('num_branches'):
        parser.error('branch coverage is missing; rerun with --cov-branch')
    branches = 100*totals['covered_branches']/totals['num_branches']
    print(f'Statements: {statements:.3f}% (required {args.statements:g}%)')
    print(f'Branches: {branches:.3f}% (required {args.branches:g}%)')
    return int(statements < args.statements or branches < args.branches)


if __name__ == '__main__':
    raise SystemExit(main())
