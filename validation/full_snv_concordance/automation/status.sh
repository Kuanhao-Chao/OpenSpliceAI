#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec /home/kchao10/miniconda3/envs/pytorch_cuda/bin/python \
    -m validation.full_snv_concordance.automation.supervisor \
    --repo-root "$repo_root" --show-state "$@"
