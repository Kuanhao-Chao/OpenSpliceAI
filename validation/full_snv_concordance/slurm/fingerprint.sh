#!/usr/bin/env bash
set -euo pipefail

# Python provides the same locale-independent GNU SHA-256 manifest digest on
# Linux and macOS. Supply the selected interpreter in --export=NONE jobs.
target="${1:?file or package directory is required}"
python_bin="${2:-python3}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$python_bin" "$script_dir/fingerprint.py" "$target"
