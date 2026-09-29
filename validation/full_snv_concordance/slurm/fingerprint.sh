#!/usr/bin/env bash
set -euo pipefail

# The digest must not depend on the caller's environment: `sort` collation is
# locale-sensitive, so an interactive submitter and an --export=NONE batch job
# would otherwise hash the same tree to different values.
export LC_ALL=C

package_dir="${1:?package directory is required}"
/usr/bin/find "$package_dir" -type f ! -path '*/__pycache__/*' -print0 \
    | /usr/bin/sort -z \
    | /usr/bin/xargs -0 /usr/bin/sha256sum \
    | /usr/bin/sha256sum \
    | /usr/bin/awk '{print $1}'
