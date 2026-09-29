#!/usr/bin/env bash
#SBATCH --job-name=fsc_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

FSC_OUTPUT_DIR=${1:?missing output directory}
FSC_EXPECTED_CODE_SHA256=${2:?missing code SHA-256}
FSC_PACKAGE_DIR=${3:?missing package directory}
FSC_REPO_ROOT=${4:?missing repository root}
FSC_PYTHON=${5:?missing Python executable}
exec >"$FSC_OUTPUT_DIR/logs/render.out" 2>"$FSC_OUTPUT_DIR/logs/render.err"
MPLCONFIGDIR="$FSC_OUTPUT_DIR/.cache/matplotlib"
/usr/bin/mkdir -p "$MPLCONFIGDIR"
fingerprint_script="$FSC_PACKAGE_DIR/slurm/fingerprint.sh"
[[ -x "$fingerprint_script" ]] || { echo "fingerprint helper missing: $fingerprint_script" >&2; exit 1; }
[[ "$("$fingerprint_script" "$FSC_PACKAGE_DIR")" == "$FSC_EXPECTED_CODE_SHA256" ]] || { echo "analysis code changed after submission" >&2; exit 1; }
cd "$FSC_REPO_ROOT"
clean_environment=(
    /usr/bin/env -i
    PATH="$(dirname "$FSC_PYTHON"):/usr/bin:/bin"
    PYTHONPATH="$FSC_REPO_ROOT"
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    MPLCONFIGDIR="$MPLCONFIGDIR"
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
)
"${clean_environment[@]}" "$FSC_PYTHON" -m validation.full_snv_concordance render-report \
    --summary "$FSC_OUTPUT_DIR/summary.json" \
    --output-dir "$FSC_OUTPUT_DIR/report"
