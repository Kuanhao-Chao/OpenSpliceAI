#!/usr/bin/env bash
#SBATCH --job-name=fsc_map
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
FSC_KIND=${1:?missing analysis kind}
FSC_PAIRS_FILE=${2:?missing frozen pairs file}
FSC_OUTPUT_DIR=${3:?missing output directory}
FSC_RUN_LABEL=${4:?missing run label}
FSC_PAIRS_SHA256=${5:?missing pairs SHA-256}
FSC_EXPECTED_CODE_SHA256=${6:?missing code SHA-256}
FSC_PACKAGE_DIR=${7:?missing package directory}
FSC_REPO_ROOT=${8:?missing repository root}
FSC_PYTHON=${9:?missing Python executable}
# Everything after the fixed nine arguments is passed straight to the mapper, which
# is how the analysis configuration (--score-bins, --thresholds, --sites-file, ...)
# reaches it. It is captured in launch.tsv and re-verified via the code fingerprint.
shift 9
FSC_MAP_ARGS=("$@")

exec >"$FSC_OUTPUT_DIR/logs/map-${SLURM_ARRAY_TASK_ID}.out" 2>"$FSC_OUTPUT_DIR/logs/map-${SLURM_ARRAY_TASK_ID}.err"
[[ "$(/usr/bin/sha256sum "$FSC_PAIRS_FILE" | /usr/bin/awk '{print $1}')" == "$FSC_PAIRS_SHA256" ]] || { echo "pairs checksum changed" >&2; exit 1; }
fingerprint_script="$FSC_PACKAGE_DIR/slurm/fingerprint.sh"
[[ -x "$fingerprint_script" ]] || { echo "fingerprint helper missing: $fingerprint_script" >&2; exit 1; }
[[ "$("$fingerprint_script" "$FSC_PACKAGE_DIR")" == "$FSC_EXPECTED_CODE_SHA256" ]] || { echo "analysis code changed after submission" >&2; exit 1; }
cd "$FSC_REPO_ROOT"
if [[ "$FSC_KIND" == "concordance" ]]; then
    command_name="map-concordance"
else
    command_name="map-seeds"
fi
clean_environment=(
    /usr/bin/env -i
    PATH="$(dirname "$FSC_PYTHON"):/usr/bin:/bin"
    PYTHONPATH="$FSC_REPO_ROOT"
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
)
"${clean_environment[@]}" "$FSC_PYTHON" -m validation.full_snv_concordance "$command_name" \
    --pairs-file "$FSC_PAIRS_FILE" \
    --output "$FSC_OUTPUT_DIR/maps/map-${SLURM_ARRAY_TASK_ID}.json" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --chunks-per-task 250 \
    --run-label "$FSC_RUN_LABEL" \
    "${FSC_MAP_ARGS[@]}"
