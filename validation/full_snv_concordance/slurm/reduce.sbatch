#!/usr/bin/env bash
#SBATCH --job-name=fsc_reduce
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

FSC_KIND=${1:?missing analysis kind}
FSC_OUTPUT_DIR=${2:?missing output directory}
FSC_PAIRS_FILE=${3:?missing frozen pairs file}
FSC_EXPECTED_TASK_COUNT=${4:?missing expected task count}
FSC_FINALITY=${5:?missing finality}
FSC_EXPECTED_TOTAL_CHUNKS=${6:?missing expected total chunk count}
FSC_EXPECTED_OVERLAP_COUNT=${7-}
FSC_PAIRS_SHA256=${8:?missing pairs SHA-256}
FSC_EXPECTED_CODE_SHA256=${9:?missing code SHA-256}
FSC_PACKAGE_DIR=${10:?missing package directory}
FSC_REPO_ROOT=${11:?missing repository root}
FSC_PYTHON=${12:?missing Python executable}
FSC_SITES_FILE=${13:-}
exec >"$FSC_OUTPUT_DIR/logs/reduce.out" 2>"$FSC_OUTPUT_DIR/logs/reduce.err"
fingerprint_script="$FSC_PACKAGE_DIR/slurm/fingerprint.sh"
[[ -x "$fingerprint_script" ]] || { echo "fingerprint helper missing: $fingerprint_script" >&2; exit 1; }
[[ "$("$fingerprint_script" "$FSC_PAIRS_FILE" "$FSC_PYTHON")" == "$FSC_PAIRS_SHA256" ]] || { echo "pairs checksum changed" >&2; exit 1; }
[[ "$("$fingerprint_script" "$FSC_PACKAGE_DIR" "$FSC_PYTHON")" == "$FSC_EXPECTED_CODE_SHA256" ]] || { echo "analysis code changed after submission" >&2; exit 1; }
cd "$FSC_REPO_ROOT"
if [[ "$FSC_KIND" == "concordance" ]]; then
    command_name="reduce-concordance"
else
    command_name="reduce-seeds"
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
reduce_command=("${clean_environment[@]}" "$FSC_PYTHON" -m validation.full_snv_concordance "$command_name" \
    --input-dir "$FSC_OUTPUT_DIR/maps" \
    --output "$FSC_OUTPUT_DIR/summary.json" \
    --pairs-file "$FSC_PAIRS_FILE" \
    --expected-task-count "$FSC_EXPECTED_TASK_COUNT" \
    --expected-total-chunks "$FSC_EXPECTED_TOTAL_CHUNKS" \
    ${FSC_SITES_FILE:+--sites-file "$FSC_SITES_FILE"} \
    --finality "$FSC_FINALITY" \
    --parquet-dir "$FSC_OUTPUT_DIR/tables")
if [[ "$FSC_KIND" == "seeds" && -n "${FSC_EXPECTED_OVERLAP_COUNT:-}" ]]; then
    reduce_command+=(--expected-overlap-count "$FSC_EXPECTED_OVERLAP_COUNT")
fi
"${reduce_command[@]}"
