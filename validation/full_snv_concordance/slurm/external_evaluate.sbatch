#!/usr/bin/env bash
#SBATCH --job-name=extval_eval
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=parallel
#SBATCH --account=ssalzbe1-chess

set -euo pipefail

run_dir=${1:?the frozen external-scoring run directory is required}
bootstrap_replicates=${2:?bootstrap replicate count is required}
expected_bundle_sha256=${3:?the externally anchored bundle SHA-256 is required}
[[ "${bootstrap_replicates}" =~ ^[0-9]+$ ]] || {
    echo "bootstrap replicate count must be a nonnegative integer" >&2
    exit 2
}

readonly python_bin=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly runner="${script_dir%/slurm}/external_score_plan.py"

clean_environment=(
    /usr/bin/env -i
    PATH=/home/kchao10/miniconda3/envs/pytorch_cuda/bin:/usr/bin:/bin
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
    SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
)

exec "${clean_environment[@]}" "${python_bin}" "${runner}" run-evaluation \
    --run-dir "${run_dir}" \
    --bootstrap-replicates "${bootstrap_replicates}" \
    --expected-bundle-sha256 "${expected_bundle_sha256}"
