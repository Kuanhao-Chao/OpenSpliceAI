#!/usr/bin/env bash
#SBATCH --job-name=extval_score
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,ica100
#SBATCH --account=ssalzbe1_gpu

set -euo pipefail

run_dir=${1:?the frozen external-scoring run directory is required}
expected_bundle_sha256=${2:?the externally anchored bundle SHA-256 is required}
: "${SLURM_ARRAY_TASK_ID:?this worker must run as a Slurm array}"

readonly python_bin=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly runner="${script_dir%/slurm}/external_score_plan.py"

clean_environment=(
    /usr/bin/env -i
    PATH=/home/kchao10/miniconda3/envs/pytorch_cuda/bin:/usr/bin:/bin
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
    SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
    SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
)
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || clean_environment+=(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
)
[[ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]] || clean_environment+=(
    NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
)

exec "${clean_environment[@]}" "${python_bin}" "${runner}" run-task \
    --run-dir "${run_dir}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --expected-bundle-sha256 "${expected_bundle_sha256}"
