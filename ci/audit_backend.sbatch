#!/bin/bash
#SBATCH --job-name=osai_comprehensive_backend
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
# Account/partition/GRES, immutable source, interpreter and output are explicit
# sbatch arguments. This script submits no jobs and changes no campaign settings.
set -euo pipefail
backend="$1"
source_directory="$2"
python_executable="$3"
output_directory="$4"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export OSAI_TF32=0 OSAI_CUDNN_BENCH=0 OSAI_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$source_directory"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/osai-audit-mpl-${SLURM_JOB_ID}"
if [[ "$backend" == keras ]]; then
    export CUDA_VISIBLE_DEVICES='' TF_USE_LEGACY_KERAS=1
fi
cd "$source_directory"
"$python_executable" -B ci/freeze_backend.py --verify
"$python_executable" -B ci/required_backend.py "$backend" --output-dir "$output_directory"
