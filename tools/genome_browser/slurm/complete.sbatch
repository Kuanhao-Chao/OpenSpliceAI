#!/usr/bin/env bash
#SBATCH --job-name=osai-browser-final
#SBATCH --partition=parallel
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${BROWSER_CAMPAIGN:?Set BROWSER_CAMPAIGN to the frozen preparation directory}"
: "${BROWSER_PYTHON:?Set BROWSER_PYTHON to the isolated Python executable}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
cd "$BROWSER_CAMPAIGN/code"
trap '/cm/shared/apps/slurm/current/bin/scontrol requeue "$SLURM_JOB_ID"' USR1
"$BROWSER_PYTHON" -u -m tools.genome_browser.complete run "$BROWSER_CAMPAIGN" --workers "${BROWSER_WORKERS:-4}" &
browser_pid=$!
wait "$browser_pid"
