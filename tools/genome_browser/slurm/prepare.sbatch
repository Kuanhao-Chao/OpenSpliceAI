#!/bin/bash
#SBATCH --account=ssalzbe1_bigmem
#SBATCH --partition=bigmem
#SBATCH --qos=qos_bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
set -euo pipefail
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OSGB_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1
cd "${OSGB_SNAPSHOT_ROOT:?}/code"
exec "${OSGB_PYTHON:?}" -m tools.genome_browser.jobs run \
  "$OSGB_SNAPSHOT_ROOT" "${OSGB_ROLE:?}" --workers "$OMP_NUM_THREADS"
