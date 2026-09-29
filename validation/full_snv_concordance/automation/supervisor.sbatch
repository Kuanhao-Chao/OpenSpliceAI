#!/usr/bin/env bash
#SBATCH --job-name=osai_campaign_supervisor
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

child_pid=""
handoff() {
    # The state file is atomically checkpointed after every transition.  Ask
    # Slurm to requeue this job before its walltime expires; if the controller
    # is unavailable, leave the checkpoint and let the scheduler's requeue
    # policy or the bootstrap retry path recover it.
    if [[ -n "${SLURM_JOB_ID:-}" && -x /cm/shared/apps/slurm/current/bin/scontrol ]]; then
        /cm/shared/apps/slurm/current/bin/scontrol requeue "$SLURM_JOB_ID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" >/dev/null 2>&1 || true
        wait "$child_pid" >/dev/null 2>&1 || true
    fi
    exit 0
}
trap handoff USR1

state_root=""
repo_root=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-root) state_root=$2; shift 2 ;;
        --repo-root) repo_root=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$state_root" && -n "$repo_root" ]] || { echo "--state-root and --repo-root are required" >&2; exit 2; }
mkdir -p "$state_root/logs"
exec >>"$state_root/logs/supervisor-${SLURM_JOB_ID:-local}.log" 2>&1
cd "$repo_root"
/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python \
    -m validation.full_snv_concordance.automation.supervisor \
    --state-root "$state_root" \
    --repo-root "$repo_root" &
child_pid=$!
wait "$child_pid"
