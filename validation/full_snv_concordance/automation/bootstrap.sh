#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
state_root="${AUTOMATION_STATE_ROOT:-/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI_precompute_scores/results/full_genome_scoring_campaign/automation}"
slurm_bin="${SLURM_BIN:-/cm/shared/apps/slurm/current/bin}"
supervisor_walltime="${SUPERVISOR_WALLTIME:-48:00:00}"
account="${AUTOMATION_ACCOUNT:-ssalzbe1-chess}"
partition="${AUTOMATION_PARTITION:-parallel}"
backoff=30
backoff_cap=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-root) state_root=$2; shift 2 ;;
        --walltime) supervisor_walltime=$2; shift 2 ;;
        --account) account=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--state-root DIR] [--walltime HH:MM:SS] [--account ACCOUNT] [--partition PARTITION]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$state_root/logs"
exec 9>"$state_root/bootstrap.lock"
if ! flock -n 9; then
    echo "another bootstrap process already holds $state_root/bootstrap.lock" >&2
    exit 0
fi
printf '%s\n' "$$" >"$state_root/bootstrap.pid"
trap 'rm -f "$state_root/bootstrap.pid"' EXIT

while :; do
    if "$slurm_bin/scontrol" ping >"$state_root/logs/scontrol-ping.out" 2>"$state_root/logs/scontrol-ping.err"; then
        break
    fi
    printf '[%s] Slurm unavailable; retrying in %ss\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$backoff" | tee -a "$state_root/logs/bootstrap.log"
    sleep "$backoff"
    if (( backoff < backoff_cap )); then
        backoff=$((backoff * 2))
        (( backoff > backoff_cap )) && backoff=$backoff_cap
    fi
done

set +e
while :; do
    set +e
    sbatch_output="$($slurm_bin/sbatch --parsable \
        --job-name=osai_campaign_supervisor \
        --chdir="$repo_root" \
        --cpus-per-task=24 \
        --mem=32G \
        --account="$account" \
        --partition="$partition" \
        --time="$supervisor_walltime" \
        --requeue \
        --signal="B:USR1@600" \
        --export=NONE \
        "$repo_root/validation/full_snv_concordance/automation/supervisor.sbatch" \
        --state-root "$state_root" \
        --repo-root "$repo_root" 2>&1)"
    sbatch_rc=$?
    set -e
    job_id="${sbatch_output%%;*}"
    if (( sbatch_rc == 0 )) && [[ "$job_id" =~ ^[0-9]+$ ]]; then
        break
    fi
    printf '[%s] supervisor submission failed (rc=%s): %s; retrying in %ss\n' \
        "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$sbatch_rc" "$sbatch_output" "$backoff" \
        | tee -a "$state_root/logs/bootstrap.log" >&2
    sleep "$backoff"
    if (( backoff < backoff_cap )); then
        backoff=$((backoff * 2))
        (( backoff > backoff_cap )) && backoff=$backoff_cap
    fi
done
printf '{"job_id":"%s","submitted_at":"%s"}\n' "$job_id" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" >"$state_root/supervisor_job.json"
printf '[%s] supervisor submitted: %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$job_id" | tee -a "$state_root/logs/bootstrap.log"
