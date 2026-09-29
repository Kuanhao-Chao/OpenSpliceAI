#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_external_scoring.sh --plan FILE --run-dir NEW_DIR [OPTIONS]

Validate and fingerprint an external-validation scoring plan, then print the
exact Slurm submission commands.  This is a dry run unless --submit is given.

Options:
  --submit                    Freeze inputs and submit the jobs.
  --no-evaluate               Do not submit the dependent CPU evaluation job.
  --bootstrap-replicates N    Evaluation bootstrap replicates (default: 1000).
  -h, --help                  Show this help.
EOF
}

plan=""
run_dir=""
submit=0
evaluate=1
bootstrap_replicates=1000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) plan=$2; shift 2 ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --submit) submit=1; shift ;;
        --no-evaluate) evaluate=0; shift ;;
        --bootstrap-replicates) bootstrap_replicates=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${plan}" && -f "${plan}" ]] || { echo "--plan must name an existing file" >&2; exit 2; }
[[ -n "${run_dir}" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ "${bootstrap_replicates}" =~ ^[0-9]+$ ]] || {
    echo "--bootstrap-replicates must be a nonnegative integer" >&2
    exit 2
}

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(realpath "${script_dir}/../../..")"
readonly runner="${script_dir%/slurm}/external_score_plan.py"
readonly python_bin=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python
readonly spliceai_bin=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/spliceai
readonly openspliceai_bin=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/openspliceai
readonly official_package_dir=/home/kchao10/miniconda3/envs/pytorch_cuda/lib/python3.9/site-packages/spliceai
readonly sbatch_bin=/cm/shared/apps/slurm/current/bin/sbatch
readonly scontrol_bin=/cm/shared/apps/slurm/current/bin/scontrol
readonly scancel_bin=/cm/shared/apps/slurm/current/bin/scancel

plan="$(realpath "${plan}")"
run_dir="$(realpath -m "${run_dir}")"
common=(
    --plan "${plan}"
    --run-dir "${run_dir}"
    --repo-root "${repo_root}"
    --python-bin "${python_bin}"
    --spliceai-bin "${spliceai_bin}"
    --openspliceai-bin "${openspliceai_bin}"
    --official-package-dir "${official_package_dir}"
)

score_command=(
    "${sbatch_bin}" --parsable --hold --export=NONE
    --chdir="${repo_root}"
    --array=0-6%2
    --account=ssalzbe1_gpu
    --partition=a100,ica100
    --time=06:00:00
    --gres=gpu:1
    --cpus-per-task=8
    --mem=32G
    --job-name=extval_score
    --output="${run_dir}/logs/%x_%A_%a.log"
    "${script_dir}/external_score_array.sbatch" "${run_dir}" '<prepared-bundle-sha256>'
)
evaluation_command=(
    "${sbatch_bin}" --parsable --export=NONE
    --chdir="${repo_root}"
    --account=ssalzbe1-chess
    --partition=parallel
    --time=12:00:00
    --cpus-per-task=8
    --mem=32G
    --job-name=extval_eval
    --output="${run_dir}/logs/%x_%j.log"
)
evaluation_script=(
    "${script_dir}/external_evaluate.sbatch" "${run_dir}" "${bootstrap_replicates}"
    '<prepared-bundle-sha256>'
)

if [[ ${submit} -eq 0 ]]; then
    "${python_bin}" "${runner}" inspect "${common[@]}"
    echo "DRY RUN: no files or jobs were created. Re-run with --submit to proceed."
    printf 'score array: '
    printf '%q ' "${score_command[@]}"
    printf '\n'
    if [[ ${evaluate} -eq 1 ]]; then
        printf 'evaluation (dependency inserted after score submission): '
        printf '%q ' "${evaluation_command[@]}" '--dependency=afterok:${score_job}' "${evaluation_script[@]}"
        printf '\n'
    fi
    exit 0
fi

score_job=""
evaluation_job=""
committed=0
submitted_jobs=()
registered_job=""
register_job_output() {
    local raw_output=$1
    local job_kind=$2
    local candidate=${raw_output%%;*}
    if [[ ${candidate} =~ ^[0-9]+$ ]]; then
        submitted_jobs+=("${candidate}")
    fi
    if [[ ! ${candidate} =~ ^[0-9]+$ ]] || \
       [[ ${raw_output} != "${candidate}" && ! ${raw_output} =~ ^${candidate}\;[[:alnum:]_.-]+$ ]]; then
        echo "${job_kind} sbatch returned an invalid parsable job ID: ${raw_output}" >&2
        return 1
    fi
    registered_job=${candidate}
}
cleanup_submission() {
    local exit_code=$1
    local failed_command=$2
    local rollback_output=""
    local rollback_state="no_jobs_created"
    trap - ERR INT TERM HUP EXIT
    set +e
    if [[ ${committed} -eq 0 ]]; then
        if [[ ${#submitted_jobs[@]} -gt 0 ]]; then
            if rollback_output=$("${scancel_bin}" "${submitted_jobs[@]}" 2>&1); then
                rollback_state="rollback_requested"
            else
                rollback_state="ROLLBACK_FAILED"
            fi
        fi
        echo "Submission transaction failed at ${failed_command}; state=${rollback_state}; jobs=${submitted_jobs[*]:-none}." >&2
        if [[ ${rollback_state} == "ROLLBACK_FAILED" ]]; then
            echo "CANCELLATION FAILED; jobs may remain active: ${submitted_jobs[*]}. scancel output: ${rollback_output}" >&2
        elif [[ -n ${rollback_output} ]]; then
            echo "scancel output: ${rollback_output}" >&2
        fi
    fi
    exit "${exit_code}"
}
cleanup_submission_signal() {
    local signal_name=$1
    local exit_code=$2
    cleanup_submission "${exit_code}" "received ${signal_name}"
}
trap 'cleanup_submission $? "$BASH_COMMAND"' ERR
trap 'cleanup_submission_signal INT 130' INT
trap 'cleanup_submission_signal TERM 143' TERM
trap 'cleanup_submission_signal HUP 129' HUP
trap 'exit_code=$?; if [[ ${committed} -eq 0 ]]; then cleanup_submission "${exit_code}" "unexpected shell exit"; fi' EXIT

prepare_result="$("${python_bin}" "${runner}" prepare "${common[@]}")"
printf '%s\n' "${prepare_result}"
bundle_sha256="$(
    printf '%s\n' "${prepare_result}" |
        /usr/bin/sed -nE 's/^[[:space:]]*"bundle_sha256": "([0-9a-f]{64})",?$/\1/p'
)"
[[ "${bundle_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "could not anchor prepared external bundle" >&2
    false
}
score_command[${#score_command[@]}-1]="${bundle_sha256}"
evaluation_script[${#evaluation_script[@]}-1]="${bundle_sha256}"
score_submission="$("${score_command[@]}")"
register_job_output "${score_submission}" "score array"
score_job=${registered_job}
if [[ ${evaluate} -eq 1 ]]; then
    evaluation_submission="$("${evaluation_command[@]}" --dependency="afterok:${score_job}" "${evaluation_script[@]}")"
    register_job_output "${evaluation_submission}" "evaluation"
    evaluation_job=${registered_job}
fi
"${python_bin}" "${runner}" record-submission \
    --run-dir "${run_dir}" \
    --score-job "${score_job}" \
    --evaluation-job "${evaluation_job}" \
    --bootstrap-replicates "${bootstrap_replicates}" \
    --expected-bundle-sha256 "${bundle_sha256}"
"${scontrol_bin}" release "${score_job}"
"${python_bin}" "${runner}" commit-submission \
    --run-dir "${run_dir}" \
    --expected-bundle-sha256 "${bundle_sha256}"
committed=1
trap - ERR INT TERM HUP EXIT
echo "Submitted score_array=${score_job} evaluation=${evaluation_job:-disabled} run=${run_dir}"
