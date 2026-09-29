#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --kind concordance|seeds --pairs-file FILE --output-dir NEW_DIR [--run-label LABEL] [--finality provisional|final] [--expected-total-chunks N] [--expected-overlap-count N] [--python PATH] [--map-concurrency N] [--map-args \"ARGS\"] [--sites-file FILE] [--submit]"
}

kind=""
pairs_file=""
output_dir=""
run_label="provisional"
finality="provisional"
expected_total_chunks="100000"
expected_overlap_count=""
submit=0
map_concurrency="12"
map_args=""
sites_file=""
python_bin="$(command -v python)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) kind="$2"; shift 2 ;;
        --pairs-file) pairs_file="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --run-label) run_label="$2"; shift 2 ;;
        --finality) finality="$2"; shift 2 ;;
        --expected-total-chunks) expected_total_chunks="$2"; shift 2 ;;
        --expected-overlap-count) expected_overlap_count="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --map-concurrency) map_concurrency="$2"; shift 2 ;;
        --map-args) map_args="$2"; shift 2 ;;
        --sites-file) sites_file="$2"; shift 2 ;;
        --submit) submit=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$kind" != "concordance" && "$kind" != "seeds" ]]; then
    echo "--kind must be concordance or seeds" >&2
    exit 2
fi
if [[ "$finality" != "provisional" && "$finality" != "final" ]]; then
    echo "--finality must be provisional or final" >&2
    exit 2
fi
if [[ ! "$expected_total_chunks" =~ ^[1-9][0-9]*$ ]]; then
    echo "--expected-total-chunks must be a positive integer" >&2
    exit 2
fi
if [[ ! "$map_concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "--map-concurrency must be a positive integer" >&2
    exit 2
fi
if [[ -n "$expected_overlap_count" && ! "$expected_overlap_count" =~ ^[0-9]+$ ]]; then
    echo "--expected-overlap-count must be a nonnegative integer" >&2
    exit 2
fi
if [[ "$kind" == "seeds" && "$finality" == "final" && -z "$expected_overlap_count" ]]; then
    echo "final seed runs require --expected-overlap-count" >&2
    exit 2
fi
if [[ -z "$pairs_file" || ! -f "$pairs_file" || -z "$output_dir" ]]; then
    echo "--pairs-file must exist and --output-dir is required" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "--python must name an executable Python interpreter" >&2
    exit 2
fi
python_bin="$(realpath "$python_bin")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "$script_dir/../../..")"
package_dir="$repo_root/validation/full_snv_concordance"
pairs_file="$(realpath "$pairs_file")"
output_dir="$(realpath -m "$output_dir")"
frozen_pairs="$output_dir/pairs.tsv"
pair_sha256="$(/usr/bin/sha256sum "$pairs_file" | /usr/bin/awk '{print $1}')"
code_sha256="$("$script_dir/fingerprint.sh" "$package_dir")"
pair_count="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$pairs_file")"
if [[ "$pair_count" -lt 1 ]]; then
    echo "pairs file contains no data rows" >&2
    exit 2
fi
task_count="$(( (pair_count + 249) / 250 ))"
last_task="$(( task_count - 1 ))"
map_command=(sbatch --parsable --hold --chdir="$repo_root" --array="0-${last_task}%${map_concurrency}" --account=ssalzbe1-chess --partition=parallel --export=NONE "$script_dir/map_array.sbatch" "$kind" "$frozen_pairs" "$output_dir" "$run_label" "$pair_sha256" "$code_sha256" "$package_dir" "$repo_root" "$python_bin" ${map_args} ${sites_file:+--sites-file "$sites_file"})

if [[ "$submit" -eq 0 ]]; then
    echo "DRY RUN: no jobs submitted. Re-run with --submit after reviewing these commands."
    printf '%q ' "${map_command[@]}"
    printf '\n'
    echo "The launcher submits the reducer and renderer with afterok dependencies, then releases the held map array."
    exit 0
fi

if [[ -e "$output_dir" ]]; then
    echo "--output-dir must not already exist; use a unique immutable run directory" >&2
    exit 2
fi
mkdir -p "$output_dir/maps" "$output_dir/logs" "$output_dir/report"
cp "$pairs_file" "$frozen_pairs"
chmod 0444 "$frozen_pairs"
if [[ "$(/usr/bin/sha256sum "$frozen_pairs" | /usr/bin/awk '{print $1}')" != "$pair_sha256" ]]; then
    echo "frozen pairs checksum mismatch" >&2
    exit 1
fi
printf 'kind\t%s\nrun_label\t%s\nfinality\t%s\nexpected_total_chunks\t%s\nexpected_overlap_count\t%s\npairs_sha256\t%s\ncode_sha256\t%s\npython\t%s\ntask_count\t%s\nmap_concurrency\t%s\nmap_args\t%s\nsites_file\t%s\n' \
    "$kind" "$run_label" "$finality" "$expected_total_chunks" "$expected_overlap_count" "$pair_sha256" "$code_sha256" "$python_bin" "$task_count" "$map_concurrency" "$map_args" "$sites_file" >"$output_dir/launch.tsv"
submitted_jobs=()
registered_job=""
committed=0
register_job_output() {
    local raw_output="$1"
    local job_kind="$2"
    local candidate="${raw_output%%;*}"
    if [[ "$candidate" =~ ^[0-9]+$ ]]; then
        submitted_jobs+=("$candidate")
    fi
    if [[ ! "$candidate" =~ ^[0-9]+$ ]] || \
       [[ "$raw_output" != "$candidate" && ! "$raw_output" =~ ^${candidate}\;[[:alnum:]_.-]+$ ]]; then
        echo "$job_kind sbatch returned an invalid parsable job ID: $raw_output" >&2
        return 1
    fi
    registered_job="$candidate"
}
record_submission_failure() {
    local exit_code="$1"
    local failed_command="$2"
    local jobs=""
    local rollback_state="${3:-no_jobs_created}"
    local rollback_output="${4:-}"
    local rollback_rc="${5:-0}"
    if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
        jobs="$(IFS=,; echo "${submitted_jobs[*]}")"
    fi
    printf 'status\tfailed\nfailed_at\t%s\nexit_code\t%s\nfailed_command\t%s\nsubmitted_jobs\t%s\nrollback_state\t%s\nrollback_rc\t%s\nrollback_output\t%s\n' \
        "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" "$failed_command" "$jobs" \
        "$rollback_state" "$rollback_rc" "$rollback_output" \
        >"$output_dir/submission_status.tsv"
}
cancel_registered_jobs() {
    local exit_code="$1"
    local failed_command="$2"
    local rollback_output=""
    local rollback_rc=0
    local rollback_state="no_jobs_created"
    trap - ERR INT TERM HUP EXIT
    set +e
    if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
        rollback_state="rollback_requested"
        rollback_output="$(scancel "${submitted_jobs[@]}" 2>&1)"
        rollback_rc=$?
        if [[ "$rollback_rc" -ne 0 ]]; then
            rollback_state="rollback_failed"
        fi
    fi
    record_submission_failure "$exit_code" "$failed_command" "$rollback_state" "$rollback_output" "$rollback_rc"
    if [[ "$rollback_state" == "rollback_failed" ]]; then
        echo "CANCELLATION FAILED; jobs may remain: ${submitted_jobs[*]:-none}; scancel rc=${rollback_rc}: ${rollback_output}" >&2
    fi
    exit "$exit_code"
}
cancel_submission_signal() {
    local signal_name="$1"
    local exit_code="$2"
    cancel_registered_jobs "$exit_code" "received $signal_name"
}
trap 'cancel_registered_jobs $? "$BASH_COMMAND"' ERR
trap 'cancel_submission_signal INT 130' INT
trap 'cancel_submission_signal TERM 143' TERM
trap 'cancel_submission_signal HUP 129' HUP
trap 'exit_code=$?; if [[ ${committed} -eq 0 ]]; then cancel_registered_jobs "${exit_code}" "unexpected shell exit"; fi' EXIT
map_submission="$("${map_command[@]}")"
register_job_output "$map_submission" "map"
map_job="$registered_job"
reduce_submission="$(sbatch --parsable --chdir="$repo_root" --dependency="afterok:$map_job" --account=ssalzbe1-chess --partition=parallel --export=NONE "$script_dir/reduce.sbatch" "$kind" "$output_dir" "$frozen_pairs" "$task_count" "$finality" "$expected_total_chunks" "$expected_overlap_count" "$pair_sha256" "$code_sha256" "$package_dir" "$repo_root" "$python_bin" "$sites_file")"
register_job_output "$reduce_submission" "reducer"
reduce_job="$registered_job"
report_submission="$(sbatch --parsable --chdir="$repo_root" --dependency="afterok:$reduce_job" --account=ssalzbe1-chess --partition=parallel --export=NONE "$script_dir/render.sbatch" "$output_dir" "$code_sha256" "$package_dir" "$repo_root" "$python_bin")"
register_job_output "$report_submission" "report"
report_job="$registered_job"
scontrol release "$map_job"
printf 'status\tsubmitted\nsubmitted_at\t%s\nmap_job\t%s\nreduce_job\t%s\nreport_job\t%s\n' \
    "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$map_job" "$reduce_job" "$report_job" \
    >"$output_dir/submission_status.tsv"
committed=1
trap - ERR INT TERM HUP EXIT
echo "Submitted map=$map_job reduce=$reduce_job report=$report_job"
