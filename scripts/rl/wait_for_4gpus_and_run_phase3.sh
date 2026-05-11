#!/usr/bin/env bash
#
# Wait until enough local GPUs look idle, then launch Phase 3 on those GPUs.
#
# Default policy is intentionally conservative for a shared 8x4090 box:
#   - need 4 GPUs
#   - each selected GPU must use <= 2000 MiB
#   - each selected GPU must have <= 5% utilization
#   - the condition must hold for 2 consecutive polls
#
# Usage:
#   bash scripts/rl/wait_for_4gpus_and_run_phase3.sh
#
# Useful overrides:
#   REQUIRED_GPUS=4
#   POLL_SECONDS=60
#   MAX_USED_MB=2000
#   MAX_UTIL_PCT=5
#   STABLE_POLLS=2
#   CANDIDATE_GPUS=0,1,2,3,4,5,6,7
#   DRY_RUN=1
#
# Extra environment such as SKIP_SMOKE, ONLY_SMOKE, SEED, EXTRA_OVERRIDES,
# ACTOR_PATH, TRAIN_PARQUET, and VAL_PARQUET is inherited by run_all_phase3.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="${VERL_ROOT:-/home/luoxuan/runcodes/verl}"
export VERL_ROOT
RUNNER="$SCRIPT_DIR/run_all_phase3.sh"

REQUIRED_GPUS="${REQUIRED_GPUS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MB="${MAX_USED_MB:-2000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-5}"
STABLE_POLLS="${STABLE_POLLS:-2}"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot monitor GPUs." >&2
    exit 2
fi

if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: Phase 3 runner not found: $RUNNER" >&2
    exit 2
fi

if ! [[ "$REQUIRED_GPUS" =~ ^[0-9]+$ ]] || [[ "$REQUIRED_GPUS" -lt 1 ]]; then
    echo "ERROR: REQUIRED_GPUS must be a positive integer." >&2
    exit 2
fi

candidate_contains() {
    local idx="$1"

    [[ -z "$CANDIDATE_GPUS" ]] && return 0
    case ",$CANDIDATE_GPUS," in
        *,"$idx",*) return 0 ;;
        *) return 1 ;;
    esac
}

find_idle_gpus() {
    local selected=()
    local line idx name mem_used util

    while IFS=, read -r idx name mem_used util; do
        idx="${idx//[[:space:]]/}"
        mem_used="${mem_used//[[:space:]]/}"
        util="${util//[[:space:]]/}"

        candidate_contains "$idx" || continue

        if [[ "$mem_used" -le "$MAX_USED_MB" && "$util" -le "$MAX_UTIL_PCT" ]]; then
            selected+=("$idx")
        fi

        if [[ "${#selected[@]}" -ge "$REQUIRED_GPUS" ]]; then
            (IFS=,; echo "${selected[*]}")
            return 0
        fi
    done < <(
        nvidia-smi \
            --query-gpu=index,name,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )

    return 1
}

echo "[$(date '+%F %T')] Waiting for $REQUIRED_GPUS idle GPU(s)."
echo "  policy: memory.used <= ${MAX_USED_MB}MiB, utilization <= ${MAX_UTIL_PCT}%, stable polls = $STABLE_POLLS"
[[ -n "$CANDIDATE_GPUS" ]] && echo "  candidates: $CANDIDATE_GPUS"
echo "  runner: $RUNNER"

stable_count=0
last_selection=""

while true; do
    if selection="$(find_idle_gpus)"; then
        if [[ "$selection" == "$last_selection" ]]; then
            stable_count=$((stable_count + 1))
        else
            stable_count=1
            last_selection="$selection"
        fi

        echo "[$(date '+%F %T')] Candidate idle GPUs: $selection (stable $stable_count/$STABLE_POLLS)"

        if [[ "$stable_count" -ge "$STABLE_POLLS" ]]; then
            # Final quick re-check narrows the race window before launching.
            final_selection="$(find_idle_gpus || true)"
            if [[ "$final_selection" != "$selection" ]]; then
                echo "[$(date '+%F %T')] GPU state changed before launch; continuing to wait."
                stable_count=0
                last_selection=""
                sleep "$POLL_SECONDS"
                continue
            fi

            echo "[$(date '+%F %T')] Launching Phase 3 with GPUS=$selection N_GPUS=$REQUIRED_GPUS"
            if [[ "$DRY_RUN" == "1" ]]; then
                echo "DRY_RUN=1 set; not launching."
                exit 0
            fi

            exec env GPUS="$selection" N_GPUS="$REQUIRED_GPUS" bash "$RUNNER"
        fi
    else
        stable_count=0
        last_selection=""
        echo "[$(date '+%F %T')] Fewer than $REQUIRED_GPUS idle GPU(s); checking again in ${POLL_SECONDS}s."
    fi

    sleep "$POLL_SECONDS"
done
