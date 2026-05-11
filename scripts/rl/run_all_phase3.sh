#!/usr/bin/env bash
#
# Phase 3 — sequential driver for the full v1+v2 ablation.
#
# Order (PHASE3_GRPO.md §3.5 + §7.1):
#   0. (gate)  v1  α=1.0  EM-only       — 100-step smoke; abort the rest if it fails
#   1.        v1  α=1.0  EM-only       — 1500 steps, seed 42
#   2.        v2  α=0.0  process-only  —  100 steps (necessary-fail baseline, §3.5)
#   3.        v2  α=0.3                — 1500 steps, seed 42
#   4.        v2  α=0.5  (main)        — 1500 steps, seed 42
#   5.        v2  α=0.7                — 1500 steps, seed 42
#   6.        v2  α=1.0  (F1-only)     — 1500 steps, seed 42
#
# The smoke gate is the only place we abort. After the gate, every run is
# launched even if a prior run failed — per the brief "中间不要停下".
# Each run streams logs into  $REPO_ROOT/logs/phase3/<run_name>.log
# and writes ckpts under   checkpoints/<run_name>/  via launch_grpo.sh.
#
# Usage:
#   bash scripts/rl/run_all_phase3.sh                # smoke gate + 6 runs
#   SKIP_SMOKE=1 bash scripts/rl/run_all_phase3.sh   # already smoked, run the 6
#   ONLY_SMOKE=1 bash scripts/rl/run_all_phase3.sh   # just the gate

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="${VERL_ROOT:-/home/luoxuan/runcodes/verl}"
export VERL_ROOT
LOG_DIR="$VERL_ROOT/logs/phase3"
mkdir -p "$LOG_DIR"

SKIP_SMOKE="${SKIP_SMOKE:-0}"
ONLY_SMOKE="${ONLY_SMOKE:-0}"
SEED="${SEED:-42}"

# GPU knobs (override per host).
#   GPUS    — comma list passed to CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3" or "0,1"
#             empty → inherit whatever is already exported.
#   N_GPUS  — value of trainer.n_gpus_per_node. If unset, derived from GPUS;
#             if GPUS also unset, defaults to 4 (current 4090 box).
GPUS="${GPUS:-}"
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    derived_n=$(awk -F, '{print NF}' <<< "$GPUS")
else
    derived_n=4
fi
N_GPUS="${N_GPUS:-$derived_n}"

# Pass-through extra hydra overrides for per-host tuning. Set as a
# space-separated string in env, e.g.:
#   EXTRA_OVERRIDES="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#                    actor_rollout_ref.rollout.gpu_memory_utilization=0.7"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-}"
read -r -a EXTRA_OVERRIDES_ARR <<< "$EXTRA_OVERRIDES_STR"

LAUNCHER="$SCRIPT_DIR/launch_grpo.sh"
[[ -x "$LAUNCHER" ]] || chmod +x "$LAUNCHER"

run_one () {
    # $1 = REWARD_VERSION, $2 = ALPHA, $3 = SMOKE (0/1), $4 = label
    local rv="$1" alpha="$2" smoke="$3" label="$4"
    local tag="phase3_${rv}_a${alpha}_s${SEED}"
    [[ "$smoke" == "1" ]] && tag="${tag}_smoke"
    local log="$LOG_DIR/${tag}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "=================================================================="
    echo "[$(date '+%F %T')] START $label  ($tag)"
    echo "  log: $log"
    echo "=================================================================="

    REWARD_VERSION="$rv" ALPHA="$alpha" SMOKE="$smoke" SEED="$SEED" \
        bash "$LAUNCHER" \
            trainer.n_gpus_per_node="$N_GPUS" \
            "${EXTRA_OVERRIDES_ARR[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    echo "[$(date '+%F %T')] END   $label  rc=$rc"
    return "$rc"
}

# ---------------------------------------------------------------------------
# 0. Smoke gate (Task 7): v1 EM-only, 100 steps. Abort the rest if it fails.
# ---------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" != "1" ]]; then
    if ! run_one v1 1.0 1 "SMOKE GATE  v1 EM-only 100 steps"; then
        echo "FATAL: smoke gate failed — refusing to launch the 6 full runs."
        echo "Inspect the smoke log under $LOG_DIR before retrying."
        exit 1
    fi
fi
[[ "$ONLY_SMOKE" == "1" ]] && { echo "ONLY_SMOKE=1 set; exiting after gate."; exit 0; }

# ---------------------------------------------------------------------------
# Sequential ablation. We DO NOT abort between runs (per "中间不要停下").
# Failures are logged; aggregate exit code is 0 iff all 6 succeed.
# ---------------------------------------------------------------------------
declare -a STATUS=()
declare -a LABELS=(
    "v1 α=1.0 EM-only"
    "v2 α=0.0 process-only (100 steps, expected to fail; §3.5)"
    "v2 α=0.3"
    "v2 α=0.5 (main)"
    "v2 α=0.7"
    "v2 α=1.0 F1-only"
)

run_one v1 1.0 0 "${LABELS[0]}"; STATUS+=("$?")
run_one v2 0.0 1 "${LABELS[1]}"; STATUS+=("$?")  # SMOKE=1 → 100 steps
run_one v2 0.3 0 "${LABELS[2]}"; STATUS+=("$?")
run_one v2 0.5 0 "${LABELS[3]}"; STATUS+=("$?")
run_one v2 0.7 0 "${LABELS[4]}"; STATUS+=("$?")
run_one v2 1.0 0 "${LABELS[5]}"; STATUS+=("$?")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "=================================================================="
echo "Phase 3 sequential driver — final summary"
echo "=================================================================="
overall=0
for i in "${!STATUS[@]}"; do
    rc="${STATUS[$i]}"
    [[ "$rc" -eq 0 ]] && tag="OK  " || { tag="FAIL"; overall=1; }
    printf "  [%s] rc=%-3s  %s\n" "$tag" "$rc" "${LABELS[$i]}"
done
exit "$overall"
