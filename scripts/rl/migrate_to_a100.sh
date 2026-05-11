#!/usr/bin/env bash
#
# One-shot rsync to resume Phase 3 on another host. Transfers (in order):
#   1. SFT starting checkpoint     ~/runcodes/verl/checkpoints/sft_hotpot_cot_hf/
#   2. RL train parquet            ~/data/hotpotqa/rl_distractor_train.parquet
#   3. RL val parquet              ~/data/hotpotqa/rl_distractor_val.parquet
#   4. HotpotQA dev (eval)         ~/data/hotpotqa/hotpot_dev_distractor_v1.json
#   5. (optional) the repo itself, excluding heavy artifacts — when git
#      clone is too slow. Enable with SYNC_CODE=1.
#
# Usage:
#   DEST_HOST=user@a100-host \
#   DEST_REPO=/path/on/remote/verl \
#   DEST_DATA=/path/on/remote/data/hotpotqa \
#       bash scripts/rl/migrate_to_a100.sh
#
# Optional: DRY_RUN=1    preview without copying
#           SSH_PORT=22  non-default ssh port
#           SSH_KEY=...  use a specific private key
#           SYNC_CODE=1  also rsync the repo working tree (slow git clone fallback)
#           ONLY_CODE=1  only rsync the code, skip ckpt + data

set -euo pipefail

: "${DEST_HOST:?set DEST_HOST=user@host}"
: "${DEST_REPO:?set DEST_REPO=/path/to/remote/verl (must already be git-cloned there)}"
: "${DEST_DATA:?set DEST_DATA=/path/to/remote/data/hotpotqa}"

DRY_RUN="${DRY_RUN:-0}"
SSH_KEY="${SSH_KEY:-}"
SSH_PORT="${SSH_PORT:-22}"
SYNC_CODE="${SYNC_CODE:-0}"
ONLY_CODE="${ONLY_CODE:-0}"
[[ "$ONLY_CODE" == "1" ]] && SYNC_CODE=1

# Build a single ssh command for both rsync (-e) and the mkdir step.
SSH_CMD="ssh -p $SSH_PORT"
[[ -n "$SSH_KEY" ]] && SSH_CMD="$SSH_CMD -i $SSH_KEY"

REPO_ROOT="${VERL_ROOT:-/home/luoxuan/runcodes/verl}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/hotpotqa}"

SFT_CKPT="$REPO_ROOT/checkpoints/sft_hotpot_cot_hf"
TRAIN_PQ="$DATA_ROOT/rl_distractor_train.parquet"
VAL_PQ="$DATA_ROOT/rl_distractor_val.parquet"
DEV_JSON="$DATA_ROOT/hotpot_dev_distractor_v1.json"

if [[ "$ONLY_CODE" != "1" ]]; then
    for f in "$SFT_CKPT" "$TRAIN_PQ" "$VAL_PQ" "$DEV_JSON"; do
        [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 2; }
    done
fi

RSYNC_OPTS=(-avzP --partial --human-readable -e "$SSH_CMD")
[[ "$DRY_RUN" == "1" ]] && RSYNC_OPTS+=(--dry-run)

echo ">>> ensuring remote dirs exist on $DEST_HOST"
$SSH_CMD "$DEST_HOST" "mkdir -p '$DEST_REPO' '$DEST_REPO/checkpoints' '$DEST_DATA'"

if [[ "$SYNC_CODE" == "1" ]]; then
    echo
    echo ">>> [code] repo working tree -> $DEST_HOST:$DEST_REPO/"
    # Exclude heavy artifacts; keep .git so future git pull works.
    rsync "${RSYNC_OPTS[@]}" \
        --exclude='checkpoints/' \
        --exclude='tensorboard_log/' \
        --exclude='logs/' \
        --exclude='runs/' \
        --exclude='wandb/' \
        --exclude='outputs/' \
        --exclude='models/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='*.egg-info/' \
        --exclude='*.whl' \
        --exclude='.env' \
        "$REPO_ROOT/" \
        "$DEST_HOST:$DEST_REPO/"
fi

if [[ "$ONLY_CODE" != "1" ]]; then
    echo
    echo ">>> [ckpt] SFT checkpoint   -> $DEST_HOST:$DEST_REPO/checkpoints/sft_hotpot_cot_hf/"
    rsync "${RSYNC_OPTS[@]}" \
        "$SFT_CKPT/" \
        "$DEST_HOST:$DEST_REPO/checkpoints/sft_hotpot_cot_hf/"

    echo
    echo ">>> [data] hotpot parquets  -> $DEST_HOST:$DEST_DATA/"
    rsync "${RSYNC_OPTS[@]}" \
        "$TRAIN_PQ" "$VAL_PQ" "$DEV_JSON" \
        "$DEST_HOST:$DEST_DATA/"
fi

echo
echo "Done. Verify on remote:"
echo "  ssh -p $SSH_PORT $DEST_HOST 'ls -lh $DEST_REPO $DEST_REPO/checkpoints/sft_hotpot_cot_hf $DEST_DATA'"
