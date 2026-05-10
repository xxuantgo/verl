#!/usr/bin/env bash
#
# One-shot rsync of the *data + ckpt* needed to resume Phase 3 on another host.
# Code is NOT transferred here — push it via git instead (see MIGRATE_TO_A100.md).
#
# What gets transferred (≈1.5 GB total):
#   - SFT starting checkpoint     ~/runcodes/verl/checkpoints/sft_hotpot_cot_hf/
#   - RL train parquet            ~/data/hotpotqa/rl_distractor_train.parquet
#   - RL val parquet              ~/data/hotpotqa/rl_distractor_val.parquet
#   - HotpotQA dev (eval)         ~/data/hotpotqa/hotpot_dev_distractor_v1.json
#
# Usage:
#   DEST_HOST=user@a100-host \
#   DEST_REPO=/path/on/remote/verl \
#   DEST_DATA=/path/on/remote/data/hotpotqa \
#       bash scripts/rl/migrate_to_a100.sh
#
# Optional: DRY_RUN=1   preview without copying
#           SSH_PORT=22 non-default ssh port
#           SSH_KEY=... use a specific private key

set -euo pipefail

: "${DEST_HOST:?set DEST_HOST=user@host}"
: "${DEST_REPO:?set DEST_REPO=/path/to/remote/verl (must already be git-cloned there)}"
: "${DEST_DATA:?set DEST_DATA=/path/to/remote/data/hotpotqa}"

DRY_RUN="${DRY_RUN:-0}"
SSH_KEY="${SSH_KEY:-}"
SSH_PORT="${SSH_PORT:-22}"

# Build a single ssh command for both rsync (-e) and the mkdir step.
SSH_CMD="ssh -p $SSH_PORT"
[[ -n "$SSH_KEY" ]] && SSH_CMD="$SSH_CMD -i $SSH_KEY"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/data/hotpotqa}"

SFT_CKPT="$REPO_ROOT/checkpoints/sft_hotpot_cot_hf"
TRAIN_PQ="$DATA_ROOT/rl_distractor_train.parquet"
VAL_PQ="$DATA_ROOT/rl_distractor_val.parquet"
DEV_JSON="$DATA_ROOT/hotpot_dev_distractor_v1.json"

for f in "$SFT_CKPT" "$TRAIN_PQ" "$VAL_PQ" "$DEV_JSON"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 2; }
done

RSYNC_OPTS=(-avzP --partial --human-readable -e "$SSH_CMD")
[[ "$DRY_RUN" == "1" ]] && RSYNC_OPTS+=(--dry-run)

echo ">>> ensuring remote dirs exist on $DEST_HOST"
$SSH_CMD "$DEST_HOST" "mkdir -p '$DEST_REPO/checkpoints' '$DEST_DATA'"

echo
echo ">>> [1/2] SFT checkpoint  -> $DEST_HOST:$DEST_REPO/checkpoints/sft_hotpot_cot_hf/"
rsync "${RSYNC_OPTS[@]}" \
    "$SFT_CKPT/" \
    "$DEST_HOST:$DEST_REPO/checkpoints/sft_hotpot_cot_hf/"

echo
echo ">>> [2/2] data files       -> $DEST_HOST:$DEST_DATA/"
rsync "${RSYNC_OPTS[@]}" \
    "$TRAIN_PQ" "$VAL_PQ" "$DEV_JSON" \
    "$DEST_HOST:$DEST_DATA/"

echo
echo "Done. Verify on remote:"
echo "  ssh $DEST_HOST 'ls -lh $DEST_REPO/checkpoints/sft_hotpot_cot_hf $DEST_DATA'"
