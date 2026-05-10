#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/data/run_data_pipeline.sh
#
# Full-size data pipeline (synthesize -> filter -> preprocess) WITHOUT SFT.
# Used when:
#   * the dry run already validated the pipeline end-to-end, and
#   * we want the real training parquet on disk while we wait for free GPUs
#     to start the SFT step separately.
#
# Outputs land in $DATA_DIR (default ~/data/hotpotqa) — the same paths
# scripts/sft/run_sft_hotpot.sh reads from. Synthesize is resumable, so if
# this script is interrupted, re-running picks up where it left off.
# ---------------------------------------------------------------------------
set -euo pipefail

N=${N:-5000}
CONCURRENCY=${CONCURRENCY:-16}
VAL_SIZE=${VAL_SIZE:-200}
DATA_DIR=${DATA_DIR:-$HOME/data/hotpotqa}
HOTPOT_INPUT=${HOTPOT_INPUT:-$DATA_DIR/hotpot_train_v1.1.json}

RAW=$DATA_DIR/sft_cot_raw.jsonl
FILT=$DATA_DIR/sft_cot_filtered.jsonl
REJ=$DATA_DIR/sft_cot_rejects.jsonl
TRAIN_PARQ=$DATA_DIR/sft_cot_train.parquet
VAL_PARQ=$DATA_DIR/sft_cot_val.parquet

step() { printf "\n========== %s ==========\n" "$*"; }

# Auto-load DEEPSEEK_API_KEY etc. from .env for the pre-flight check.
[[ -f .env ]] && { set -a; . ./.env; set +a; }
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "ERROR: DEEPSEEK_API_KEY not set" >&2; exit 1; }
[[ -f "$HOTPOT_INPUT" ]] || { echo "ERROR: $HOTPOT_INPUT missing" >&2; exit 1; }

step "1/3 synthesize  (n=$N, concurrency=$CONCURRENCY) — resumable"
# NOTE: we deliberately do NOT rm $RAW. synthesize_cot.py's load_done_qids
# scans existing JSONL and skips already-written qids, so an interrupted
# run resumes seamlessly. If you want a clean start, rm it manually.
python scripts/data/synthesize_cot.py \
    --dataset hotpot \
    --input "$HOTPOT_INPUT" \
    --output "$RAW" \
    --n "$N" \
    --concurrency "$CONCURRENCY"

[[ -s "$RAW" ]] || { echo "ERROR: $RAW empty after synthesize" >&2; exit 1; }
echo "  raw lines: $(wc -l <"$RAW")"

step "2/3 filter"
python scripts/data/filter_cot.py \
    --input "$RAW" \
    --output "$FILT" \
    --rejects "$REJ"

KEPT=$(wc -l <"$FILT")
echo "  kept: $KEPT  rejected: $(wc -l <"$REJ" 2>/dev/null || echo 0)"
(( KEPT > 0 )) || { echo "ERROR: filter kept 0 rows" >&2; exit 1; }

# Adjust val_size only if filter killed too much (shouldn't happen on full run,
# but cheap to guard).
if (( KEPT <= VAL_SIZE * 2 )); then
    VAL_SIZE=$(( KEPT / 5 ))
    echo "  shrinking val_size to $VAL_SIZE"
fi

step "3/3 preprocess  (val_size=$VAL_SIZE)"
python scripts/data/preprocess_hotpot_sft.py \
    --input "$FILT" \
    --train-out "$TRAIN_PARQ" \
    --val-out   "$VAL_PARQ" \
    --val-size "$VAL_SIZE"

step "DATA PIPELINE OK"
echo "  train: $TRAIN_PARQ"
echo "  val  : $VAL_PARQ"
echo "  next : bash scripts/sft/run_sft_hotpot.sh   # when GPUs are free"
