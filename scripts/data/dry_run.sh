#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/data/dry_run.sh
#
# End-to-end smoke test of the HotpotQA SFT cold-start pipeline.
#
# Goal: prove the four scripts (synthesize -> filter -> preprocess -> SFT)
# are wired up correctly *before* spending money on the full 5k synthesis
# run and hours on the full SFT. We use:
#   * a tiny sample (n=50) of HotpotQA,
#   * the smallest Qwen (0.5B) so any single 24GB (or even 12GB) GPU works,
#   * 1 epoch + low save_freq so a checkpoint actually lands on disk,
# and bail out at the first failing step.
#
# What this dry run validates:
#   1. DEEPSEEK_API_KEY / model id / base URL all work (synthesize succeeds).
#   2. Teacher prompt yields enough format-OK records to survive the filter.
#   3. parquet schema is what verl's MultiTurnSFTDataset expects (no
#      pyarrow type-inference blowups on extra_info / supporting_facts).
#   4. verl SFT trainer launches under FSDP, loads the tokenizer, runs at
#      least one optimizer step, and writes a checkpoint.
#
# Run from the repo root:
#   bash scripts/data/dry_run.sh
#
# Override anything via env, e.g.:
#   N=100 MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct bash scripts/data/dry_run.sh
# ---------------------------------------------------------------------------

# -e: stop at first error (pipeline is strictly sequential — later steps
#     would crash anyway if an earlier one failed).
# -u: catch typos in env var names.
# -o pipefail: if a piped command fails, fail the whole pipe (otherwise
#              `synthesize | tee` would mask synthesize errors).
set -euo pipefail

# --- Tunables --------------------------------------------------------------
# Override any of these on the command line:
#   N=20 NPROC=1 bash scripts/data/dry_run.sh
N=${N:-50}                          # how many HotpotQA examples to synthesize
CONCURRENCY=${CONCURRENCY:-4}       # async workers hitting DeepSeek
VAL_SIZE=${VAL_SIZE:-10}            # validation split out of the kept rows
NPROC=${NPROC:-1}                   # SFT GPUs; 1 is enough for 0.5B
EPOCHS=${EPOCHS:-1}                 # SFT epochs (dry run: just want 1)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/data/hotpotqa}
HOTPOT_INPUT=${HOTPOT_INPUT:-$DATA_DIR/hotpot_train_v1.1.json}
SAVE_PATH=${SAVE_PATH:-checkpoints/sft_hotpot_dryrun}

# Pin to a single physical GPU. The host has multiple cards; GPU 2 is the
# free one for this run. Setting this *before* torchrun is what matters —
# torchrun's --nproc_per_node only allocates among CUDA-visible devices.
# Override on the command line if a different card is free:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/data/dry_run.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

# Dry-run-only artifacts. Kept under DATA_DIR/dryrun/ so we never overwrite
# the real run's outputs and can `rm -rf` them in one shot.
WORK=$DATA_DIR/dryrun
RAW=$WORK/sft_cot_raw.jsonl
FILT=$WORK/sft_cot_filtered.jsonl
REJ=$WORK/sft_cot_rejects.jsonl
TRAIN_PARQ=$WORK/sft_cot_train.parquet
VAL_PARQ=$WORK/sft_cot_val.parquet

mkdir -p "$WORK"

# Pretty step banner so it's obvious in the log where each stage starts/ends.
step() { printf "\n========== %s ==========\n" "$*"; }

# --- Pre-flight ------------------------------------------------------------
# Don't even start if the things that *will* fail are easy to detect now:
#   * HotpotQA dump must exist (synthesize would crash on open()).
#   * DEEPSEEK_API_KEY must be set (synthesize errors out on a clean SystemExit
#     anyway, but we want the message up front, not after argparse spam).
[[ -f "$HOTPOT_INPUT" ]] || {
    echo "ERROR: HotpotQA dump not found at $HOTPOT_INPUT" >&2
    echo "       Download hotpot_train_v1.1.json from https://hotpotqa.github.io/" >&2
    exit 1
}

# .env is auto-loaded by synthesize_cot.py via python-dotenv, but `set -u`
# would trip if we referenced $DEEPSEEK_API_KEY here directly, so source it
# manually for the pre-flight check only.
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; . ./.env; set +a
fi
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
    echo "ERROR: DEEPSEEK_API_KEY not set (check .env or export it)" >&2
    exit 1
}

# --- Step 1: synthesize (CPU, DeepSeek API) --------------------------------
# Output is appended -> if a previous dry run wrote partial output, blow it
# away first so we measure today's behavior, not yesterday's resumed state.
step "1/4 synthesize  (n=$N, concurrency=$CONCURRENCY)"
rm -f "$RAW"
python scripts/data/synthesize_cot.py \
    --dataset hotpot \
    --input "$HOTPOT_INPUT" \
    --output "$RAW" \
    --n "$N" \
    --concurrency "$CONCURRENCY"

# Sanity: at least one line written. If DeepSeek was completely down or the
# model id is wrong, the file will be empty and step 2 would silently produce
# 0 rows; better to fail here with a clear message.
[[ -s "$RAW" ]] || { echo "ERROR: $RAW is empty — synthesize produced no records" >&2; exit 1; }
echo "  raw lines: $(wc -l <"$RAW")"

# --- Step 2: filter (CPU, regex/F1) ----------------------------------------
# No --tokenizer flag on purpose: dry run uses whitespace word count so we
# don't pay the HF model download. The real run can pass --tokenizer for a
# tighter token-cap check.
step "2/4 filter"
python scripts/data/filter_cot.py \
    --input "$RAW" \
    --output "$FILT" \
    --rejects "$REJ"

# Need *some* rows to survive into SFT. If kept==0 the pipeline is broken at
# the prompt or filter level — no point launching torchrun on an empty parquet.
KEPT=$(wc -l <"$FILT")
echo "  kept: $KEPT  rejected: $(wc -l <"$REJ" 2>/dev/null || echo 0)"
(( KEPT > 0 )) || { echo "ERROR: filter kept 0 rows — check $REJ for reasons" >&2; exit 1; }

# Make val_size sane: if filter killed enough rows that val>=kept the train
# split would be empty. Cap val at kept/2.
if (( KEPT <= VAL_SIZE * 2 )); then
    VAL_SIZE=$(( KEPT / 2 ))
    echo "  shrinking val_size to $VAL_SIZE so train split isn't empty"
fi

# --- Step 3: preprocess to parquet (CPU, pandas+pyarrow) -------------------
step "3/4 preprocess  (val_size=$VAL_SIZE)"
python scripts/data/preprocess_hotpot_sft.py \
    --input "$FILT" \
    --train-out "$TRAIN_PARQ" \
    --val-out   "$VAL_PARQ" \
    --val-size "$VAL_SIZE"

# --- Step 4: SFT (GPU) -----------------------------------------------------
# This is the one step that needs a GPU. With NPROC=1 + 0.5B model + max ~50
# samples it fits in 12-24GB and completes in ~10-20 min including the model
# download on first run.
#
# trainer.* overrides:
#   total_epochs=$EPOCHS  → dry run wants 1, full run uses 3
#   save_freq=10          → guarantee a checkpoint lands during a tiny run
#                            (default 200 wouldn't trigger on n=50)
#   test_freq=10          → same: trigger eval at least once
step "4/4 SFT  (model=$MODEL_PATH, nproc=$NPROC, epochs=$EPOCHS)"
NPROC="$NPROC" \
SAVE_PATH="$SAVE_PATH" \
DATA_DIR="$WORK" \
MODEL_PATH="$MODEL_PATH" \
bash scripts/sft/run_sft_hotpot.sh \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=10 \
    trainer.test_freq=10

step "DRY RUN OK"
echo "  data:        $WORK"
echo "  checkpoints: $SAVE_PATH"
echo
echo "Next: re-run with N=5000, MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct, NPROC=4"
echo "and the real DATA_DIR ($DATA_DIR) — see scripts/sft/run_sft_hotpot.sh."
