#!/usr/bin/env bash
#
# Phase 3 — HotpotQA Distractor GRPO launcher.
#
# Hyperparameters frozen per PHASE3_GRPO.md §5. Two knobs are exposed via env:
#   REWARD_VERSION ∈ {v1, v2}   (default: v1)
#   ALPHA          ∈ [0,1]      (default depends on version)
#
# Examples:
#   # v1 EM-only smoke test (100 steps)
#   REWARD_VERSION=v1 SMOKE=1 bash scripts/rl/launch_grpo.sh
#
#   # v2 main run, α=0.5, seed 42
#   REWARD_VERSION=v2 ALPHA=0.5 SEED=42 bash scripts/rl/launch_grpo.sh
#
#   # α ablation: 0.0 (process-only baseline, expected to fail) — auto-stops at 100 steps
#   REWARD_VERSION=v2 ALPHA=0.0 SMOKE=1 bash scripts/rl/launch_grpo.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Run-level config (env-overridable)
# ---------------------------------------------------------------------------

REWARD_VERSION="${REWARD_VERSION:-v1}"
# ALPHA default: v1 forces 1.0 (only EM branch is read), v2 defaults to 0.5.
if [[ "$REWARD_VERSION" == "v1" ]]; then
    DEFAULT_ALPHA="1.0"
else
    DEFAULT_ALPHA="0.5"
fi
ALPHA="${ALPHA:-$DEFAULT_ALPHA}"

SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"           # 1 → 100-step smoke test (overrides total_training_steps)
LENGTH_PENALTY="${LENGTH_PENALTY:-0}"  # leave off until length hacking is observed (PHASE3_GRPO.md §3.6)

# Paths — adjust HOME-relative if you move artifacts.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTOR_PATH="${ACTOR_PATH:-$REPO_ROOT/checkpoints/sft_hotpot_cot_hf/global_step_500}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$HOME/data/hotpotqa/rl_distractor_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$HOME/data/hotpotqa/rl_distractor_val.parquet}"
REWARD_PY="${REWARD_PY:-$REPO_ROOT/src/rewards/hotpot.py}"

DATE_TAG="$(date +%Y%m%d)"
RUN_NAME="phase3_${REWARD_VERSION}_a${ALPHA}_s${SEED}_${DATE_TAG}"
[[ "$SMOKE" == "1" ]] && RUN_NAME="${RUN_NAME}_smoke"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/$RUN_NAME}"

# ---------------------------------------------------------------------------
# Pre-flight: fail loudly if required inputs are missing.
# ---------------------------------------------------------------------------

[[ -d "$ACTOR_PATH"      ]] || { echo "ERROR: actor ckpt not found: $ACTOR_PATH"; exit 2; }
[[ -f "$REWARD_PY"       ]] || { echo "ERROR: reward script missing: $REWARD_PY"; exit 2; }
[[ -f "$TRAIN_PARQUET"   ]] || { echo "ERROR: train parquet missing: $TRAIN_PARQUET (run scripts/data/preprocess_hotpot_rl.py first)"; exit 2; }
[[ -f "$VAL_PARQUET"     ]] || { echo "ERROR: val parquet missing:   $VAL_PARQUET"; exit 2; }

mkdir -p "$OUTPUT_DIR"
echo "REWARD_VERSION=$REWARD_VERSION ALPHA=$ALPHA SEED=$SEED SMOKE=$SMOKE" | tee "$OUTPUT_DIR/run_env.txt"

# ---------------------------------------------------------------------------
# GRPO hyperparameters (PHASE3_GRPO.md §5)
# ---------------------------------------------------------------------------

# Smoke override: 100 steps, eval every 25.
if [[ "$SMOKE" == "1" ]]; then
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=50
    TEST_FREQ=25
else
    TOTAL_TRAINING_STEPS=1500
    SAVE_FREQ=100
    TEST_FREQ=100
fi

# ---------------------------------------------------------------------------
# Reward env: read by src/rewards/hotpot.py at import time.
# ---------------------------------------------------------------------------

export REWARD_VERSION ALPHA LENGTH_PENALTY

# Disable flashinfer sampler — its JIT compile fails on this box because
# `cuda/functional` (CCCL header) is missing from the system CUDA install,
# and vLLM falls back to the torch sampler with no measurable cost on 1.5B.
# See PHASE3_GRPO.md kickoff log 2026-05-10 for the original error trace.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=64 \
    data.max_prompt_length=1536 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path="$ACTOR_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    custom_reward_function.path="$REWARD_PY" \
    custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_hotpot_grpo' \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=2 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.seed=$SEED \
    "$@"
