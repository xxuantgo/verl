#!/usr/bin/env bash
# Run a single-GPU PPO training demo on GSM8K with Qwen2.5-0.5B-Instruct.
#
# Usage:
#   conda activate verl
#   cd /home/luoxuan/runcodes/verl
#   CUDA_VISIBLE_DEVICES=0 bash examples/run_gsm8k_ppo_0_5b.sh
#
# You can switch GPU by changing CUDA_VISIBLE_DEVICES, for example:
#   CUDA_VISIBLE_DEVICES=2 bash examples/run_gsm8k_ppo_0_5b.sh

set -euo pipefail

# Print each command before it runs. This is useful while learning because the
# terminal shows exactly what bash expands and executes.
set -x

# Resolve the repository root from this script location, so the script can be
# launched from any current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep Python logs unbuffered so tmux and tee show progress immediately.
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Reduce repeated Ray logs and tokenizer thread warnings/noise.
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false

# Pick one GPU by default. Override from the command line when another GPU is
# free, e.g. CUDA_VISIBLE_DEVICES=2 bash examples/run_gsm8k_ppo_0_5b.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Local data and model paths. Using a local model avoids downloading from
# Hugging Face during the first training run.
TRAIN_FILE="${TRAIN_FILE:-${HOME}/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-${HOME}/data/gsm8k/test.parquet}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct}"
export MODEL_PATH

# Put logs under logs/ so the repo root stays tidy.
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/gsm8k_ppo_0_5b_$(date +%Y%m%d_%H%M%S).log}"

# Lightweight preflight checks. They fail early with a clear message instead of
# letting verl fail much later during worker initialization.
test -f "${TRAIN_FILE}"
test -f "${VAL_FILE}"
test -d "${MODEL_PATH}"
test -f "${MODEL_PATH}/config.json"

python3 - <<'PY'
import os
import torch
import transformers
from transformers import AutoTokenizer

model_path = os.environ["MODEL_PATH"]
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment.")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if not hasattr(tokenizer, "all_special_tokens_extended"):
    raise SystemExit(
        "Tokenizer is missing all_special_tokens_extended. "
        "This is usually caused by an incompatible transformers version. "
        "For vLLM 0.11.0, use transformers 4.x, for example: "
        "pip install 'transformers==4.57.6'"
    )
print("tokenizer:", tokenizer.__class__.__name__)
PY

# Main PPO command.
#
# The command uses Hydra-style key=value overrides. Values not listed here come
# from verl/trainer/config/ppo_trainer.yaml.
python3 -m verl.trainer.main_ppo \
 data.train_files="${TRAIN_FILE}" \
 data.val_files="${VAL_FILE}" \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path="${MODEL_PATH}" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path="${MODEL_PATH}" \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 \
 2>&1 | tee "${LOG_FILE}"

echo "Log saved to: ${LOG_FILE}"
