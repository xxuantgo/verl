#!/bin/bash
# SFT cold-start on HotpotQA CoT data using verl's built-in trainer.
# Tested layout: 4x RTX 4090, Qwen2.5-1.5B-Instruct, FSDP + remove_padding.

set -x

gpu_ids=${GPU_IDS:-}
nproc_per_node=${NPROC:-4}
save_path=${SAVE_PATH:-checkpoints/sft_hotpot_cot}
data_dir=${DATA_DIR:-$HOME/data/hotpotqa}
model_path=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
tensorboard_dir=${TENSORBOARD_DIR:-tensorboard_log/hotpot-sft/qwen2.5-1.5b-cot}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-4}
max_length=${MAX_LENGTH:-4096}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-8192}
lr=${LR:-2e-5}
total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-100}

if [[ -n "$gpu_ids" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    IFS=',' read -ra visible_gpu_list <<< "$gpu_ids"
    visible_gpu_count=${#visible_gpu_list[@]}
    if [[ "$nproc_per_node" == "4" && "$visible_gpu_count" != "4" ]]; then
        nproc_per_node=$visible_gpu_count
    elif [[ "$nproc_per_node" != "$visible_gpu_count" ]]; then
        echo "ERROR: NPROC=$nproc_per_node but GPU_IDS exposes $visible_gpu_count GPU(s): $gpu_ids" >&2
        exit 1
    fi
fi

export TENSORBOARD_DIR="$tensorboard_dir"

[[ -f "$data_dir/sft_cot_train.parquet" ]] || { echo "ERROR: missing $data_dir/sft_cot_train.parquet" >&2; exit 1; }
[[ -f "$data_dir/sft_cot_val.parquet" ]] || { echo "ERROR: missing $data_dir/sft_cot_val.parquet" >&2; exit 1; }

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$data_dir/sft_cot_train.parquet \
    data.val_files=$data_dir/sft_cot_val.parquet \
    data.messages_key=messages \
    data.train_batch_size=$train_batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.max_length=$max_length \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=$max_token_len_per_gpu \
    data.truncation=right \
    optim.lr=$lr \
    optim.weight_decay=0.0 \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.1 \
    optim.clip_grad=1.0 \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=1 \
    model.path=$model_path \
    model.use_remove_padding=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=hotpot-sft \
    trainer.experiment_name=qwen2.5-1.5b-cot \
    trainer.logger="[console,tensorboard,wandb]" \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.seed=2026 \
    trainer.n_gpus_per_node=$nproc_per_node \
    "$@"
