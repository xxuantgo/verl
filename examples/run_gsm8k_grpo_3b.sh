#!/bin/bash
set -x  # 打印每条执行的命令，方便调试

python3 -m verl.trainer.main_ppo \
    # ========== 算法选择 ==========
    algorithm.adv_estimator=grpo \
    # GRPO: 不需要 Critic，用同 prompt 多个回答的分数做组内归一化
    # 可选: gae（标准PPO，需要Critic）、reinforce_plus_plus、rloo 等

    # ========== 数据配置 ==========
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    # train_batch_size=16: 每次 rollout 生成 16 条 prompt 对应的回答
    # 完整含义见"参数详解"章节
    data.max_prompt_length=512 \
    # prompt 最大 token 数，超过会被截断或过滤
    data.max_response_length=1024 \
    # 模型最多生成 1024 个 token 的回答
    data.filter_overlong_prompts=True \
    # 过滤掉 prompt 长度超过 max_prompt_length 的样本，而不是截断
    data.truncation='error' \
    # 如果遇到需要截断的情况，抛出错误（严格模式）
    data.shuffle=False \
    # 不打乱数据顺序（训练集通常要打乱，这里关闭便于复现）

    # ========== 模型配置 ==========
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    # 模型路径：可以是本地路径（如 ./models/Qwen2.5-3B-Instruct）
    # 也可以是 HuggingFace Hub ID（会自动下载）
    actor_rollout_ref.model.lora_rank=64 \
    # LoRA 低秩适配：用 64 维的低秩矩阵代替全量参数更新，大幅减少显存
    # 设为 0 则关闭 LoRA，进行全量微调
    actor_rollout_ref.model.lora_alpha=32 \
    # LoRA 缩放系数：实际 LoRA 贡献 = (lora_alpha / lora_rank) × ΔW
    # 这里 alpha=32, rank=64，缩放系数=0.5
    actor_rollout_ref.model.use_remove_padding=True \
    # 去掉 padding token 后再计算，减少无效计算量
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    # 梯度检查点：用重新计算换显存，训练速度略降但显存大幅减少

    # ========== Actor（策略网络）训练配置 ==========
    actor_rollout_ref.actor.optim.lr=3e-6 \
    # 学习率：PPO/GRPO 训练通常用比 SFT 更小的学习率
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    # mini-batch 大小：每次 PPO 更新时用多少样本（详见参数详解）
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    # 每个 GPU 每次前向/反向的实际 batch 大小（梯度累积单元）
    actor_rollout_ref.actor.use_kl_loss=True \
    # 在 Actor loss 中加入 KL 散度惩罚项
    # 作用：防止 policy 偏离 reference model 太远
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    # KL loss 的权重系数（越大越保守）
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    # KL 估计方式：low_var_kl 是方差较低的估计，训练更稳定
    actor_rollout_ref.actor.entropy_coeff=0 \
    # 熵正则系数：0 表示不加熵正则
    # 熵正则可以防止策略过早收敛（鼓励探索）
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    # 不把模型参数 offload 到 CPU（关闭可提高速度，但占更多显存）
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    # 不把优化器状态 offload 到 CPU（同上）

    # ========== Rollout（推理生成）配置 ==========
    actor_rollout_ref.rollout.name=vllm \
    # 使用 vLLM 作为推理引擎（高效批量推理）
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    # Tensor 并行：把模型的注意力头/FFN 层切分到 2 个 GPU 上
    # 适合显存不够放整个模型的情况
    actor_rollout_ref.rollout.n=5 \
    # 每个 prompt 生成 5 个不同的回答
    # GRPO 用这 5 个回答的分数做组内归一化计算 Advantage
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    # vLLM 使用 60% 的 GPU 显存作为 KV Cache
    # 剩余 40% 留给 FSDP 训练阶段
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    # rollout 阶段计算 log_prob 时的 micro batch 大小
    actor_rollout_ref.rollout.load_format=safetensors \
    # 权重加载格式：safetensors 比 pytorch bin 更快更安全
    actor_rollout_ref.rollout.layered_summon=True \
    # 逐层加载权重（减少峰值显存）

    # ========== Reference Model 配置 ==========
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    # Reference model 计算 log_prob 时的 micro batch 大小
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    # Reference model 参数 offload 到 CPU
    # 原因：ref model 只在计算 KL 时用，可以牺牲速度换显存

    # ========== 算法细节 ==========
    algorithm.use_kl_in_reward=False \
    # 不把 KL 加到 reward 里（而是用 use_kl_loss 加到 loss 里）
    # 两种方式都能防止偏离，选一种即可

    # ========== Critic 配置 ==========
    trainer.critic_warmup=0 \
    # GRPO 不需要 Critic，所以 critic_warmup=0（跳过 critic 预热）

    # ========== 训练器配置 ==========
    trainer.logger='["console","wandb"]' \
    # 日志输出到终端 + WandB（需要先 wandb login）
    # 如果不用 wandb，改为 '["console"]'
    trainer.project_name='verl_grpo_example_gsm8k' \
    # WandB 项目名
    trainer.experiment_name='qwen2.5_3b_grpo_lora' \
    # WandB 实验名（每次运行的标识）
    trainer.n_gpus_per_node=2 \
    # 每个节点使用 2 块 GPU（我们用 2 块 4090）
    trainer.nnodes=1 \
    # 只有 1 个节点（单机训练）
    trainer.save_freq=20 \
    # 每 20 步保存一次 checkpoint
    trainer.test_freq=5 \
    # 每 5 步在验证集上跑一次评估
    trainer.total_epochs=15
    # 总共训练 15 个 epoch