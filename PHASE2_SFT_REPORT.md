# Phase 2 — HotpotQA CoT SFT 冷启动报告

> 对应 [PROJECT_PLAN1.md](PROJECT_PLAN1.md) §4 (Phase 2)
> 产出 checkpoint：`checkpoints/sft_hotpot_cot/global_step_500`
> 报告日期：2026-05-10
> 状态：**✅ 通过验收，可进入 Phase 3 GRPO 训练**

---

## 1. 阶段目标回顾

把 base 模型 `Qwen2.5-1.5B-Instruct` 通过 SFT 教会两件事：

1. **输出协议**：稳定输出 `<think>…</think><answer>…</answer>` 格式，让后续 RL 的 reward 函数可解析、可打分。
2. **多跳推理范式**：在给定 passages 的前提下，先在 `<think>` 中拼接证据，再在 `<answer>` 中给短答案。

产物 `Qwen2.5-1.5B-HotpotCoT` 同时是 Phase 2 的交付物，也是 Phase 3 GRPO 的初始权重。

---

## 2. 数据流水线（Phase 2 上半段）

[scripts/data/](scripts/data/) 三步管线，由 [run_data_pipeline.sh](scripts/data/run_data_pipeline.sh) 串起：

```
hotpot_train_v1.1.json
        │
        ▼  scripts/data/synthesize_cot.py
sft_cot_raw.jsonl              ← teacher (DeepSeek) 合成的 5000 条 CoT
        │
        ▼  scripts/data/filter_cot.py
sft_cot_filtered.jsonl         ← 规则过滤后保留的高质量样本
sft_cot_rejects.jsonl          ← 被剔除的样本（含原因）
        │
        ▼  scripts/data/preprocess_hotpot_sft.py  --val-size 200
sft_cot_train.parquet  (4612)  ← verl SFT 训练集
sft_cot_val.parquet    (200)   ← verl SFT 验证集
```

### 2.1 合成 — synthesize_cot.py

- 教师模型：DeepSeek（异步并发 16）
- 模板：将 question + passages + gold answer + supporting_facts 喂给教师，强制输出 `<think>…</think><answer>…</answer>`
- 5000 条原始 → 写入 `sft_cot_raw.jsonl`，**断点续跑**

### 2.2 过滤 — filter_cot.py

保留满足以下全部条件的样本：

- `<think>…</think><answer>…</answer>` 整体可被正则解析
- `<think>` 长度 ≥ `--min-think-chars`
- `<answer>` 与 gold answer 归一化后**精确匹配或互为子串**
- 总 token 数 ≤ `--max-tokens`

最终 4812 条进入下一步（4612 train + 200 val），通过率 ~96%。**注意：val 是过滤后的子集，不是原始 HotpotQA dev**——这一点直接决定了第 5 节"评测的乐观偏差"问题。

### 2.3 转格式 — preprocess_hotpot_sft.py

把 JSONL 转成 verl `MultiTurnSFTDataset` 要求的 parquet schema：

```python
{
  "messages": [
    {"role": "system",    "content": "...格式约定..."},
    {"role": "user",      "content": "Passages: ...\nQuestion: ..."},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"},
  ],
  "data_source": "hotpotqa",
  "extra_info": {"qid", "raw_question", "raw_answer", "supporting_facts", "meta"},
}
```

verl 训练时调用 tokenizer 的 chat_template，**自动只对 assistant turn 算 loss**（system / user 部分被 mask 掉），免去手写 loss masking。

---

## 3. SFT 训练（Phase 2 下半段）

### 3.1 启动命令（实际跑过的）

```bash
GPU_IDS=0,1 NPROC=2 \
MODEL_PATH=/home/luoxuan/runcodes/verl/models/Qwen2.5-1.5B-Instruct \
TRAIN_BATCH_SIZE=16 \
MICRO_BATCH_SIZE_PER_GPU=4 \
MAX_LENGTH=4096 \
MAX_TOKEN_LEN_PER_GPU=8192 \
bash scripts/sft/run_sft_hotpot.sh
```

完整脚本：[scripts/sft/run_sft_hotpot.sh](scripts/sft/run_sft_hotpot.sh)

### 3.2 关键超参与对应口径

| 超参 | 取值 | 说明 |
|---|---|---|
| `model.path` | Qwen2.5-1.5B-Instruct | full fine-tuning，不用 LoRA |
| `engine` | fsdp | 2 卡分片 |
| `data.train_batch_size` | 16 | global，每 step 见 16 条样本 |
| `data.micro_batch_size_per_gpu` | 4 | 单卡 micro，配合 dynamic bsz |
| `data.use_dynamic_bsz` | true | 按 token 数动态打包 |
| `data.max_token_len_per_gpu` | 8192 | dynamic bsz 上限 |
| `data.max_length` | 4096 | 单样本截断 |
| `data.truncation` | right | 超长从右截 |
| `model.use_remove_padding` | true | 去 padding 计算，提速 |
| `optim.lr` | 2e-5 | 标准 SFT lr |
| `optim.lr_warmup_steps_ratio` | 0.03 | ~26 步 warmup |
| `optim.lr_scheduler_type` | cosine | |
| `optim.min_lr_ratio` | 0.1 | 末端衰到 2e-6 |
| `optim.clip_grad` | 1.0 | |
| `optim.weight_decay` | 0.0 | |
| `trainer.total_epochs` | 3 | |
| `trainer.save_freq` / `test_freq` | 100 / 100 | 每 100 步存 ckpt + 跑 val |
| `trainer.seed` | 2026 | 可复现 |

总 step 推算：⌈4612 / 16⌉ × 3 = 289 × 3 = **867 step**，与实际 wandb 上 864 步、最终 [global_step_864](checkpoints/sft_hotpot_cot/global_step_864) 完全吻合。

### 3.3 wandb 训练曲线分析

W&B run：`hotpot-sft / qwen2.5-1.5b-cot`

| 曲线 | 形态 | 解读 |
|---|---|---|
| `train/loss` | 1.0 → 0.2，**step 290 / 580 两个台阶** | 正常；台阶对应 epoch 边界，是小数据多 epoch 的典型记忆化痕迹 |
| `train/grad_norm` | 第 1 步 spike 30+ → 稳定在 5 以下 | 健康；clip_grad=1.0 工作正常 |
| `train/lr` | 0 → 2e-5（warmup ~26 步）→ 余弦衰至 ~2e-6 | 与配置一致 |
| `train/global_tokens` | 22k–30k 区间抖动 | dynamic batch 在按 token 打包，没有失控 |
| `train/total_tokens(B)` | 单调线性升至 0.022B | 累计训练 ~22M token，规模合理 |
| `train/mfu` | 全程 0 | 没传入 model FLOPs 表，**非 bug**，不影响训练 |
| `val/loss` | step 100 ~0.54 → step 500 **0.51（谷底）**→ step 600 起跳到 0.6 持平 | **明确的 epoch 3 过拟合信号**；最佳 ckpt 在 step 500 附近 |

**关键决策**：根据 val/loss 谷底，选 **`global_step_500`** 作为 Phase 3 的起点，而**不是**最后的 `global_step_864`。

---

## 4. Checkpoint 整理（FSDP → HF）

### 4.1 两种格式的区别

| 路径 | 用途 | 关键文件 |
|---|---|---|
| [checkpoints/sft_hotpot_cot/global_step_500/](checkpoints/sft_hotpot_cot/global_step_500/) | **训练态**，可 resume | `model_world_size_2_rank_*.pt`, `optim_*.pt`, `extra_state_*.pt`, `data_*.pt`, `fsdp_config.json`, `huggingface/`（仅 tokenizer/config） |
| [checkpoints/sft_hotpot_cot_hf/global_step_500/](checkpoints/sft_hotpot_cot_hf/global_step_500/) | **推理 / RL 起点 / HF 上传** | `model.safetensors`, `config.json`, `tokenizer*`, `chat_template.jinja`, `generation_config.json` |

### 4.2 转换命令

```bash
/home/luoxuan/anaconda3/envs/verl/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/sft_hotpot_cot/global_step_500 \
  --target_dir checkpoints/sft_hotpot_cot_hf/global_step_500
```

`verl.model_merger` 做的事：(1) 把 FSDP 各 rank 的分片张量按切分规则**还原回完整张量**；(2) 复制 `huggingface/` 下的 tokenizer/config；(3) 输出标准 `.safetensors`。**优化器和 RNG state 会被丢弃**——它们只对 resume 有用。

权重在两边完全等价，只是布局不同。Phase 3 的 verl actor 只接受 HF 格式，所以**RL 启动一定指 `_hf` 路径**。

---

## 5. 健康检查评测（eval_quick）

### 5.1 命令

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/luoxuan/anaconda3/envs/verl/bin/python scripts/sft/eval_quick.py \
  --ckpt checkpoints/sft_hotpot_cot_hf/global_step_500 \
  --base /home/luoxuan/runcodes/verl/models/Qwen2.5-1.5B-Instruct \
  --val /home/luoxuan/data/hotpotqa/sft_cot_val.parquet \
  --n 100 \
  --max-new-tokens 512
```

脚本：[scripts/sft/eval_quick.py](scripts/sft/eval_quick.py)。逻辑：从 `sft_cot_val.parquet` 200 条中采样 100 条（seed=2026），剥掉 gold assistant 轮，对 SFT 与 base 各做一次 greedy 生成，比较三个指标。

### 5.2 结果

| 指标 | BASE (Qwen2.5-1.5B-Instruct) | SFT step_500 | Δ |
|---|---|---|---|
| format_rate | 0.00 | **1.00** | +1.00 |
| exact_match | 0.01 | **0.55** | +0.54 |
| substring_match | 0.01 | **0.65** | +0.64 |

### 5.3 解读

- **format_rate 0 → 100%**：SFT 冷启动的核心价值——把"输出协议"刻进模型。这是 Phase 3 的硬性前提，没有它，奖励函数大部分时候打 0 分，GRPO 学不动。
- **EM 0.01 → 0.55**：base 答对 1% 是纯偶然；SFT 后远超 PROJECT_PLAN1.md §4.3 给出的 30–38% 期望区间。
- **substring − EM = 10pp**：这部分错主要是冗余措辞 / 单复数 / 标点，不是答非所问。

### 5.4 这些数字的乐观偏差（重要）

**val 集是合成 + 过滤后的 in-distribution 子集**，与 train 同源、system prompt 相同、passages 排版一致，且经过"gold answer 必须出现在 `<answer>`"的过滤。所以 55% EM 是乐观上界。

**等 Phase 3 在原版 HotpotQA dev distractor 全集（7405 条）上做严格评测，预期会掉到 30s 区间**——那才是 RL 的真起跑线。本节数字仅作健康检查，**不能对外宣称为 HotpotQA EM 数字**。

---

## 6. 这个 SFT 模型够格进 Phase 3 吗？

### 6.1 硬指标全部通过

| 检查 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 格式合规率 | ≥ 95% | 100% | ✅ |
| in-domain EM | ≥ 30% | 55% | ✅ |
| val/loss 是否找到谷底 | 必需 | step 500 = 0.51 | ✅ |
| HF 格式可加载 | 必需 | 已 merge | ✅ |
| chat_template / tokenizer 一致 | 必需 | merge 时已复制 | ✅ |
| 训练曲线无异常（grad spike / loss NaN） | 必需 | 仅首步 grad spike，已被 clip 吸收 | ✅ |

### 6.2 是否需要重训 / 调参

**不需要重训**。按当前数据规模 (4612) 和模型尺度 (1.5B)，再投入算力的边际收益很低。如果未来真要优化 Phase 2，优先级排序：

1. **数据扩量到 8–10k** —— PROJECT_PLAN1.md §7.2 消融 1 已经预告"3k 后边际收益递减"，但 4.6k → 8k 仍可能再涨 2–4 个 EM 点。比调超参更划算。
2. **训练只跑 2 epoch** —— 当前曲线显示 epoch 3 已经过拟合，2 epoch + step_500 附近就够。
3. **不动 lr / batch size** —— 2e-5 + cosine + bsz 16 是 SFT 标准配置，曲线表明它工作得很稳，没有调的必要。
4. **不要切到 LoRA** —— 1.5B 全量调单卡显存有余，且 Phase 3 RL 也是 full fine-tuning，保持一致避免接口对接问题。

### 6.3 已知风险与 Phase 3 应对

| 风险 | 表征 | Phase 3 应对 |
|---|---|---|
| in-domain 评测乐观 | val EM 55% 但 dev distractor 全集预计只有 30s | Phase 3 第一件事：在原版 HotpotQA dev 上跑 baseline，记录真实起跑数字 |
| epoch 3 已过拟合 | val/loss 0.51 → 0.6 | 已通过选 step_500 而非 step_864 规避 |
| 评测脚本不算 supporting facts F1 | 没有 Joint F1 数字 | Phase 3 评测脚本必须补上 sf-F1 / Joint F1，对应 v2 reward |
| substring_match 太宽松 | 短答案被误判对 | Phase 3 改用 HotpotQA 官方 token-level F1 + EM |
| eval_quick 对 base 不公平 | base format_rate=0 是必然 | 仅作健康检查；正式对比要给 base 加 1-shot 演示或换公开 benchmark 数字 |

---

## 7. 下一步行动清单（进入 Phase 3 前）

按优先级：

1. ✅ **就用 `checkpoints/sft_hotpot_cot_hf/global_step_500` 作为 Phase 3 actor.model.path 起点**（不要用 step_864）。
2. ⬜ 写 [scripts/data/preprocess_hotpot_rl.py](scripts/data/) （新文件），把 `hotpot_dev_distractor_v1.json` 转成 verl RL 要求的 parquet（PROJECT_PLAN1.md §5.3 Task 3.1）。
3. ⬜ 在原版 dev distractor 全集上跑一次 SFT 模型评测（Joint F1 + Answer F1 + sf-F1），记录为 RL baseline 数字。
4. ⬜ 实现 [verl/utils/reward_score/hotpotqa.py](verl/utils/reward_score/) 的 reward v1（answer F1）和 v2（answer F1 + sf F1）。
5. ⬜ 写 `scripts/rl/train_grpo_distractor_v1.sh`，参考 PROJECT_PLAN1.md §5.3 Task 3.3 的超参表。
6. ⬜ 跑 v1 训练 500 步看趋势，确认 reward 单调上升、KL 稳定 < 0.5、response_length 不爆。

---

## 8. 关键文件索引

**数据流水线**
- [scripts/data/synthesize_cot.py](scripts/data/synthesize_cot.py)
- [scripts/data/filter_cot.py](scripts/data/filter_cot.py)
- [scripts/data/preprocess_hotpot_sft.py](scripts/data/preprocess_hotpot_sft.py)
- [scripts/data/prompt_templates.py](scripts/data/prompt_templates.py)
- [scripts/data/run_data_pipeline.sh](scripts/data/run_data_pipeline.sh)

**SFT 训练**
- [scripts/sft/run_sft_hotpot.sh](scripts/sft/run_sft_hotpot.sh)
- 入口模块：`verl.trainer.sft_trainer`

**评测**
- [scripts/sft/eval_quick.py](scripts/sft/eval_quick.py)

**Checkpoint 转换**
- 模块：`verl.model_merger`（[verl/model_merger/fsdp_model_merger.py](verl/model_merger/fsdp_model_merger.py)）

**产物**
- 训练态：[checkpoints/sft_hotpot_cot/global_step_500/](checkpoints/sft_hotpot_cot/global_step_500/)
- HF 推理态：[checkpoints/sft_hotpot_cot_hf/global_step_500/](checkpoints/sft_hotpot_cot_hf/global_step_500/) ← **Phase 3 起点**
- 数据：`~/data/hotpotqa/sft_cot_{train,val}.parquet`
- W&B：`hotpot-sft / qwen2.5-1.5b-cot`
- TensorBoard：[tensorboard_log/hotpot-sft/qwen2.5-1.5b-cot/](tensorboard_log/hotpot-sft/qwen2.5-1.5b-cot/)
