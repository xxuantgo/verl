# HotpotQA Multi-hop Agentic RL — 当前进度纪要

> 日期：2026-05-10
> 阶段：Phase 2（SFT 冷启动）数据管线已就绪；正式合成 5000 条进行中
> 下一步：等数据落盘 → 多卡空闲后跑正式 SFT → 进 Phase 3 (RL rollout)

---

## 1. 项目目标回顾

在 [verl-project/verl](https://github.com/verl-project/verl) 上做一个**多跳 QA agentic RL** 的最小可运行 pipeline，基座为 Qwen2.5-1.5B-Instruct，数据集 HotpotQA。

三个阶段：

| 阶段 | 目的 | 状态 |
|---|---|---|
| **Phase 1** 数据准备 | 拿到 HotpotQA 原始 dump，理解 schema | ✅ 已完成 |
| **Phase 2** SFT 冷启动 | 用 DeepSeek 老师生成 `<think>...</think><answer>...</answer>` 格式的 CoT 数据，SFT 给 Qwen2.5-1.5B 做格式 + reasoning prior | 🔄 数据合成中 |
| **Phase 3** RL | 在 SFT 模型上跑 GRPO/PPO，reward = 答案 EM/F1 + 格式奖励 | ⏸ 待启动 |

---

## 2. 本次会话做了什么

### 2.1 抽取共享 prompt 模板（解耦 Phase 2 / Phase 3）

新增 [scripts/data/prompt_templates.py](scripts/data/prompt_templates.py)，作为「学生 prompt + 答案格式正则」的**单一来源**，避免 SFT 训出来的格式 ↔ RL 推理时给的 prompt ↔ reward 抽取答案的正则三方漂移。

文件包含：
- `STUDENT_SYSTEM_PROMPT` / `render_student_user(...)` — SFT 预处理 + 未来 RL rollout 复用
- `TEACHER_SYSTEM_PROMPT` / `render_teacher_user(...)` — 仅 synthesizer 用（leak gold）
- `FORMAT_RE` / `parse_format(text)` — 完整 `<think>...</think><answer>...</answer>` 匹配
- `ANSWER_RE` / `extract_answer(text)` — 仅抽 `<answer>`（取最后一个，避免 think 段误伤），留给 Phase 3 reward
- `FORMAT_SPEC` — 格式说明字符串本身的单点定义

`preprocess_hotpot_sft.py` 与 `synthesize_cot.py` 都改成 `from prompt_templates import ...`。

### 2.2 修 DeepSeek 限流逻辑

[scripts/data/synthesize_cot.py](scripts/data/synthesize_cot.py) 的 worker 之前对所有 `Exception` 一律指数退避，会在「model id 错」「key 错」这类**永远不会自愈**的 4xx 上空等 4 次。改成按异常类型分流：

| 异常 | 处理 |
|---|---|
| `HTTPStatusError` 429 / 5xx | 读 `Retry-After`（数值秒或 HTTP-date），加抖动后退避 |
| `HTTPStatusError` 其他 4xx | **fail fast**，打印 body 跳过 |
| `TransportError` / `TimeoutException` | 指数退避 |
| 其他 | 立即跳过 |

新增 `_retry_after_seconds(response)` helper 处理两种 RFC 7231 时间格式。

### 2.3 修 HotpotQA loader 兼容 HuggingFace schema

原 loader 假设 Stanford 官方 dump 的 `_id` + `[[title, [sents]]]` 格式，但本机数据是 HuggingFace `hotpot_qa` 的 `id` + `{"title": [], "sentences": [[]]}` 字典并行数组格式，dry run 第一次跑直接 `KeyError: '_id'`。

修复策略：在 `load_hotpot()` / `_format_hotpot_passages()` 里同时支持两种 schema，`supporting_facts` 也归一化成 `[[title, sent_id], ...]`，下游 parquet 看到统一格式。

### 2.4 Dry run 脚本 [scripts/data/dry_run.sh](scripts/data/dry_run.sh)

端到端冒烟测试，用最小数据量（n=50, Qwen2.5-0.5B, 1 卡, 1 epoch）跑通 1→2→3→4 全流程：

- `set -euo pipefail` — 第一步挂就停
- 预检查 HotpotQA 文件 + `DEEPSEEK_API_KEY`
- 每步只校验"对下一步够不够"（synthesize 非空 / KEPT>0 / val_size 自动收缩）
- `save_freq=10 test_freq=10` 覆盖默认值，确保小数据也能触发 ckpt
- `export CUDA_VISIBLE_DEVICES=2` 显式锁卡（按当前唯一空闲 GPU）
- 全部产物写到 `$DATA_DIR/dryrun/`，不污染正式跑

**Dry run 结果**：
- Step 1 synthesize 50/50 成功，**186 秒**（DeepSeek 单次 ~15s, concurrency=4）
- Step 2 filter **kept 46/50 = 92%**，老师 prompt 质量 OK
- Step 3 preprocess 36 train + 10 val parquet 落盘
- Step 4 SFT 启动正常（FSDP / tokenizer / dataset 都加载到位），因为开始正式跑被人工停掉

### 2.5 SFT 脚本 [scripts/sft/run_sft_hotpot.sh](scripts/sft/run_sft_hotpot.sh) 调参

已落地的修改：

| 参数 | 旧 | 新 | 原因 |
|---|---|---|---|
| `data.max_length` | 2048 | **4096** | 防止 right-truncation 砍掉 `</answer>`，污染 SFT 监督目标 |
| `optim.min_lr_ratio` | (默认 0.0) | **0.1** | cosine 终值不归零，最后 ~30 step 还能学 |
| `trainer.save_freq` | 200 | **100** | 282 step 的总长度，至少存 2 次中间 ckpt |
| `trainer.seed` | (默认 1) | **2026** | 可复现 |
| `trainer.logger` | `console` | **`[console,tensorboard]`** | FSDP 多卡 console 日志糊一块，tb 利于跨阶段对照曲线 |

### 2.6 数据管线 only 脚本 [scripts/data/run_data_pipeline.sh](scripts/data/run_data_pipeline.sh)

只跑 1→2→3，跳过 SFT。用在「dry run 已通 + 现在没空闲多卡」的窗口期，让 5000 条 parquet 先落盘，等卡空了再单独跑 SFT。synthesize 资料续跑，意外中断重启即可。

---

## 3. 当前文件清单

```
scripts/
├── data/
│   ├── prompt_templates.py       # 共享 prompt + 答案正则（学生/老师分开）
│   ├── synthesize_cot.py         # DeepSeek 异步合成 CoT，async + 智能限流
│   ├── filter_cot.py             # 格式/think 长度/答案匹配/token cap 过滤
│   ├── preprocess_hotpot_sft.py  # JSONL → verl SFT parquet
│   ├── dry_run.sh                # 端到端最小冒烟（n=50, 1 卡）
│   └── run_data_pipeline.sh      # 全量数据管线，跳过 SFT（n=5000）
└── sft/
    └── run_sft_hotpot.sh         # verl SFT 启动器（FSDP + dynamic_bsz）
```

`.env` 已配置 `DEEPSEEK_API_KEY` / `TEACHER_BASE_URL` / `TEACHER_MODEL=deepseek-v4-pro`。

---

## 4. 当前在跑的任务

tmux session **`dryrun`**，日志 `/tmp/data_pipeline.log`：

```bash
N=5000 CONCURRENCY=16 bash scripts/data/run_data_pipeline.sh
```

预估：synthesize **30–80 分钟**（DeepSeek 限流 + 网络抖动可能拉到 1.5h）；filter / preprocess 秒级。

完成标志：`========== DATA PIPELINE OK ==========` + 两个 parquet 落到 `~/data/hotpotqa/sft_cot_{train,val}.parquet`。

---

## 5. SFT 关键超参 — 是否需要再调？

### 5.1 当前关键超参一览

| 类别 | 参数 | 当前值 | 重要性 |
|---|---|---|---|
| **数据** | `data.max_length` | 4096 | ⭐⭐⭐ 直接决定有没有样本被截 |
| **数据** | `data.train_batch_size` | 32 | ⭐⭐ 影响梯度噪声 / 收敛速度 |
| **数据** | `data.max_token_len_per_gpu` | 8192 | ⭐⭐⭐ dynamic_bsz 下真正决定单卡显存峰值 |
| **优化** | `optim.lr` | 2e-5 | ⭐⭐⭐⭐ 最敏感，过大破坏 instruct 对齐，过小学不动 |
| **优化** | `optim.lr_warmup_steps_ratio` | 0.03 | ⭐⭐ 影响早期稳定性 |
| **优化** | `optim.min_lr_ratio` | 0.1 | ⭐ 影响末段是否白跑 |
| **优化** | `optim.weight_decay` | 0.0 | ⭐ 短跑设 0，长跑 ≥0.01 |
| **流程** | `trainer.total_epochs` | 3 | ⭐⭐⭐ 过少欠拟合，过多 think 过拟合 |
| **流程** | `engine` / `model.use_remove_padding` | fsdp / true | ⭐ 显存 + 吞吐，非数值敏感 |

### 5.2 第一次跑通后**是否需要调参**？

**短答：先看 3 个观测信号再决定，不要盲调**。

观测信号（按优先级）：

1. **train loss 曲线**：1.5B + 3k 样本 + 3 epoch，第一个 epoch 末 loss 一般落到 **0.6–1.0** 之间。如果：
   - loss 一直在 1.5+ 不下 → **lr 偏小** 或 **数据格式不对**（先查 chat template / mask）
   - loss 第一个 epoch 末就 < 0.3，第三个 epoch 接近 0 → **过拟合**，把 epoch 改 2 或加 weight_decay 0.01
   - loss 周期性飙升 → **lr 太大** 或 **未启用 grad clip**（已设 1.0，应该没事）

2. **val loss / val 准确率**：每 100 step 测一次。如果 train loss 单调降但 val loss 在第二个 epoch 后开始抬头 → 经典过拟合，**total_epochs 改 2** 是最便宜的修复。

3. **生成质量抽查（最重要）**：训完抽 20 条 val 让模型回答，看：
   - 格式合规率：`<think>...</think><answer>...</answer>` 占比应 > 90%。如果 < 70% → 数据 think/answer 配比差，回头收紧 filter
   - 答案 EM/substring 命中率应该比 base instruct 模型高 5–15 个点。涨幅低于 5 个点 → 老师生成的 think 不够"教学性"，回到 synthesize 阶段调 teacher prompt（让推理更结构化）

### 5.3 万一要调，先动哪个？

| 现象 | 第一刀 | 第二刀 |
|---|---|---|
| OOM | `max_token_len_per_gpu` 8192 → 6144 | `max_length` 4096 → 3072 |
| 收敛慢 | `lr` 2e-5 → 3e-5 | `epochs` 3 → 4 |
| 过拟合 | `epochs` 3 → 2 | `weight_decay` 0 → 0.01 |
| 格式不稳 | 不调 SFT，回 synthesize 调 teacher prompt | 加 filter token cap |
| 答案不准 | 不调 SFT，看 filter 通过率（应 > 80%） | 提升 synthesize 数据量到 1w+ |

**核心心法**：SFT 的瓶颈 80% 在数据质量、20% 在超参。第一次跑出来如果不理想，先回头看 `sft_cot_rejects.jsonl`、抽 20 条 train 数据肉眼读 think 是否合理，再来动 lr / epoch。

### 5.4 不需要调的（已经合理）

- `optim.lr_warmup_steps_ratio=0.03`、`clip_grad=1.0`、`weight_decay=0` — 这套是 1.5B 短 SFT 的标准组合，不值得 bikeshed
- `engine=fsdp`、`use_remove_padding=true`、`use_dynamic_bsz=true` — 工程项，配错会报错而不是悄悄变差，跑通就别动
- `seed=2026` — 唯一作用是复现，不影响最终性能

---

## 6. 下一步规划

1. ⏳ 等 `run_data_pipeline.sh` 跑完（~1h），拿到 `sft_cot_train.parquet` (~4500 行)
2. 等多卡空闲（4×4090）→ `bash scripts/sft/run_sft_hotpot.sh`，预计 1.5–3h
3. 抽查 SFT 模型生成质量（见 §5.2）
4. 通过后进 Phase 3：写 reward function（复用 `prompt_templates.extract_answer`）+ rollout 配置 + GRPO trainer
