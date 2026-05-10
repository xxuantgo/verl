# Phase 3 — HotpotQA Distractor GRPO 设计与决策记录

> 对应 [PROJECT_PLAN1.md §5](PROJECT_PLAN1.md) (Phase 3) 与 [PHASE2_SFT_REPORT.md](PHASE2_SFT_REPORT.md)
> 起点 checkpoint：`checkpoints/sft_hotpot_cot_hf/global_step_500`
> 报告日期：2026-05-10
> 状态：**🟡 设计稿已定，启动前还差 7 个 kickoff action（见 §10）**

---

## 0. 这份文档存在的意义

Phase 3 的代码量不大（reward 函数 ~200 行 + launch 脚本），但**决策密度高**——每条决策错一格，下游可能多烧一周 GPU。本文逐项记录"选什么 / 为什么这么选 / 参考来源"，让未来的我们（或 review 的人）能追溯到底是别人的结论错了还是我们的实现错了。

每个决策的格式：

> **决策**：选项 X
> **为什么**：简短理由
> **参考来源**：论文/代码/issue 编号
> **反向证据**：（如有）相反结论的出处

---

## 1. 外部知识基线（影响 Phase 3 决策的核心证据）

| # | 证据 | 来源 | 对我们的意义 |
|---|------|------|-------------|
| **E1** | F1 reward 会导致 "answer avoidance" 训练崩溃——模型学会不输出 `<answer>` 而不是冒险输错；需要 action-level penalty 才能让 F1 反超 EM | Search-R1++ "How to Train Your Deep Research Agent?" (arXiv 2602.19526, 2026) | v1 默认改用 EM；v2 用 F1 但必须叠加 action penalty |
| **E2** | GRPO 在 PPO / GRPO / REINFORCE 三种主流算法里训练稳定性最差 | 同上 Search-R1++ | 必须默认开启 DAPO 风格稳定化（clip-higher、token-mean、low-var KL） |
| **E3** | verl 官方 GRPO 推荐 `kl_loss_coef=0.001`、`kl_loss_type=low_var_kl`、`loss_agg_mode=token-mean` | verl docs `algo/grpo`、`perf/best_practices` | 文档原值 `kl_loss_coef=0.01` 偏紧 10×，需要松绑 |
| **E4** | Search-R1 原版 reward 极简（只用 EM，不显式 format reward）；Tree-GRPO 加 `λ_f=0.2` format bonus；R1-Searcher stage-2 给格式违规 -2 惩罚 | Search-R1 (2503.09516)、Tree-GRPO (2509.21240)、R1-Searcher (2503.05592) | 我们的 reward 落在这条谱线的某个位置 |
| **E5** | DAPO 的 Clip-Higher / Soft Overlong / Dynamic Sampling 在 0.5B 小模型上有时反而拖累；1.5B 处于中间地带，应**逐项启用** | "DAPO Case Study with SLM" 报告 | 不一股脑全开，先开 clip-higher + token-mean，length penalty 等指标触发再开 |

---

## 2. 任务定义（先定边界）

> Phase 3 = "在 HotpotQA distractor 上做 GRPO 训练，比较 v1（EM-only）和 v2（F1+sf-F1+penalty）两版 reward"。
> **不**引入检索/工具调用——那是 Phase 4。
> **不**在评测中混入 fullwiki—— Phase 3 评测全部在 `hotpot_dev_distractor_v1.json`（7,405 条）。

**Train / dev split 边界（回答 PROJECT_PLAN1.md 中的歧义）**

- 训练集：`hotpot_train_v1.1.json`（90,447 条）
- 评测集：`hotpot_dev_distractor_v1.json`（7,405 条）
- 这两份是 HotpotQA 官方 train / dev split，**问题完全不重叠**——不是数据泄漏。
- "distractor" 指**输入格式**（gold + distractor passages 都给模型），不是某一批样本。

---

## 3. Reward 设计（最核心，见 §1 E1/E4）

### 3.1 v1：EM-only，而不是文档原写的 F1

> **决策**：v1 = EM + format_bonus + action_penalty
> **为什么**：F1 reward 会让模型偷偷学会"宁可不答也不乱答"——空 `<answer>` 也得 0，错 `<answer>` 也得 0，但前者方差小、后者方差大，GRPO 的 group-relative 机制会偏向选前者；一旦发生，组内 advantage 全为 0，梯度归零，训练卡住。EM 的离散 {0,1} 没有这个问题。
> **参考来源**：Search-R1++ §4.2（answer avoidance 现象描述）；Search-R1 原版 reward 也是 EM-only，是有标准参照的 baseline。
> **反向证据**：F1 在评测时显然比 EM 更全面——所以**评测**仍报 F1（HotpotQA 官方 metric），但**训练 reward** 用 EM。这两件事独立。

### 3.2 v2：F1 + sf-F1 + format + action penalty

> **决策**：`r = α · answer_f1 + (1-α) · sf_f1 + format_bonus + action_penalty`，default `α = 0.5`
> **为什么**：v2 的论点是"过程奖励对 multi-hop QA 有用"——把 answer 和 supporting facts 都打分，逼模型先找对证据再答。
> **参考来源**：HotpotQA 官方 `hotpot_evaluate_v1.py` 用 set-based F1 算 supporting facts；GRPO + process reward 的组合在 PRIME-RL、StepCoder 系列工作里反复验证。
> **反向证据**：纯 process（α=0）一般不收敛；纯 outcome（α=1）退化为 v1 + format。两个极端都作为 sanity check 跑（见 §3.5）。

### 3.3 三段式 reward 合成（公式）

```python
def compute_reward(answer_score, sf_score, format_state, length_tokens, alpha):
    correctness  = alpha * answer_score + (1 - alpha) * sf_score   # ∈ [0,1]
    format_bonus = +0.1 if format_state == "ok" else 0.0
    if format_state == "malformed":
        format_bonus = -0.1                                        # 标签错位/嵌套乱
    action_penalty = -0.5 if is_empty_or_hedge(answer_text) else 0.0
    length_penalty = soft_overlong(length_tokens)                  # 默认前 200 步关闭
    return correctness + format_bonus + action_penalty + length_penalty
```

> **决策**：`format_bonus` 只 ±0.1（小），`action_penalty` -0.5（大）
> **为什么**：format bonus 是奖励，过大会被 reward hacking（学一个模版后在 think 里写空话）；action penalty 是负向激励，必须显著大于 correctness 的最大值（1.0）的"放弃成本"，否则模型仍会选择不答。Search-R1++ 的实验表明 -0.5 量级足以让 F1 reward 反超 EM。
> **参考来源**：Tree-GRPO `λ_f=0.2`（format bonus 的量级参考），我们 SFT 已教过格式所以减半到 0.1；Search-R1++（action penalty 量级）。

### 3.4 Supporting facts F1（v2 用）的具体算法

> **决策**：title-level set-based F1，方案 A
> **为什么**：模型在 `<think>` 里能稳定写出 title，但要它精确到 sent_id 需要回 Phase 2 改 SFT 数据，跨 phase。title-level 已能区分"答对+证据齐"vs"答对+瞎猜"。
> **抽取方式**：从 `<think>` 段用 `re.findall` 抓**引号包裹**或**加粗**的 entity（避免子串匹配让模型把 10 个 distractor title 全列一遍 precision 拉满的退化）。
> **指标口径**：HotpotQA 官方 `update_sp` 用的 set-based precision/recall + harmonic mean，**不要**用 token-level F1。
> **参考来源**：[`hotpot_evaluate_v1.py`](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)。
> **反向证据**：方案 B（要求模型显式 `<evidence>title sent</evidence>`）是更"论文"的做法，留作 Phase 5 ablation。

### 3.5 v1/v2 ablation 一次性合并

> **决策**：v1（α=1, EM）+ v2 五个 α（0.0/0.3/0.5/0.7 + 1.0_F1）合并到 Phase 3 一起跑，**不**留到 Phase 5
> **为什么**：一次完整训练 8-12h，把 PROJECT_PLAN1.md §7.2 消融 2（process reward 权重 ablation）合并进来不到 4 天，省一次重训。α=0 的"必败 baseline"只跑 100 步证明不收敛即停。
> **参考来源**：项目自身时间预算（Phase 3 = Week 5-7 共 21 天，留 14 天给训练 + 7 天给报告）。

### 3.6 Length penalty

> **决策**：前 200 步关闭，触发后启用 soft overlong
> **公式**：`if length > 1024: -0.1 * min((length - 1024) / 256, 1.0)`
> **为什么**：SFT 起点已在分布内，length explosion 是"after fact"现象，过早惩罚会限制探索；soft 形式比硬截断稳。
> **参考来源**：DAPO 论文 §3.4 "Soft Overlong Punishment"。

### 3.7 Reward 函数实现接口

> **决策**：走 verl 的 `custom_reward_function` plugin 路径，**不**改 `verl/utils/reward_score/` 源码
> **为什么**：verl 几乎每周一次升级，改源码 = 每次 rebase 痛苦；plugin 路径是官方推荐做法。
> **参考来源**：`verl/trainer/config/reward/reward.yaml`、`verl/workers/reward_manager/naive.py:30`。
> **签名**：`compute_score(data_source, solution_str, ground_truth, extra_info) -> dict`
> **返回 dict 字段**：`score`（必有）、`answer_em`、`answer_f1`、`sf_f1`、`format_ok`、`action_penalty`、`response_len`、`reward_version`。verl 自动把 dict 所有 key 写到 wandb 前缀 `critic/rewards/*`（见 `naive.py:92-96`）。

---

## 4. 数据与 prompt 一致性

### 4.1 复用现有 prompt 模块（不要新建）

> **决策**：Phase 3 直接 `from scripts.data.prompt_templates import ...`，不创建 `prompts/hotpot_distractor.py` 副本
> **为什么**：[`scripts/data/prompt_templates.py`](scripts/data/prompt_templates.py) 已经是 Phase 2 的"单一来源"（Phase 2 SFT 数据是从这个模块的 `STUDENT_SYSTEM_PROMPT` + `render_student_user` 生成的）。如果 Phase 3 自建副本，两边任何一方改动后立刻发生 SFT 学的格式 ↔ RL prompt ↔ reward 抽取正则的三方漂移——这正是该模块当初被建立要避免的事。
> **参考来源**：[PROJECT_STATUS.md §2.1](PROJECT_STATUS.md)（项目历史已经为此抽过一次）。
> **落地动作**：Phase 3 的 `preprocess_hotpot_rl.py` 与 `src/rewards/hotpot.py` 都从这个模块 import，与 SFT preprocessing 引用同一份代码。

### 4.2 Passage 顺序 shuffle

> **决策**：在 RL 训练 parquet 生成时一次性 shuffle 每条样本的 10 个 passages，**不**在 rollout 时动态 shuffle
> **为什么**：一次性 shuffle 避免位置偏置；rollout 内固定顺序保证 group 内的 8 次采样可比（同一 prompt → 同一 group）。
> **参考来源**：常识；多数 multi-hop QA RL 工作（Search-R1、Tree-GRPO）也这么做。

### 4.3 过滤 SFT 已解样本

> **决策**：用 SFT checkpoint 在 train set 上跑一遍（每条 4 sample, T=1.0），剔除 `F1≥0.9 且 sf_f1≥0.8` 的样本
> **预期保留**：~70k / 90k（剔除 15-25%）
> **为什么**：把 GRPO 算力集中在"还有提升空间"的样本上；省 ~25% 训练时间。
> **风险**：剩余样本更难 → group 全错的概率上升 → advantage = 0 退化。**配套对策**：开启 verl 的 dynamic sampling（`actor_rollout_ref.actor.use_dynamic_bsz` + dynamic sampling），让 std=0 的 group 被 filter 掉再凑齐 batch。
> **参考来源**：DAPO 论文 §3.3 "Dynamic Sampling"；verl 已原生支持。

---

## 5. GRPO 关键超参（最终配置）

> 基础对照：[PROJECT_PLAN1.md §5.3 Task 3.3](PROJECT_PLAN1.md) 的初版表，本节是修订版。

| 超参 | 推荐值 | 与原文档差异 | 理由/出处 |
|------|--------|-------------|-----------|
| `algorithm.adv_estimator` | `grpo` | — | — |
| `algorithm.norm_adv_by_std_in_grpo` | `True` | — | GRPO 论文要求 |
| `actor.use_kl_loss` | `True` | — | GRPO 默认；KL 进 loss 不进 reward |
| `algorithm.use_kl_in_reward` | `False` | — | verl best_practices 明确说明（与上一条配套） |
| **`actor.kl_loss_coef`** | **`0.001`** | ⚠️ 文档原 `0.01` | verl docs 默认；R1-Searcher / GlobalRAG / Tree-GRPO 都用 0.001 |
| **`actor.kl_loss_type`** | **`low_var_kl`** (K3) | ➕ 新增 | DeepSeek-Math 论文推荐；多个 verl issue 显示 K3 比 K1 稳 |
| **`actor.loss_agg_mode`** | **`token-mean`** | ➕ 新增 | DAPO / Dr.GRPO 推荐；长 CoT 上比默认 `seq-mean-token-mean` 稳；verl docs 说默认值 "may be unstable" |
| `actor.optim.lr` | `1e-6` | — | Search-R1 / GlobalRAG / R1-Searcher 一致 |
| `actor.entropy_coeff` | `0.0` | — | 现代 GRPO 实践基本都关闭 |
| `actor.clip_ratio_low` | `0.2` | — | 标准 |
| **`actor.clip_ratio_high`** | **`0.28`** | ➕ 新增 | DAPO Clip-Higher（Search-R1++ 报告 GRPO 稳定性差，clip-higher 是已知最有效对策） |
| **`rollout.n`** | **`5`** | ⚠️ 文档原 `8` | Search-R1 / GlobalRAG 都是 5；4090 24GB 上 1.5B + n=8 + max_resp=1024 显存吃紧 |
| `rollout.temperature` | `1.0` | — | 标准 |
| `rollout.top_p` | `1.0` | — | 标准 |
| `data.train_batch_size` | `64`（prompt 数） | — | 即 64 prompt × 5 sample = 320 sequences/step |
| `data.max_prompt_length` | `1536` | — | distractor 上下文较长 |
| `data.max_response_length` | `1024` | — | — |
| `actor.ppo_mini_batch_size` | `32` | — | 32 prompt × 5 = 160 seq/mini-batch |
| **`actor.ppo_micro_batch_size_per_gpu`** | **`2`** | ⚠️ 文档原 `4` | 4090 24GB + 1.5B + 2560 token 上下文 + 5 rollout，micro=4 大概率 OOM |
| `model.use_remove_padding` | `True` | — | Qwen2.5 支持；显著提速 |
| `model.enable_gradient_checkpointing` | `True` | — | 24GB 卡几乎必开 |
| `actor.fsdp_config.param_offload` | `False`（先关） | — | 4×24GB 应能装下 1.5B；OOM 时再开 |
| `actor.fsdp_config.optimizer_offload` | `True` | — | AdamW 状态吃显存大头 |
| `rollout.gpu_memory_utilization` | `0.55-0.65` | — | verl perf_tuning 推荐 0.5-0.7；4090 偏保守 |
| `rollout.tensor_model_parallel_size` | `1` | — | 1.5B 不需要 TP，DP=4 即可 |
| **`trainer.total_epochs`** | **`2`** | ⚠️ 文档原 `3` | 实测 500-1000 步看到主要提升；70k/64 ≈ 1100 步/epoch，2 epoch 已 > 2000 步 |

> **最关键的两个改动**：`kl_loss_coef` 0.01 → 0.001（10× 松绑），`loss_agg_mode` 默认 → `token-mean`。这两个一起改 = 把我们的 GRPO 升级成「Dr.GRPO + DAPO 部分技巧」。

### 5.1 Dynamic Sampling 是否开

> **决策**：开，但作为可控开关
> **为什么**：剔除已解样本后，剩余样本仍可能在某些 group 里全错/全对，dynamic sampling 把 std=0 的 group filter 掉再凑齐 batch，挽回 10-30% 训练效率。
> **监控**：`filtered_reward` 指标。如果 batch filter 率 > 60%，说明任务对当前模型太难/太易，stop & inspect。
> **参考来源**：DAPO 论文；verl 原生支持。

---

## 6. 评测节奏

### 6.1 双层评测

> **决策**：
>
> - **每 25 步：mini-eval** — dev set 64 条小样本（按难度分层抽样），greedy decode，只算 EM/F1
> - **每 100 步：sub-eval** — dev set 500 条，HotpotQA 官方 metric 全套（EM/F1/sf-EM/sf-F1/Joint）
> - **训练结束：full-eval** — 7,405 条全集，仅在 best ckpt 上跑
>
> **为什么**：mini-eval ~30 秒不影响节奏，能早期发现 reward hacking（train reward 涨而 mini-eval EM 不涨）；sub-eval 给 Joint F1 可信估计；full-eval 是最终交付数字。
> **参考来源**：项目自身需求；多数 RL 训练框架都用类似双层节奏。

### 6.2 Best checkpoint 选择

> **决策**：按 dev sub-eval **Joint F1** 选 best，**不**按 train reward
> **为什么**：reward hacking 让 train reward 与下游 metric 解耦的情况在 GRPO 里很常见。Joint F1 同时反映 answer 和 sf 表现，更接近最终评测口径。

### 6.3 Early stopping

> **决策**：两个独立条件，任一触发即停
>
> 1. **性能 plateau**：连续 5 个 sub-eval 节点（500 步）Joint F1 提升 < 0.5 分
> 2. **训练稳定性失效**：`actor/kl_divergence` 单点 > 1.0，或连续 3 个 sub-eval 节点 EM 下降
>
> **为什么**：条件 2 是 Search-R1++ 报告的 GRPO collapse 早期信号，触发后**不要重启接着训**，回去看是哪个 reward 组件出问题。

---

## 7. v1 vs v2 公平对比

### 7.1 Seed 与 run 数

> **决策**：
>
> - v1（EM-only）：1 seed (42)
> - v2-α0.5（主论点）：2 seed (42, 1337)
> - v2-α0.3 / v2-α0.7：1 seed each（α ablation 用）
> - α0.0 必败 baseline：1 seed，跑 100 步即停
>
> 总成本：6 次训练，每次 8-12h，约 3-4 天。
>
> **为什么**：v1 我们已基本知道结果（接近 Search-R1 baseline），跑 1 seed 是验证 pipeline；v2-α0.5 是论点所在，2 seed 的方差（如 < 1 个 F1 点）足以撑住简历那行字。

### 7.2 训练步数对齐

> **决策**：所有 run 都跑 1500 步，**不**用"训练到收敛"
> **为什么**：「训练到收敛」在 GRPO 里没有客观定义，且 plateau 后继续训只是测耐心。1500 步基于 Search-R1 经验值（500-2000 步 plateau）。

### 7.3 Baseline 对照组

最终对比表必须有三个 baseline：

1. **Qwen2.5-1.5B-Instruct（裸 base）** — "如果不做任何 post-training"
2. **Phase 2 SFT 模型**（`global_step_500`）— "SFT 自己能涨多少"
3. **Search-R1 风格 RL**（v1 = EM-only + format）— "外面那篇论文方法在我们设置下"

这样 v2 涨的部分就是 process reward 的净贡献。

---

## 8. 工程与可重现性

### 8.1 Reward 组件分离 logging

> **决策**：`compute_score` 必须返回 dict，不返回 float
>
> ```python
> return {
>     "score": total_reward,         # 必须，verl 取这个做 RL 信号
>     "answer_em": answer_em,
>     "answer_f1": answer_f1,
>     "sf_f1": sf_f1,
>     "format_ok": float(format_ok),
>     "action_penalty": action_penalty,
>     "response_len": len_tokens,
>     "reward_version": str_label,
> }
> ```
>
> **为什么**：verl `naive.py:92-96` 自动把 dict 所有 key 写到 wandb（`critic/rewards/*` 前缀）。如果偷懒返 float，未来 reward 异常时**你不知道是哪个组件挂了**，调试时间从 1 小时 → 1 天。
> **参考来源**：`verl/workers/reward_manager/naive.py:46-122`。

### 8.2 Checkpoint 策略

| 配置 | 取值 |
|------|------|
| `trainer.save_freq` | 100 |
| `trainer.max_actor_ckpt_to_keep` | 3 |
| `best/` 目录 | 手动管理，sub-eval 触发新 best 时复制 |

1.5B 全精度 ckpt ~3GB，optimizer state ~6GB → 单 ckpt ~10GB；3 + 1 best = 40GB；6 个实验合计 ~240GB。500GB 硬盘下没问题但要监控。

### 8.3 可复现性

- 所有 run `seed=42`；v2-α0.5 双 seed (42, 1337)
- 启动训练时 git commit 当前代码 + dump config 到 `runs/<run_name>/config.yaml`
- wandb run name：`phase3_{reward_version}_a{alpha}_s{seed}_{date}`，如 `phase3_v2_a0.5_s42_20260512`

---

## 9. 失败模式预案

> 看到 X 现象就采取 Y 行动，避免凌晨被报警吵醒后乱调超参。

| 现象 | 触发阈值 | 预案 |
|------|---------|------|
| `<answer>` 标签出现率下降 | 50 步内从 ≥95% 降到 < 80% | **answer avoidance**（E1）！立即停训，提高 action_penalty 到 -1.0；或切到 EM reward |
| `actor/reward_mean` 持续 < 0.05 | 50 步 | 先看 SFT 模型 format 正确率（chat template 没对齐？）；其次看 reward 函数返回值 nan |
| `actor/kl_divergence` 单点 > 1.0 | 1 个数据点 | lr × 0.5、kl_loss_coef × 5 后从 best ckpt 续训 |
| `actor/entropy` 单调下降到 < 0.1 | 100 步内 | 熵崩塌（DAPO 报告的典型问题）；提高 `clip_ratio_high` 到 0.3 |
| `actor/response_length` 单步增长 > 5% | 持续 30 步 | length hacking；开 soft overlong penalty |
| dev F1 涨但 train reward 不涨 | 50 步以上 | Advantage 退化检查：log batch group std 分布，> 50% group std=0 → 开 dynamic sampling |
| dev sub-eval Joint F1 下降 | 连续 3 次 sub-eval | reward hacking；查 reward 组件 wandb 曲线找主导信号 |
| GPU OOM | 突发 | 优先降 `ppo_micro_batch_size_per_gpu` 到 1；其次开 `param_offload=True`；最后降 `rollout.n` 到 4 |

---

## 10. Kickoff 7 件事（开 Task 3.1 前必须完成）

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | 写 PHASE3_GRPO.md（本文档） | ✅ | 含每条决策的"为什么 + 来源" |
| 2 | 复用 [scripts/data/prompt_templates.py](scripts/data/prompt_templates.py) | ⏳ | **不要新建** `prompts/hotpot_distractor.py`；Phase 3 直接 import |
| 3 | 写 [src/rewards/hotpot.py](src/rewards/hotpot.py) | ⏳ | `compute_score` 返回 dict；通过 `extra_info["reward_version"]` + `extra_info["alpha"]` 切 v1/v2 |
| 4 | SFT format 正确率 sanity check（dev 64） | ⏳ | < 95% 就回 Phase 2 修；Phase 2 in-domain 已 100% 但 dev 是 OOD |
| 5 | 过滤 SFT 已解样本流程 → ~70k train parquet | ⏳ | 用 SFT 在 train set 跑 4-sample 推断，剔除 F1≥0.9∧sf_f1≥0.8 |
| 6 | 写 [scripts/rl/launch_grpo.sh](scripts/rl/launch_grpo.sh) | ⏳ | 写死 §5 表，预留 `REWARD_VERSION`、`ALPHA` env vars |
| 7 | 100 步 v1 EM-only smoke test | ⏳ | 任一异常**绝对不要**直接进正式训练 |

> **执行顺序**：1 → 2/3（并行）→ 6 → 4 → 5 → 7 → 进 Task 3.1 数据格式转换。

---

## 11. 关于 "Phase 3 的重点不是写 reward，而是建立 reward debug loop"

reward 代码本身只有 ~200 行，但围绕它的 7 组决策每一个错了都会让你白跑一周。**最值得花时间的事其实是 wandb 上的观测规则**——一眼看出问题在哪一层，而不是回到训练日志一行行扒：

- `critic/rewards/answer_em` vs `critic/rewards/format_ok` 应**高度相关**；解耦了说明模型在格式上偷懒
- `critic/rewards/sf_f1` 滞后于 `answer_em` 上涨，是 v2 期望中的现象（"先学会答对，再学会引证"）
- `actor/response_length` 与 `critic/rewards/answer_em` **不应**强相关；如果强相关 → length hacking
- `actor/entropy` 在前 200 步应平缓下降到 ~0.5 然后 plateau；陡降是预警

跑完 v1 形成肌肉记忆后，v2 调试速度会快好几倍——这本身是面试时一个好的故事切入点。

---

## 12. 参考资料汇总

### 论文

- **Search-R1** (Jin et al., arXiv 2503.09516) — EM-only reward baseline
- **Search-R1++** "How to Train Your Deep Research Agent?" (arXiv 2602.19526) — answer avoidance、GRPO 稳定性差的实验证据
- **Tree-GRPO** (arXiv 2509.21240) — format bonus 量级参考
- **R1-Searcher** (arXiv 2503.05592) — kl_coef=0.001、stage-2 格式惩罚
- **DAPO** (ByteDance, 2025) — Clip-Higher / Soft Overlong / Dynamic Sampling / token-mean loss agg
- **Dr.GRPO** — token-mean loss aggregation 的理论支撑
- **DeepSeek-Math** — low_var_kl (K3 estimator) 推荐
- **HotpotQA** (Yang et al., 2018) — 数据集论文

### 代码

- [verl GRPO 配置](verl/trainer/config/) — `kl_loss_coef=0.001` / `kl_loss_type=low_var_kl` 默认值
- [verl reward manager](verl/workers/reward_manager/naive.py) — dict 返回值自动 log 机制
- [verl Search-R1 reward 参考](verl/utils/reward_score/search_r1_like_qa_em.py) — EM 实现参考
- [HotpotQA 官方评测脚本](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py) — F1/sf-F1 算法

### 项目内

- [PROJECT_PLAN1.md §5](PROJECT_PLAN1.md) — Phase 3 主计划
- [PHASE2_SFT_REPORT.md](PHASE2_SFT_REPORT.md) — SFT 起点 ckpt 信息
- [scripts/data/prompt_templates.py](scripts/data/prompt_templates.py) — 共享 prompt 单一来源
