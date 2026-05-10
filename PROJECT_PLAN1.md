# 多跳问答 Agentic RL 系统 - 完整实施计划

> 本文档是项目的主计划文件。包含完整的阶段划分、子任务、验收标准、关键命令、超参数与排错指南。所有阶段的工作均围绕本文档展开，agent 工具（如 Claude Code）可基于本文档拆解 todolist 并按步骤执行。
>
> **使用建议**：Claude Code 在每个 Phase 开始时，应先阅读对应章节的"目标"、"子任务"、"验收标准"，然后再进入实施。每完成一个子任务（标记 `Task X.Y`），应回到本文档核对验收标准。

---

## 0. 项目概览

### 0.1 项目定位

- **项目标题（对外/简历）**：基于 verl 的多跳问答 Agentic RL 系统：从 SFT 冷启动到工具调用 GRPO 训练
- **项目目标（实质）**：完整走通"数据合成 → SFT 冷启动 → RLVR 训练 → 工具增强 Agentic RL → 全套 benchmark 评测"的链路，产出可在大厂后训练岗位面试中聊 30-40 分钟的旗舰项目。
- **范式**：RLVR (Reinforcement Learning with Verifiable Rewards)，**不是** RLHF。简历可两个词都用，面试时要能区分。
- **核心算法**：GRPO（Group Relative Policy Optimization）

### 0.2 最终交付物

1. **GitHub 仓库**：包含完整代码、训练脚本、数据处理脚本、评测脚本、详尽 README
2. **训练好的模型权重**（HuggingFace 上传，至少 3 个 checkpoint：SFT、GRPO-distractor、GRPO-agentic）
3. **Wandb 实验报告链接**（公开访问，所有训练曲线可查）
4. **技术报告**（Markdown 格式，含方法、实验、消融、结论）
5. **简历项目条目**（4-6 行，含具体数字）

### 0.3 时间线

总周期：**12 周**，按 4 周一个里程碑划分。

| 阶段 | 周次 | 核心产出 |
|------|------|---------|
| Phase 0 | Week 0 | 环境就绪，能跑 vLLM 推理 |
| Phase 1 | Week 1-2 | verl 跑通 GSM8K baseline，看懂 wandb |
| Phase 2 | Week 3-4 | SFT 冷启动模型 Qwen2.5-1.5B-HotpotCoT |
| Phase 3 | Week 5-7 | HotpotQA distractor GRPO 模型 + v1/v2 reward 对比 |
| Phase 4 | Week 8-10 | Agentic RL 模型 + BM25 检索服务 + 多轮 rollout |
| Phase 5 | Week 11-12 | 消融实验 + 技术报告 + 简历整理 |

### 0.4 简历呈现目标

最终简历项目条目目标形态（可作为 Phase 5 的"目标读者"）：

> **多跳问答 Agentic RL 系统**（verl / Qwen2.5 / 4×RTX 4090）
> - 完整 RL 后训练 pipeline：CoT 数据合成与 SFT 冷启动、verl 框架下 GRPO 训练、6 个开放域 QA benchmark 端到端评测
> - 从 HotpotQA distractor 起步扩展到 fullwiki agentic 场景，自研 BM25 检索服务并改造 verl 多轮 rollout，使模型可在推理过程中调用 `<search>` 工具
> - 在 Qwen2.5-1.5B 上将 HotpotQA Joint F1 从 SFT 基线 [X] 提升至 RL 后 [Y]，NQ EM 从 [X] 提升至 [Y]
> - 消融验证：supporting_facts 作为 process reward 比纯 outcome reward 收敛快 ~30%；冷启动 SFT 数据量超过 3k 后边际收益递减

---

## 1. 硬件与环境

### 1.1 硬件配置

- **GPU**：4-8 × RTX 4090（24GB / 卡）
- **总显存**：96GB（4卡）或 192GB（8卡）
- **CPU/内存**：建议 ≥ 32 核 / 256GB RAM（rollout 和评测时 CPU 较吃）
- **磁盘**：≥ 500GB SSD
  - 模型权重：~10GB（多个 checkpoint）
  - 数据集：~30GB（HotpotQA + NQ + 维基百科 dump）
  - 训练日志：~20GB
  - vLLM KV cache 等临时文件：~50GB

### 1.2 模型选择决策

**主线全程使用 Qwen2.5-1.5B-Instruct**，理由：

- 4 卡 GPU 下迭代速度最快（单步训练 ~30 秒）
- 显存有余量，可以开较大 batch size，rollout 阶段可以采样 8 条/prompt
- SFT、GRPO、Agentic RL 三个阶段都能在合理时间内完成
- 1.5B 在 HotpotQA 这类任务上有足够的可提升空间，能看出训练效果

**可选 scaling 实验**：在 Phase 5 如有时间和 8 卡可用，复制 Phase 3 的训练脚本到 Qwen2.5-3B，得到对比数据，简历中可写"验证了方法在 1.5B / 3B 双尺度上的有效性"。

**不使用 7B 的原因**：4 卡显存不够，8 卡也很紧；单次实验时间过长导致迭代不动；与本项目"算法理解 + 工程闭环"的定位不匹配。

### 1.3 软件栈

| 组件 | 推荐版本 | 备注 |
|------|---------|------|
| Python | 3.10 | verl 兼容性最好 |
| CUDA | 12.1+ | 4090 + Hopper 架构友好 |
| PyTorch | 2.4+ | verl 当前主推 |
| verl | latest main | 主训练框架 |
| vLLM | 0.6.3+ | rollout 引擎 |
| transformers | 4.45+ | |
| wandb | 0.18+ | 实验跟踪 |
| pyserini | 0.22+ | BM25 检索（Phase 4） |
| TRL | 0.11+ | SFT 阶段使用 |
| flash-attn | 2.6+ | 加速训练 |

---

## 2. Phase 0：环境搭建（Week 0）

### 2.1 目标

搭建可复用的训练环境，能跑通最小化 vLLM 推理与 verl 单步训练，验证硬件无误。

### 2.2 子任务

#### Task 0.1：创建 conda 环境

```bash
conda create -n verl python=3.10 -y
conda activate verl
```

#### Task 0.2：安装 PyTorch

```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Task 0.3：安装 verl

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
pip install vllm==0.6.3
pip install flash-attn --no-build-isolation
```

#### Task 0.4：安装辅助库

```bash
pip install wandb datasets accelerate trl pyserini
```

#### Task 0.5：下载基础模型

```bash
# 用 HuggingFace 缓存
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct  # Phase 5 备用
```

#### Task 0.6：vLLM 健康检查

写一个最小推理脚本 `scripts/test_vllm.py`，加载 Qwen2.5-1.5B-Instruct，让它回答一个简单问题，确保 vLLM 能正常启动并推理。

#### Task 0.7：wandb 配置

```bash
wandb login
# 创建项目 verl-hotpot-rl
```

### 2.3 验收标准

- [ ] `nvidia-smi` 能看到全部 GPU
- [ ] vLLM 推理脚本能正常输出
- [ ] verl 仓库可 import，无依赖错误
- [ ] wandb 能成功上报一个测试 run

### 2.4 常见问题

- **flash-attn 编译失败**：检查 CUDA 版本和 PyTorch 版本是否匹配；用预编译 wheel 安装
- **vLLM OOM**：降低 `--gpu-memory-utilization` 到 0.85
- **HuggingFace 下载慢**：设置 `HF_ENDPOINT=https://hf-mirror.com`

---

## 3. Phase 1：Baseline 复现与 verl 学习（Week 1-2）

### 3.1 总目标

**不动手改代码，先看懂 verl**。通过复现一个最简单的 GSM8K + GRPO 训练，理解 verl 的代码结构、训练循环、wandb 指标含义。这一步不要急着改自己的项目，否则后面会一直踩坑。

### 3.2 子目标

1. 看懂 verl 的代码结构，能说出每个核心模块的职责
2. 跑通 GSM8K + Qwen2.5-1.5B + GRPO 的官方示例
3. 看着 wandb 曲线能解释每条曲线在做什么、什么样算正常

### 3.3 子任务

#### Task 1.1：verl 代码阅读清单（建议 3-4 天）

按以下顺序阅读，**做笔记**记录每个文件的作用：

**优先级 A（必读）**：
- `verl/trainer/ppo/ray_trainer.py` - 训练主循环，理解 actor / ref / rollout 的协调
- `verl/trainer/ppo/core_algos.py` - GRPO / PPO 算法核心，loss 计算、advantage 计算
- `verl/utils/reward_score/` - 内置的 reward 函数（GSM8K、math 等）
- `examples/grpo_trainer/run_qwen2_5-1.5b_gsm8k.sh` - GRPO 启动脚本

**优先级 B（看懂大致流程即可）**：
- `verl/workers/actor/dp_actor.py` - actor 训练的具体实现
- `verl/workers/rollout/vllm_rollout/` - vLLM rollout 集成
- `verl/utils/dataset/rl_dataset.py` - RL 数据加载

**输出物**：一份 `notes/verl_architecture.md`，包含：
- verl 数据流图（prompt → rollout → reward → advantage → loss → update）
- 每个核心文件的一句话职责
- GRPO loss 公式与代码对应关系

#### Task 1.2：GSM8K Baseline 复现

```bash
# 进入 verl 目录
cd verl
# 用官方脚本启动（修改 GPU 数为本机配置）
bash examples/grpo_trainer/run_qwen2_5-1.5b_gsm8k.sh
```

**预期训练时间**：4×4090 上约 6-10 小时跑完官方默认步数。

**预期最终性能**：GSM8K test accuracy 从 base ~50% 提升到 ~75-80%。

#### Task 1.3：Wandb 曲线解读笔记

跑完后，写一份 `notes/wandb_metrics_explained.md`，解释以下每条曲线的含义、正常形态、异常形态：

**核心指标**：
- `actor/reward_mean`：平均 reward。**正常**：单调上升，最终趋于平稳。**异常**：长时间不动→reward 函数有 bug；剧烈震荡→学习率过大
- `actor/response_length`：模型输出长度。**正常**：上升后稳定。**异常**：持续单调上升伴随 reward 上升 → length hacking
- `actor/kl_divergence`：与 ref policy 的 KL 散度。**正常**：稳定在 0.01-0.5。**异常**：突然暴涨 → 模型偏离 ref 太远
- `actor/pg_loss`：policy gradient loss。**正常**：负值，缓慢趋近 0。**异常**：剧烈抖动 → advantage 估计有问题
- `actor/entropy`：策略熵。**正常**：缓慢下降（exploration → exploitation）。**异常**：过快下降 → 过早收敛
- `critic/value_loss`：（GRPO 没有 critic 可忽略）

### 3.4 验收标准

- [ ] `notes/verl_architecture.md` 完成，能口述 verl 数据流
- [ ] GSM8K baseline 训练完成，准确率达到预期
- [ ] `notes/wandb_metrics_explained.md` 完成
- [ ] 能回答："为什么 GRPO 不需要 critic 网络？" "advantage 在 GRPO 里怎么算？"

### 3.5 常见问题

- **训练 OOM**：降低 `actor.ppo_mini_batch_size` 和 `rollout.n`
- **训练卡住**：通常是 ray worker 死锁，重启即可
- **wandb 不上报**：检查 `trainer.logger=['console','wandb']` 配置

---

## 4. Phase 2：SFT 冷启动副项目（Week 3-4）

### 4.1 总目标

构造一个带"思维链"的 SFT 数据集，在 Qwen2.5-1.5B-Instruct 上做 SFT，得到 `Qwen2.5-1.5B-HotpotCoT`。这个模型同时是 SFT 副项目的产出 与 Phase 3 的训练起点。

### 4.2 子目标

1. 学会用 API 合成训练数据（这是大厂常见任务）
2. 掌握 SFT 训练的关键技术（loss masking、prompt 模板、early stopping）
3. 让模型学会 `<think>...</think><answer>...</answer>` 的输出格式
4. 产出可作为 RL 起点的 checkpoint

### 4.3 子任务

#### Task 2.1：HotpotQA 原始数据准备

```bash
# 下载 HotpotQA 训练集和 dev set
mkdir -p data/hotpotqa
cd data/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
```

**数据规格**：
- 训练集：90,447 条
- dev distractor：7,405 条
- dev fullwiki：7,405 条

#### Task 2.2：CoT 数据合成

**目标**：从训练集采样 5000 条，调 API 为每条生成"先推理再回答"的链式响应。

**API 选择**：
- 优先：DeepSeek-V3 / DeepSeek-R1（便宜、中文友好、推理强）
- 备选：GPT-4o-mini / Claude Haiku

**Prompt 模板**（写到 `scripts/data/synthesize_cot.py`）：
```
Given the following question and supporting passages, generate a step-by-step reasoning process and final answer.

Question: {question}
Passages:
{context}

Format your response strictly as:
<think>
Step 1: [identify what info is needed]
Step 2: [extract relevant facts from passages]
Step 3: [combine facts to answer]
</think>
<answer>{short answer}</answer>

Gold answer (must match): {gold_answer}
Supporting facts (must reference): {supporting_facts}
```

**关键工程点**：
- **质量过滤**：合成后用规则过滤——`<think>` 段必须 ≥ 30 字符；`<answer>` 必须包含 gold answer 的关键词；总长度 ≤ 1024 token
- **预期产出**：5000 条原始 → 过滤后 3000-4000 条高质量 SFT 数据
- **API 成本预估**：DeepSeek-V3 大约 $5-10

**输出文件**：`data/hotpotqa/sft_cot_train.jsonl`

每行格式：
```json
{
  "instruction": "Question: ... \n\nPassages: ...",
  "output": "<think>...</think>\n<answer>...</answer>",
  "raw_question": "...",
  "raw_answer": "...",
  "supporting_facts": [...]
}
```

#### Task 2.3：SFT 训练脚本

**框架选择**：使用 TRL 的 `SFTTrainer`（轻量、上手快）。或直接用 verl 自带的 SFT 模块。

**关键技术点**：
- **Loss masking**：只对 `<think>...</answer>` 部分算 loss，instruction 部分不算
- **学习率**：2e-5（标准 SFT 学习率）
- **batch size**：micro 4，gradient_accumulation 8 → effective 32
- **epochs**：3
- **Max length**：2048
- **LoRA vs Full**：用 **full fine-tuning**（1.5B 模型 4 卡完全够，且后续 RL 阶段也是 full fine-tuning，保持一致）

**训练脚本骨架** `scripts/sft/train_sft.py`：
```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
dataset = load_dataset("json", data_files="data/hotpotqa/sft_cot_train.jsonl")

config = SFTConfig(
    output_dir="checkpoints/sft_hotpot_cot",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    report_to="wandb",
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"], args=config,
    dataset_text_field="text",  # 需要预处理把 instruction+output 合并
)
trainer.train()
```

**预期训练时间**：3 epochs × 3-4k 数据，4×4090 上约 2-3 小时。

#### Task 2.4：SFT 模型评测

在 dev set 100 条样本上，让 SFT 模型直接生成回答，统计：
- **格式正确率**：输出是否包含 `<think>` 和 `<answer>` 标签
- **Answer F1**（与 gold 答案的 F1）
- **Joint F1**（如果模型在 think 中提到了支撑文档标题）

**预期数值**：
- 格式正确率 ≥ 95%
- Answer F1 在 30-38 之间
- Joint F1 在 20-28 之间

> ⚠️ 这些数字是 SFT 后未经 RL 的基线，会作为 Phase 3 的起点。

### 4.4 验收标准

- [ ] `data/hotpotqa/sft_cot_train.jsonl` 包含 ≥3000 条高质量样本
- [ ] SFT 训练正常完成，loss 曲线下降平稳
- [ ] `checkpoints/sft_hotpot_cot/` 包含可加载的 checkpoint
- [ ] dev set 评测格式正确率 ≥ 95%，Answer F1 在 30-38

### 4.5 常见问题

- **格式正确率低**：检查 SFT 数据中 `<think>` 标签是否一致；增加 epochs 到 5
- **过拟合**：训练 loss 持续下降但 dev 性能不涨 → 减小 epochs 或加入更多数据
- **API 调用超时**：分批合成、加 retry 逻辑、用 async 并发

---

## 5. Phase 3：HotpotQA Distractor GRPO（Week 5-7）

### 5.1 总目标

以 SFT 模型为起点，在 HotpotQA distractor setting 上做 GRPO 训练，对比两版 reward 函数的效果，学习 RL 训练的核心技术。

### 5.2 子目标

1. 跑通完整 GRPO 训练循环，能驾驭 reward 设计、超参调优
2. 完成 v1 (outcome reward only) vs v2 (outcome + process reward) 对比实验
3. 在 HotpotQA dev distractor 上将 Joint F1 提升 8-15 个点

### 5.3 启动前必须讨论确认的关键问题（Phase 3 Kickoff Checklist）

> 这一节是 Phase 3 真正的"重点"。在写第一行 reward 代码前，下面 7 组问题应当**先讨论、定稿、记录到 `notes/phase3_design_decisions.md`**。每组问题都列出常见选项与权衡，方便对话时勾选。
>
> **注意 setting 边界**：Phase 3 训练与评测**全部使用 HotpotQA distractor**（train split 训练、dev split 评测），不引入检索/工具调用——那些是 Phase 4 的任务。本阶段的目标是"在已有 10 篇 passages 中做多跳推理"，不是"找到证据"。

#### 讨论组 1：Reward 函数细节（最核心）

这是 Phase 3 最容易踩坑、也最值得花时间打磨的部分。需要明确：

1. **Format reward 的严格度**
   - 选项 A（硬 0）：缺 `<answer>` 标签直接 reward = 0
   - 选项 B（软惩罚）：format 正确给 +0.1 bonus，错误不归零只扣分
   - 选项 C（先 warmup）：训练前 100 步只用 format reward 让模型稳住格式，再接入 F1
   - 风险：选 A 时，如果 SFT 模型格式正确率不够（< 95%），整个 batch 大量样本 reward = 0，advantage 退化为 0，训练根本动不起来

2. **Answer F1 的具体口径**
   - 用 HotpotQA 官方 `hotpot_evaluate_v1.py` 的 normalize（去 stopwords、lower、去标点）还是自己写？→ **必须用官方口径**，否则评测数字不可比
   - 是否拆 EM 与 F1 做 reward？目前默认只用 F1（EM 太稀疏）

3. **Process reward（v2）：从 `<think>` 中提取 supporting facts 的方式**
   - Supporting facts gold 格式是 `[(title, sent_id), ...]`，但模型在 `<think>` 里只可能写 title（很难精确到句子号）
   - 选项 A：只匹配 title，sf_f1 退化为 title-level F1
   - 选项 B：要求模型显式写 `<evidence>Title_A sent 0; Title_B sent 2</evidence>` 这种结构 → 但 SFT 数据没这样训过，需要先回 Phase 2 改 prompt 模板
   - **建议先选 A**，简单可行；选 B 是更好的论文做法但会跨 phase

4. **v1 vs v2 的权重设置**
   - 当前文档写的是 `0.5 * answer_f1 + 0.5 * sf_f1`
   - 是否做 reward 权重的 ablation（α ∈ {0.3, 0.5, 0.7}）？这其实就是 [Phase 5 消融 2](#消融-2process-reward-权重-ablation)，可以提前到 Phase 3 一起做，省一次完整训练

5. **Length penalty 的形式**
   - 硬阈值（>1024 token 扣 0.1）vs 软衰减（与 length 平滑相关）
   - 是否在生成早期就开启？建议前 100 步关闭，避免和 format reward 互相干扰

#### 讨论组 2：数据格式与 prompt 一致性

1. **Prompt 模板必须与 Phase 2 SFT 完全对齐**——SFT 学的是格式 A，RL 用格式 B 会让 SFT 冷启动的优势消失。需要在动手前**逐字符对照** [Task 2.2](#task-22cot-数据合成) 的模板与 [Task 3.1](#task-31数据格式转换) 的模板
2. **Passage 顺序**：每个 epoch 是否对 distractor 顺序做 shuffle？(防止模型记住"gold 总是出现在第 1、3 位"这种伪信号)
3. **是否过滤 SFT 已经能解决的样本**：可以先用 SFT 模型在 train set 上跑一遍，把 F1 ≥ 0.9 的样本剔除（约 20-30%），把 GRPO 的训练算力集中在"还有提升空间"的样本上。代价是工程多一步，但能省 ~25% 训练时间

#### 讨论组 3：GRPO 关键超参

文档 [5.3 Task 3.3](#task-33grpo-训练配置) 已经给了一组参考值，但有几个需要专门讨论：

1. **`rollout.n`（group size）**：8 是 GRPO 经典设置，但 4090 上 1.5B + n=8 + max_response=1024 显存吃紧。如果 OOM，先降到 6 而不是降 batch
2. **KL 系数 `kl_loss_coef`**：0.01 偏松 / 0.05 偏紧。SFT 起点质量好的话先用 0.01，发现 KL 暴涨再提
3. **Loss 形态**：`actor.use_kl_loss=True`（KL 进 loss）vs KL 进 reward shaping。verl 默认前者，保持默认即可
4. **Advantage normalization**：`algorithm.norm_adv_by_std_in_grpo` 默认 True，必须开（GRPO 论文要求）

#### 讨论组 4：评测频率与 early stopping

1. **训练中评测**：每 50 步在 dev 100 条小样本上评，还是每 100 步？小评测会拖慢训练但能更早发现 reward hacking
2. **完整 dev set 评测**（7,405 条）：只在训练结束后跑一次，还是中途也跑？建议只跑一次，但保留每 200 步的 checkpoint 以备回溯
3. **Early stopping 触发条件**：连续 N 个评测点 dev F1 不涨就停？还是固定跑满 500 步？

#### 讨论组 5：v1 vs v2 公平对比

1. **Seed 控制**：v1 与 v2 必须同 seed、同 batch 顺序、同初始化。建议都跑 2 个 seed 取均值（成本翻倍但结论才稳）
2. **训练步数对齐**：用相同的步数比较，还是用相同的"训练到收敛"步数比较？前者公平，后者更代表"模型真实能力"
3. **是否需要第三个对照组**：纯 SFT（不做 RL）作为 baseline 必须有，已在 [Task 3.6](#task-36完整-dev-set-评测) 列出

#### 讨论组 6：工程与可重现性

1. **Reward 函数挂载方式**：写在 `verl/utils/reward_score/hotpotqa.py` 里（侵入式）vs 通过 verl 的 `custom_reward_function` 配置（plugin 式）→ **强烈建议 plugin 式**，方便后续升级 verl
2. **Reward 组件分别 log**：在 `compute_score` 里把 `answer_f1`、`sf_f1`、`format_ok`、`length` 都 return 出去并单独写到 wandb，否则 reward 一旦不对你不知道是哪个组件挂了
3. **Checkpoint 策略**：`save_freq` 设多少？磁盘只有 500GB，1.5B 模型每 ckpt 约 3GB，建议每 100 步存且只保留最近 3 个 + best

#### 讨论组 7：失败模式预案

在跑训练前先想好"看到 X 现象就采取 Y 行动"，避免在凌晨被报警吵醒后乱调超参：

| 现象 | 触发阈值 | 预案 |
|------|---------|------|
| reward < 0.05 持续 50 步 | — | 停训，回看 SFT 模型 format 正确率，可能要先 warmup format |
| KL > 1.0 | 单点 | lr × 0.5、kl_coef × 5 后续训 |
| response_length 单步增长 > 5% 持续 30 步 | — | 立即开启 length penalty 或加大已有 penalty |
| dev F1 涨但 train reward 不涨 | 50 步以上 | 检查 reward 计算是否正确（advantage 退化） |

#### Phase 3 重点小结

> Phase 3 的"重点"不是单纯实现 reward，而是 **"reward 设计 + 训练动力学诊断"**。reward 代码本身只有 ~50 行，但围绕它的 7 组决策每一个错了都会让你白跑一周。
>
> **建议执行顺序**：先和我（或人类项目持有者）走完上面 7 组讨论 → 把决策写进 `notes/phase3_design_decisions.md` → 再开始 Task 3.1 数据格式转换。

---

### 5.4 子任务（Task List）

#### Task 3.1：数据格式转换

把 HotpotQA distractor 训练集转成 verl 要求的格式：
```python
{
  "prompt": "Question: ...\n\nPassages:\n[1] Title_A: ...\n[2] Title_B: ...\n\nThink step by step then answer.",
  "ground_truth": {
    "answer": "London",
    "supporting_facts": [["Title_A", 0], ["Title_B", 2]]
  }
}
```

**输出**：`data/hotpotqa/rl_distractor_train.parquet`

#### Task 3.2：实现 Reward 函数

**位置**：`verl/utils/reward_score/hotpotqa.py`（或自己写 plugin）

**Reward v1（仅 answer F1）**：
```python
def compute_reward_v1(solution_str: str, ground_truth: dict) -> float:
    # 1. 从 solution_str 解析出 <answer>...</answer> 内容
    pred_answer = extract_answer(solution_str)
    if pred_answer is None:
        return 0.0  # 格式错误直接 0 分
    # 2. 计算 F1
    return compute_f1(pred_answer, ground_truth["answer"])
```

**Reward v2（answer F1 + supporting facts F1）**：
```python
def compute_reward_v2(solution_str: str, ground_truth: dict) -> float:
    pred_answer = extract_answer(solution_str)
    pred_sf = extract_supporting_facts(solution_str)  # 从 <think> 中提取提到的标题
    if pred_answer is None:
        return 0.0
    answer_f1 = compute_f1(pred_answer, ground_truth["answer"])
    sf_f1 = compute_sf_f1(pred_sf, ground_truth["supporting_facts"])
    return 0.5 * answer_f1 + 0.5 * sf_f1
```

**重要：format reward 的处理**：
- 输出无 `<answer>` 标签：reward = 0
- 输出长度超限（>1024 token）：reward -= 0.1（length penalty）

#### Task 3.3：GRPO 训练配置

**写训练脚本** `scripts/rl/train_grpo_distractor_v1.sh`：

关键超参（参考值）：
| 超参 | v1/v2 通用值 | 说明 |
|------|--------|------|
| `actor.model.path` | `checkpoints/sft_hotpot_cot` | SFT 模型作为起点 |
| `actor.optim.lr` | 1e-6 | RL 阶段学习率比 SFT 小一个数量级 |
| `actor.ppo_mini_batch_size` | 32 | |
| `actor.ppo_micro_batch_size` | 4 | |
| `data.train_batch_size` | 64 | 每个 step 的 prompt 数 |
| `actor.use_kl_loss` | True | |
| `actor.kl_loss_coef` | 0.01 | KL 系数 |
| `algorithm.adv_estimator` | grpo | 使用 GRPO 算法 |
| `actor_rollout_ref.rollout.n` | 8 | 每个 prompt 采样 8 次（GRPO 必需） |
| `actor_rollout_ref.rollout.temperature` | 1.0 | 采样温度 |
| `data.max_prompt_length` | 1536 | distractor 上下文较长 |
| `data.max_response_length` | 1024 | |
| `trainer.total_epochs` | 3 | |

**预期训练步数**：`(90k / 64) × 3 ≈ 4200 步`，但通常 500-1000 步就能看到主要提升。建议先跑 500 步看趋势。

**预期单步时间**：4×4090 上约 1-2 分钟，500 步约 8-16 小时。

#### Task 3.4：v1 训练 + 监控

跑 `train_grpo_distractor_v1.sh`，全程监控：
- `actor/reward_mean` 应单调上升
- `actor/response_length` 不应剧烈上涨
- `actor/kl_divergence` 应稳定 < 0.5
- 每 50 步在 dev set 100 条样本上评测一次（自定义 callback）

#### Task 3.5：v2 训练 + 监控

复制 v1 脚本，将 reward 函数切到 v2，重新训练。其他超参严格保持一致（控制变量）。

#### Task 3.6：完整 dev set 评测

两个模型分别在 HotpotQA dev distractor 全集 (7,405 条) 上跑评测，记录：
- Answer EM
- Answer F1
- Supporting Facts EM
- Supporting Facts F1
- **Joint EM**（answer EM × sf EM）
- **Joint F1**（answer F1 × sf F1）

**预期数值**（参考范围，实际以你跑出的为准）：

| 模型 | Answer F1 | Joint F1 |
|------|-----------|----------|
| Qwen2.5-1.5B-Instruct（无 SFT 无 RL） | 20-28 | 12-18 |
| Phase 2 SFT 模型 | 30-38 | 20-28 |
| GRPO v1 | 40-48 | 28-36 |
| GRPO v2 | 41-49 | 32-42 |

**关键观察**：v2 的 Joint F1 应明显高于 v1（process reward 帮助找到正确证据），但 Answer F1 可能差别不大。这就是简历上的消融结论。

#### Task 3.7：训练动力学分析报告

写 `reports/phase3_dynamics.md`，记录：
- v1 vs v2 的训练曲线对比图
- 收敛速度对比（达到 90% 最终 reward 的步数）
- 任何观察到的异常和调试过程
- length hacking 是否发生、如何处理

### 5.5 验收标准

- [ ] v1 训练正常完成，dev Answer F1 ≥ 38
- [ ] v2 训练正常完成，dev Joint F1 ≥ 28，且明显高于 v1
- [ ] `reports/phase3_dynamics.md` 完成
- [ ] 能回答："你的 reward 怎么设计的，为什么这么设计？" "v1 和 v2 的差别是什么？"
- [ ] 至少处理过一次训练异常（length hacking 或 KL 爆炸）并解决

### 5.6 常见问题与排错

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| reward 一直接近 0 | 模型不输出 `<answer>` 标签 | 检查 SFT 模型格式正确率；可加格式 reward bonus |
| reward 上涨但 dev 性能不涨 | reward hacking | 检查 reward 函数；做错题分析 |
| response_length 单调上升 | length hacking | 加 `length_penalty` |
| KL 暴涨（>1.0） | lr 太大或 kl_coef 太小 | lr 降到 5e-7；kl_coef 提到 0.05 |
| 训练 OOM | batch 太大或上下文太长 | 降 micro_batch_size；上下文截断 |
| pg_loss 剧烈震荡 | advantage 数值不稳定 | 开启 advantage normalization |

---

## 6. Phase 4：Agentic RL with Search Tool（Week 8-10）

### 6.1 总目标

把 Phase 3 的 distractor 模型升级为可以**主动调用搜索工具**的 agent，从给定段落场景升级到 fullwiki 开放域场景。这是项目的"高潮"，也是简历最亮眼的部分。

### 6.2 子目标

1. 搭建本地 BM25 检索服务（基于维基百科 abstract）
2. 改造 verl 的 rollout，支持模型在生成中调用 `<search>` 工具
3. 跑通 Agentic GRPO 训练
4. 在 6 个开放域 QA benchmark 上完整评测

### 6.3 子任务

#### Task 4.1：搭建维基百科 BM25 检索服务

**数据源**：使用 KILT 项目提供的维基百科 dump（约 6GB，5.9M 篇 abstract）
- 下载链接：`http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json`

**索引构建**（用 pyserini）：
```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/wiki_corpus_jsonl \
  --index indexes/wiki_bm25 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 --storePositions --storeDocvectors --storeRaw
```

**索引时间**：~2-3 小时（CPU 任务，与 GPU 无关）。

**HTTP 服务封装**（用 FastAPI），`services/bm25_server.py`：
```python
from fastapi import FastAPI
from pyserini.search.lucene import LuceneSearcher

app = FastAPI()
searcher = LuceneSearcher("indexes/wiki_bm25")
searcher.set_bm25(k1=0.9, b=0.4)

@app.post("/search")
def search(query: str, top_k: int = 5):
    hits = searcher.search(query, k=top_k)
    return [{"title": ..., "text": ...} for h in hits]
```

启动：`uvicorn services.bm25_server:app --port 8765 --workers 4`

**性能要求**：QPS ≥ 50（rollout 阶段会高并发调用）。

#### Task 4.2：改造 verl 多轮 rollout

**参考**：`verl/examples/sglang_multiturn/` 模板

**核心改造点**：rollout 时检测 `<search>query</search>` 标签，若出现：
1. 暂停生成
2. 调用 BM25 服务获取结果
3. 把结果包装为 `<information>...</information>` 注入对话
4. 模型继续生成

**实现位置**：`verl/workers/rollout/vllm_rollout/multiturn_search.py`（新文件）

**关键代码骨架**：
```python
class SearchAugmentedRollout:
    MAX_TURNS = 4  # 最多 4 轮搜索
    
    def generate_with_tools(self, prompt, model):
        history = prompt
        for turn in range(self.MAX_TURNS):
            output = model.generate(history, stop=["</search>"])
            history += output
            if "<search>" in output and "</search>" in output:
                query = extract_search_query(output)
                results = call_bm25_service(query, top_k=3)
                history += f"\n<information>\n{format_results(results)}\n</information>\n"
            elif "<answer>" in output:
                break
        return history
```

#### Task 4.3：构造 Agentic 训练数据

**数据来源**：
- HotpotQA fullwiki train（重新格式化）
- NQ-open train
- 2WikiMultiHopQA train

**Prompt 模板**：
```
Answer the question by searching for information when needed.

Available actions:
- <think>your reasoning</think>
- <search>search query</search>  # to look up information
- <answer>final answer</answer>  # when you have enough info

Question: {question}
```

**初始数据规模**：每个数据集采 10k 条混合，共 30k 训练样本。

#### Task 4.4：Agentic GRPO 训练

**重点超参变更**：
- `max_response_length`：从 1024 → 2048（多轮交互需要更长）
- `rollout.n`：从 8 → 5（每条采样减少，因为每次 rollout 更耗时）
- `learning_rate`：1e-6 → 5e-7（更稳）

**预期训练时间**：4×4090 上每步 5-10 分钟（rollout 因为多轮调用变慢），1000 步约 4-7 天。

> ⚠️ 这是项目最耗时的阶段，建议先用 5k 数据小规模验证 24 小时，确认无 bug 再放大。

#### Task 4.5：6-数据集 Benchmark 评测

**评测集**：
| 数据集 | 类型 | 测试集大小 |
|--------|------|-----------|
| NQ-open | 单跳 | 3,610 |
| TriviaQA | 单跳 | 11,313 (sample 2k) |
| HotpotQA | 多跳 | 7,405 |
| 2WikiMultiHop | 多跳 | 12,576 (sample 2k) |
| MuSiQue | 多跳 | 2,417 |
| Bamboogle | 多跳 | 125 |

**评测指标**：每个数据集报 EM 和 F1。

**对比模型**：
1. SFT-only（Phase 2 产出）
2. Phase 3 GRPO-distractor（无搜索工具）
3. Phase 4 Agentic-GRPO（最终模型）

**预期数值**（参考 Search-R1 论文 1.5-3B 范围）：

| 数据集 | SFT only EM | Agentic-GRPO EM |
|--------|-------------|----------------|
| NQ | 18-24 | 30-38 |
| TriviaQA | 35-45 | 50-60 |
| HotpotQA | 22-28 | 32-42 |
| 2WikiMultiHop | 18-25 | 28-38 |
| MuSiQue | 5-10 | 10-18 |
| Bamboogle | 15-22 | 28-42 |

### 6.4 验收标准

- [ ] BM25 服务稳定运行，QPS ≥ 50
- [ ] verl rollout 改造完成，能正确处理 `<search>` 调用
- [ ] Agentic-GRPO 训练完成，loss 与 reward 曲线正常
- [ ] 6-数据集 benchmark 全部完成
- [ ] 在 NQ 上 EM 从 SFT 基线提升 ≥ 10 个点
- [ ] 在 HotpotQA 上 Joint F1 比 Phase 3 distractor 模型再提升

### 6.5 风险与备选

- **如果改造 rollout 太复杂**：可降级方案——使用 verl 现有的 sglang_multiturn 框架，工具调用通过该框架的 hook 注入
- **如果训练时间超预期**：减少训练数据规模到 10k；用 LoRA fine-tuning 加速

---

## 7. Phase 5：消融、报告、简历（Week 11-12）

### 7.1 总目标

通过精心设计的消融实验提升项目深度，整理成可对外展示的完整产出。

### 7.2 消融实验设计

至少完成以下 3 个，每个都是简历可以单独写一句话的"小创新"：

#### 消融 1：Cold Start SFT 数据规模 ablation

固定其他所有变量，SFT 数据从 1k / 3k / 5k / 10k 各跑一遍，看 final RL 性能变化。

**预期发现**：3k 后边际收益递减（写到简历）。

#### 消融 2：Process Reward 权重 ablation

Reward = α × answer_f1 + (1-α) × sf_f1，α 取 0.0 / 0.3 / 0.5 / 0.7 / 1.0。

**预期发现**：α=0.5 综合最优；α=0.0（纯 process）不收敛；α=1.0（纯 outcome）Joint F1 最低。

#### 消融 3：Search 调用次数限制

最多允许调用 1 / 3 / 5 / 无限 次。

**预期发现**：3-5 次最优；无限次时模型会做不必要的搜索浪费 reward。

#### 可选消融 4（如有 8 卡和时间）：模型尺度 scaling

复制 Phase 4 训练到 Qwen2.5-3B，对比 1.5B 和 3B 的最终性能。

### 7.3 GitHub 仓库整理

仓库结构（建议）：
```
hotpotqa-agentic-rl/
├── README.md              # 主文档（含 figures、results table）
├── PROJECT_PLAN.md        # 本文档
├── reports/
│   ├── technical_report.md
│   ├── phase3_dynamics.md
│   └── ablations.md
├── scripts/
│   ├── data/              # 数据合成与处理
│   ├── sft/               # SFT 脚本
│   ├── rl/                # GRPO 训练脚本
│   ├── eval/              # 评测脚本
│   └── test_vllm.py
├── services/
│   └── bm25_server.py     # BM25 检索服务
├── verl_patches/          # 对 verl 的改造
│   └── multiturn_search.py
├── data/                  # 数据（用 .gitignore 排除大文件）
├── checkpoints/           # 模型权重（.gitignore，传 HF）
├── figures/               # 论文风格的图
└── notes/                 # 学习笔记
```

### 7.4 README 必备章节

1. **项目摘要（一段话）**
2. **Headline 数字表**（6 数据集 × 3 模型对比）
3. **方法**（含架构图）
4. **实验**（含曲线图、对比图）
5. **消融**（每个消融一段）
6. **如何复现**（命令级别）
7. **致谢**（verl、Search-R1 等）

### 7.5 简历最终条目（再次强调）

写好上述内容后，在简历项目区写：
- 项目名 + 技术栈
- 4-5 行 bullet points，**每行必须有数字**
- GitHub 链接 + Wandb 链接

### 7.6 验收标准

- [ ] 至少 3 个消融实验完成，写入 `reports/ablations.md`
- [ ] GitHub repo 结构清晰，README 完整
- [ ] 所有 checkpoint 上传 HuggingFace
- [ ] 简历项目条目敲定
- [ ] 准备好至少 5 个面试可能追问的问题及答案
  1. GRPO vs PPO 的区别和选择理由
  2. Reward 设计的迭代过程
  3. 训练中遇到的最大问题和解决
  4. SFT 在 RL 中的作用
  5. Agentic RL 与传统 RL 的区别

---

## 8. 全局风险管理

### 8.1 时间风险

- **Phase 4 是最大风险点**：rollout 改造可能花 2 周。如果第 9 周末仍未跑通，启用降级方案（用 verl 原生 multi-turn 模板）
- **总体 Buffer**：每个 Phase 留 1 天 Buffer 处理意外问题

### 8.2 性能风险

- **如果 GRPO 不收敛**：先回到 Phase 1 GSM8K baseline 验证环境无误；再检查 reward 函数；最后调超参
- **如果数字达不到预期范围**：诚实写出，简历用相对提升而非绝对值（"提升 X 个点"）

### 8.3 算力风险

- **如果 4 卡不够**：降到 Qwen2.5-0.5B（仍有同样的方法学价值）
- **如果 8 卡可用**：把 scaling 到 3B 作为加分项

---

## 9. 关键参考资料

### 论文
- Search-R1 (Jin et al., 2025) - 项目核心参考
- DeepSeek-R1 (DeepSeek, 2025) - cold start + RLVR
- DAPO (ByteDance, 2025) - GRPO 改进
- HotpotQA (Yang et al., 2018) - 数据集论文

### 代码仓库
- verl: https://github.com/volcengine/verl
- Search-R1 official: https://github.com/PeterGriffinJin/Search-R1
- pyserini: https://github.com/castorini/pyserini

### 数据集
- HotpotQA: https://hotpotqa.github.io/
- KILT (Wikipedia): https://github.com/facebookresearch/KILT
- NQ-open, TriviaQA, 2WikiMultiHop, MuSiQue, Bamboogle: HuggingFace datasets

---

## 10. 附录：训练命令速查

### SFT 训练
```bash
python scripts/sft/train_sft.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --data_path data/hotpotqa/sft_cot_train.jsonl \
    --output_dir checkpoints/sft_hotpot_cot \
    --epochs 3 --lr 2e-5 --bs 4 --grad_acc 8
```

### Phase 3 GRPO（v1）
```bash
bash scripts/rl/train_grpo_distractor_v1.sh
```

### Phase 4 Agentic GRPO
```bash
# 先启动 BM25 服务
nohup uvicorn services.bm25_server:app --port 8765 --workers 4 &
# 再启动训练
bash scripts/rl/train_grpo_agentic.sh
```

### 评测
```bash
python scripts/eval/eval_benchmarks.py \
    --model_path checkpoints/agentic_grpo_final \
    --datasets nq triviaqa hotpotqa 2wiki musique bamboogle \
    --output reports/results_agentic.json
```

---

## 11. 给 Claude Code 的执行建议

> 本节专门写给 Claude Code 等 agent，便于其规划 todolist。

1. **执行节奏**：建议按 Phase 0 → 5 顺序执行，每个 Phase 完成后人类 review 再进入下一阶段
2. **Task 颗粒度**：单个 Task（如 Task 3.4 训练 v1）通常需要数小时到一天，不应在单次 agent session 内完成训练，应该是"启动训练 → 监控 → 等待完成 → 分析结果"分多次会话
3. **代码生成原则**：
   - 优先复用 verl 官方示例脚本，避免从零写
   - 修改 verl 时，**单独建 patches 目录**，不要直接修改 verl 源码（方便升级）
   - 所有训练命令必须可重复（固定 seed、记录 config）
4. **遇到不确定时**：参考 Search-R1 官方代码作为最权威的实现样板
5. **验收触发**：每个 Phase 末尾的"验收标准"必须由人类（项目持有者）确认通过，agent 不自行决定