# 迁移到 A100×2 服务器 — Phase 3 GRPO

> 目标：在另一台服务器（2× A100）上从 SFT checkpoint `global_step_500` 开始跑 Phase 3 GRPO。
> 源机：当前 4090 节点。日期：2026-05-11。

## 目标机连接信息

- host: `210.45.79.253`
- ssh port: `1220`
- user: `ustcwzy`
- 项目目录: `/home/ustcwzy/runcodes/verl`
- 数据目录: `/mnt/nv0/ustcwzy/xyzp/data/hotpotqa`

迁移分四步：**①推代码 → ②传数据 → ③装环境 → ④调参起跑**。

---

## Step 1 — 代码：通过 git 推到 fork

源机 `origin` 是上游 `verl-project/verl`（CLAUDE.md §1 禁止直接 push）。把代码 push 到你的私有 fork。

### 1.1 配置 fork remote

```bash
cd ~/runcodes/verl
git remote add fork <YOUR_FORK_URL>     # 例如 git@github.com:lxxyz/verl.git
git remote -v                           # 确认 fork 出现在列表里
```

### 1.2 检查改动清单

这次提交涉及多块工作，先核对一下要进 commit 的内容：

```bash
git status -s                           # 看 modified + untracked
git diff --stat                         # 看 modified 行数
```

预期看到：

- **新文档**：`PHASE2_SFT_REPORT.md`、`PHASE3_GRPO.md`、`MIGRATE_TO_A100.md`
- **修改文档**：`PROJECT_PLAN1.md`（Phase 3 章节扩展）
- **共享 prompt**：`scripts/data/prompt_templates.py`（加 RL 渲染入口）
- **RL 数据流水线**：`scripts/data/preprocess_hotpot_rl.py`（新）、`scripts/data/filter_sft_solved.py`（新，两阶段 cache + score）
- **SFT format sanity**：`scripts/eval/sft_format_sanity.py`（新）
- **Reward 函数**：`src/__init__.py`、`src/rewards/hotpot.py`（新，dict 返回 v1/v2 reward）
- **GRPO launch**：`scripts/rl/launch_grpo.sh`、`scripts/rl/run_all_phase3.sh`、`scripts/rl/migrate_to_a100.sh`（新）
- **verl 注释**：`verl/trainer/ppo/core_algos.py`、`verl/trainer/ppo/ray_trainer.py`（仅加学习注释，不改逻辑）

**不要提交**：`tensorboard_log/`（训练产物）、`__pycache__/`（已被 .gitignore 处理）。

### 1.3 暂存 + 提交

```bash
# 暂存（显式列出，避免 git add -A 把 tensorboard_log/ 误带进来）
git add PROJECT_PLAN1.md \
        PHASE2_SFT_REPORT.md PHASE3_GRPO.md MIGRATE_TO_A100.md \
        scripts/data/prompt_templates.py \
        scripts/data/preprocess_hotpot_rl.py \
        scripts/data/filter_sft_solved.py \
        scripts/eval/sft_format_sanity.py \
        scripts/rl/launch_grpo.sh \
        scripts/rl/run_all_phase3.sh \
        scripts/rl/migrate_to_a100.sh \
        src/__init__.py src/rewards/ \
        verl/trainer/ppo/core_algos.py \
        verl/trainer/ppo/ray_trainer.py

# 确认暂存区
git diff --cached --stat

# 提交（CLAUDE.md §2 要求 trailer + 说清 AI 协助）
git commit -m "$(cat <<'EOF'
Phase 3 GRPO scaffolding: data pipeline, reward, launcher, migration

Adds the full Phase 3 toolchain on top of the SFT cold-start (PR-merged
Phase 2 work):

* Docs
  - PHASE3_GRPO.md: per-decision design log (reward formulation, GRPO
    hyperparams, eval cadence, failure-mode playbook, kickoff checklist).
  - PHASE2_SFT_REPORT.md: SFT outcome summary, used as the GRPO start.
  - MIGRATE_TO_A100.md: 4-step migration runbook for A100 hosts.
  - PROJECT_PLAN1.md: expanded Phase 3 section.

* Data pipeline
  - scripts/data/preprocess_hotpot_rl.py: HotpotQA distractor -> RL parquet
    with one-shot passage shuffle (PHASE3_GRPO.md §4.2).
  - scripts/data/filter_sft_solved.py: two-stage SFT-solved filter
    (vLLM gen cache -> CPU scoring) so a scoring crash no longer wastes
    the multi-hour generation run.
  - scripts/data/prompt_templates.py: RL-side rendering entrypoint sharing
    the SFT prompt source (PHASE3_GRPO.md §4.1).

* Reward
  - src/rewards/hotpot.py: compute_score returning the dict required by
    verl's naive reward manager; v1 (EM-only) and v2 (alpha * answer_f1
    + (1-alpha) * sf_f1 + format_bonus + action_penalty) selected via
    REWARD_VERSION/ALPHA env (PHASE3_GRPO.md §3.1-3.4, §8.1).

* Eval / sanity
  - scripts/eval/sft_format_sanity.py: dev-set tag-correctness check
    before launching RL.

* Training launcher
  - scripts/rl/launch_grpo.sh: GRPO launcher with frozen hyperparams
    (kl_loss_coef=0.001, kl_loss_type=low_var_kl, loss_agg_mode=token-mean,
    clip_ratio_high=0.28, rollout.n=5) per PHASE3_GRPO.md §5.
  - scripts/rl/run_all_phase3.sh: sequential driver: smoke gate (v1, 100
    steps) then v1 + five v2 alphas (0.0/0.3/0.5/0.7/1.0); GPUS / N_GPUS
    / EXTRA_OVERRIDES env knobs for per-host tuning.
  - scripts/rl/migrate_to_a100.sh: rsync-based one-shot for ckpt + data
    (code is pushed via git, not rsynced).

* verl annotations
  - verl/trainer/ppo/core_algos.py, ray_trainer.py: comments only
    (learning notes), no behavioral change.
EOF
)"

# AI assistance: Claude Code drafted the docs, reward, launcher, driver,
# and migration script. Every changed line was reviewed; smoke-test path
# verified end-to-end on the 4090 node.

# Co-authored-by: Claude

# 推送到 fork
git push -u fork study/hotpot-agentic-rl
```

### 1.4 在 A100 机部署代码

**首选**：`git clone` 你的 fork（若网速可以）：

```bash
ssh -p 1220 ustcwzy@210.45.79.253
mkdir -p /home/ustcwzy/runcodes
cd /home/ustcwzy/runcodes
git clone -b study/hotpot-agentic-rl <YOUR_FORK_URL> verl
cd verl && git log --oneline -3
exit
```

**fallback**：git clone 一直很慢时，用 rsync 直接传代码（约 160MB，比 git clone 快很多，因为不走 GitHub）。已扩展 `migrate_to_a100.sh` 支持 `SYNC_CODE=1` / `ONLY_CODE=1`：

```bash
# 在源机；只传代码，跳过 ckpt + 数据
cd ~/runcodes/verl

ONLY_CODE=1 \
DEST_HOST=ustcwzy@210.45.79.253 \
SSH_PORT=1220 \
DEST_REPO=/home/ustcwzy/runcodes/verl \
DEST_DATA=/mnt/nv0/ustcwzy/xyzp/data/hotpotqa \
    bash scripts/rl/migrate_to_a100.sh
```

rsync 会自动跳过：`checkpoints/`、`tensorboard_log/`、`logs/`、`runs/`、`wandb/`、`outputs/`、`models/`、`.venv/`、`__pycache__/`、`*.whl`、`*.pyc`、`.env`、`*.egg-info/`。`.git` 目录会同步过去，远端仍能 `git pull`、`git status` 正常工作。

---

## Step 2 — 数据 + ckpt：rsync

前提：Step 1.4 已经在 A100 机上 git clone 完成（`/home/ustcwzy/runcodes/verl` 已存在）。

### 2.1 先 dry-run 看清单

```bash
# 在源机
cd ~/runcodes/verl

DEST_HOST=ustcwzy@210.45.79.253 \
SSH_PORT=1220 \
DEST_REPO=/home/ustcwzy/runcodes/verl \
DEST_DATA=/mnt/nv0/ustcwzy/xyzp/data/hotpotqa \
DRY_RUN=1 \
    bash scripts/rl/migrate_to_a100.sh
```

确认会传 4 项（合计 ~1.5GB），无报错后去掉 `DRY_RUN=1` 实跑。

### 2.2 实际传输

```bash
DEST_HOST=ustcwzy@210.45.79.253 \
SSH_PORT=1220 \
DEST_REPO=/home/ustcwzy/runcodes/verl \
DEST_DATA=/mnt/nv0/ustcwzy/xyzp/data/hotpotqa \
    bash scripts/rl/migrate_to_a100.sh
```

家庭带宽下 1.5GB ≈ 5–15 分钟。`rsync --partial -P` 支持断点续传，挂了直接重跑同一条。

### 2.3 远端验收

```bash
ssh -p 1220 ustcwzy@210.45.79.253 \
    "ls -lh /home/ustcwzy/runcodes/verl/checkpoints/sft_hotpot_cot_hf/global_step_500/ \
     && ls -lh /mnt/nv0/ustcwzy/xyzp/data/hotpotqa/"
```

预期看到：

- ckpt 目录里 `*.safetensors` + `tokenizer*` + `chat_template.jinja` + `config.json`
- 数据目录里 `rl_distractor_train.parquet` (~271M)、`rl_distractor_val.parquet` (~1.8M)、`hotpot_dev_distractor_v1.json` (~45M)

### 2.4 传送清单

| 项 | 源路径 | 目标路径 | 大小 |
| --- | --- | --- | --- |
| SFT ckpt | `~/runcodes/verl/checkpoints/sft_hotpot_cot_hf/` | `/home/ustcwzy/runcodes/verl/checkpoints/sft_hotpot_cot_hf/` | 1.2 GB |
| RL 训练集 | `~/data/hotpotqa/rl_distractor_train.parquet` | `/mnt/nv0/ustcwzy/xyzp/data/hotpotqa/` | 271 MB |
| RL 验证集 | `~/data/hotpotqa/rl_distractor_val.parquet` | `/mnt/nv0/ustcwzy/xyzp/data/hotpotqa/` | 1.8 MB |
| Eval 集 | `~/data/hotpotqa/hotpot_dev_distractor_v1.json` | `/mnt/nv0/ustcwzy/xyzp/data/hotpotqa/` | 45 MB |

**不传**：`hotpot_train_v1.1.json`（538MB，已派生为 parquet，新机不需要）；SFT 中间数据；filter_gen_cache.jsonl；wandb/tensorboard 历史。

---

## Step 3 — 环境：在 A100 机新建 venv

### 3.1 安装 uv（绕过 anaconda libcurl 冲突）

报错 `libcurl.so.4: no version information available` 是因为 anaconda 的 libcurl 劫持了 LD 加载，导致 `curl` 退化/卡死。**不要用 curl 安装 uv**。两种替代：

#### A. 用 anaconda 的 pip 装 uv（最简单，推荐）

```bash
pip install uv
which uv                  # 应在 ~/anaconda3/bin/uv
```

#### B. 临时禁用 anaconda libcurl 再走官方 curl 脚本

```bash
LD_LIBRARY_PATH= /usr/bin/curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3.2 建虚拟环境 + 装依赖

```bash
cd /home/ustcwzy/runcodes/verl

uv venv --python 3.12
source .venv/bin/activate

# verl 本身（含训练栈）
uv pip install -e .

# 训练 + RL 额外依赖
uv pip install vllm ray hydra-core wandb tensorboard
uv pip install pre-commit
pre-commit install
```

> 如果 `uv pip install -e .` 抱怨找不到某些 build 依赖（flash-attn 等），可以先 `uv pip install --upgrade pip setuptools wheel` 再重试。flash-attn 在 A100 上建议直接 `pip install flash-attn --no-build-isolation`。

### 3.3 登录外部服务

**HF 认证**（拉 Qwen2.5 tokenizer/base 时用）：

```bash
huggingface-cli login   # 粘贴 token
```

如果你的 SFT ckpt 里已经包含 tokenizer（已确认包含），base 模型不需要重新拉。

**wandb 登录**：

```bash
wandb login   # 粘贴 API key
```

---

## Step 4 — 起跑：driver 脚本带 GPU 旋钮

A100×2，每张 40GB（或 80GB）。

```bash
# 推荐起手参数（A100-40GB）：
GPUS=0,1 \
N_GPUS=2 \
EXTRA_OVERRIDES="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
                 actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    bash scripts/rl/run_all_phase3.sh
```

A100-80GB 可进一步把 micro batch 拉到 8、`gpu_memory_utilization=0.8`。

**变量含义**：

- `GPUS=0,1` → 设 `CUDA_VISIBLE_DEVICES`
- `N_GPUS=2` → `trainer.n_gpus_per_node`
- `EXTRA_OVERRIDES` → 任意 hydra override 透传

**只先跑 smoke gate**（推荐，新机第一步必做）：

```bash
GPUS=0,1 N_GPUS=2 ONLY_SMOKE=1 \
EXTRA_OVERRIDES="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
                 actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    bash scripts/rl/run_all_phase3.sh
```

100 步（~30–60 min）跑通无报错，再去掉 `ONLY_SMOKE=1` 跑全套 6 个 run。

---

## 验收 checklist

迁移完成后 A100 机上确认：

- [ ] `ls checkpoints/sft_hotpot_cot_hf/global_step_500/` 有 `*.safetensors` + `tokenizer*` + `chat_template.jinja`
- [ ] `ls -lh ~/data/hotpotqa/` 三个文件齐全
- [ ] `python -c "import verl; import vllm; import ray; print('ok')"` 无报错
- [ ] `nvidia-smi` 看到 2 张 A100 空闲
- [ ] `bash scripts/rl/run_all_phase3.sh` 的 smoke gate 跑完不挂

任一不过 → 回到对应 Step 排查，不要硬上正式训练。
