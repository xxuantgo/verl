# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Phase 3 kickoff Task 4 — SFT format-correctness sanity check.

Runs the SFT checkpoint on a small sample of the *raw* HotpotQA dev
distractor set (out-of-distribution relative to the SFT val parquet) and
reports the rate at which the policy emits a parseable
``<think>...</think><answer>...</answer>``.

PHASE3_GRPO.md §2.2 ("E1") makes this a hard gate: if format_ok < 95%, GRPO
will silently fail (most prompts will hit the "missing tags" branch and
contribute zero advantage — the answer-avoidance precursor). In that case
the right move is to fix the SFT data or train another epoch, NOT to
compensate in the reward.

Exits non-zero when below threshold so it can be wired into a smoke-test
pipeline.

Example:
    CUDA_VISIBLE_DEVICES=0 \\
    /home/luoxuan/anaconda3/envs/verl/bin/python scripts/eval/sft_format_sanity.py \\
        --ckpt checkpoints/sft_hotpot_cot_hf/global_step_500 \\
        --dev /home/luoxuan/data/hotpotqa/hotpot_dev_distractor_v1.json \\
        --n 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data"))
from prompt_templates import (  # noqa: E402
    STUDENT_SYSTEM_PROMPT,
    classify_format,
    extract_answer,
    render_student_user,
)

# Reuse the exact same passage rendering + per-sample shuffle used by the RL
# parquet so this sanity check is faithful to what the policy will see in
# Phase 3 rollouts.
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data"))
from preprocess_hotpot_rl import _passages_with_titles  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="SFT checkpoint (HF format)")
    p.add_argument("--dev", required=True, help="Path to hotpot_dev_distractor_v1.json")
    p.add_argument("--n", type=int, default=64, help="Number of dev rows to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--threshold", type=float, default=0.95, help="Pass threshold for format_ok rate")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    with open(os.path.expanduser(args.dev), "r", encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(raw)
    sample = raw[: args.n]
    print(f"Sampled {len(sample)} dev rows.")

    chat_inputs: list[str] = []
    questions: list[str] = []
    answers: list[str] = []
    for ex in sample:
        passages, _titles = _passages_with_titles(ex.get("context", []), rng)
        user_msg = render_student_user(passages=passages, question=ex["question"])
        msgs = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        chat_inputs.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        questions.append(ex["question"])
        answers.append(ex["answer"])

    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        seed=args.seed,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, n=1, seed=args.seed)
    outputs = llm.generate(chat_inputs, sampling)

    n_ok = 0
    n_missing = 0
    n_malformed = 0
    n_answer_present = 0
    examples: list[dict] = []
    for q, gold, out in zip(questions, answers, outputs):
        text = out.outputs[0].text
        state = classify_format(text)
        ans = extract_answer(text)
        if state == "ok":
            n_ok += 1
        elif state == "malformed":
            n_malformed += 1
        else:
            n_missing += 1
        if ans is not None:
            n_answer_present += 1
        if len(examples) < 3 and state != "ok":
            examples.append({"question": q, "gold": gold, "state": state, "completion": text[:400]})

    n = len(outputs)
    rate_ok = n_ok / n
    rate_answer = n_answer_present / n
    print("=" * 60)
    print(f"  format_ok       : {n_ok}/{n} = {rate_ok:.1%}")
    print(f"  format_missing  : {n_missing}/{n}")
    print(f"  format_malformed: {n_malformed}/{n}")
    print(f"  has <answer>    : {n_answer_present}/{n} = {rate_answer:.1%}")
    print("=" * 60)
    for i, e in enumerate(examples):
        print(f"\n[FAIL EXAMPLE {i + 1} | state={e['state']}]")
        print(f"  Q: {e['question']}")
        print(f"  gold: {e['gold']}")
        print(f"  completion (first 400 chars): {e['completion']}")

    if rate_ok < args.threshold:
        print(f"\n❌ FAIL: format_ok {rate_ok:.1%} < threshold {args.threshold:.0%}. "
              "Fix Phase 2 SFT before launching GRPO.")
        sys.exit(1)
    print(f"\n✅ PASS: format_ok {rate_ok:.1%} ≥ threshold {args.threshold:.0%}.")


if __name__ == "__main__":
    main()
