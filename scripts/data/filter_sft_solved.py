# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Filter SFT-already-solved HotpotQA training samples.

For Phase 3 GRPO we want to focus the rollout budget on prompts where the SFT
checkpoint still has room to improve. This script runs in two stages so a
crash in scoring doesn't lose the (expensive) vLLM generation pass:

  Stage A (generate):
    - Load the RL train parquet.
    - For each prompt, sample N times with vLLM (T=1.0).
    - Append one jsonl row per prompt to ``--gen-cache`` containing all N
      completions: ``{"row_idx": int, "qid": str, "completions": [str, ...]}``.
    - If ``--gen-cache`` already has len(df) lines we skip generation entirely.
      This makes scoring re-runs cheap: tweak threshold, rerun, get fresh
      kept-parquet in ~30 seconds.

  Stage B (score):
    - Stream the cache jsonl, score every completion with src/rewards/hotpot.py
      at v2 α=1.0 (so we get raw answer_f1 / sf_f1 components).
    - Mark a prompt "solved" iff at least one completion has
      ``answer_f1 ≥ --f1-threshold`` AND ``sf_f1 ≥ --sf-threshold``.
    - Write kept rows to ``--output-kept`` and per-row stats to
      ``--output-audit``.

Expected runtime on 4×4090 with 1.5B + 90k prompts × 4 samples (≈360k gens at
~512 tokens each): roughly 1.5–3h for stage A; stage B is CPU-bound, ~5 min.

PHASE3_GRPO.md §4.3 explains the rationale and the 70k-row target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Reward function (and prompt-template helpers it pulls in).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# We force REWARD_VERSION=v2/ALPHA=1.0 so the dict we get back exposes both
# answer_f1 and sf_f1 cleanly. Setting before import is required because the
# module reads env at import time.
os.environ.setdefault("REWARD_VERSION", "v2")
os.environ.setdefault("ALPHA", "1.0")

from src.rewards.hotpot import compute_score  # noqa: E402


def _format_chat(prompt_msgs, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        list(prompt_msgs),
        tokenize=False,
        add_generation_prompt=True,
    )


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def stage_generate(df: pd.DataFrame, args) -> None:
    """Run vLLM and append per-prompt jsonl rows to the cache."""
    cache_path = Path(os.path.expanduser(args.gen_cache))
    existing = _count_lines(cache_path)
    if existing >= len(df):
        print(f"[gen] cache already has {existing} >= {len(df)} rows; skipping vLLM")
        return
    if existing > 0:
        # Partial cache from a prior crash mid-generation: easiest correct
        # behavior is to start over (vLLM .generate runs the whole batch in
        # one call so partial-resume is hard). We rename the partial file out
        # of the way rather than silently delete.
        bak = cache_path.with_suffix(cache_path.suffix + f".partial-{existing}")
        cache_path.rename(bak)
        print(f"[gen] partial cache ({existing} rows) preserved at {bak}; regenerating from scratch")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    prompt_strs = [_format_chat(row["prompt"], tokenizer) for _, row in df.iterrows()]
    qids = [dict(row["extra_info"]).get("qid") for _, row in df.iterrows()]

    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(f"[gen] generating {len(prompt_strs)} × n={args.n_samples} = "
          f"{len(prompt_strs) * args.n_samples} completions...")
    outputs = llm.generate(prompt_strs, sampling)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row_idx, out in enumerate(outputs):
            rec = {
                "row_idx": row_idx,
                "qid": qids[row_idx],
                "completions": [s.text for s in out.outputs],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.rename(cache_path)
    print(f"[gen] wrote {len(outputs)} cached prompt rows -> {cache_path}")


def stage_score(df: pd.DataFrame, args) -> None:
    """Score cached completions, write kept parquet + audit jsonl."""
    cache_path = Path(os.path.expanduser(args.gen_cache))
    n_cached = _count_lines(cache_path)
    if n_cached < len(df):
        raise RuntimeError(
            f"[score] cache {cache_path} has {n_cached} rows but df has {len(df)}; "
            f"run generation first or pass matching --limit"
        )

    keep_idx: list[int] = []
    audit_rows: list[dict] = []

    df_rows = list(df.iterrows())
    with cache_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if row_idx >= len(df_rows):
                break
            rec = json.loads(line)
            assert rec["row_idx"] == row_idx, f"cache misaligned at line {row_idx}: rec={rec['row_idx']}"
            _, row = df_rows[row_idx]

            ground_truth = dict(row["reward_model"]["ground_truth"])
            if "gold_titles" in ground_truth and ground_truth["gold_titles"] is not None:
                ground_truth["gold_titles"] = list(ground_truth["gold_titles"])

            extra_info = dict(row["extra_info"])
            if "candidate_titles" in extra_info:
                extra_info["candidate_titles"] = list(extra_info["candidate_titles"])

            max_f1 = 0.0
            max_sf_f1 = 0.0
            max_combined = 0.0
            for text in rec["completions"]:
                score = compute_score(
                    data_source="hotpotqa",
                    solution_str=text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                af = float(score["answer_f1"])
                sf = float(score["sf_f1"])
                if af > max_f1:
                    max_f1 = af
                if sf > max_sf_f1:
                    max_sf_f1 = sf
                if af >= args.f1_threshold and sf >= args.sf_threshold:
                    cand = min(af, sf)
                    if cand > max_combined:
                        max_combined = cand

            solved = max_combined >= min(args.f1_threshold, args.sf_threshold)
            audit_rows.append({
                "row_idx": row_idx,
                "qid": extra_info.get("qid"),
                "max_answer_f1": max_f1,
                "max_sf_f1": max_sf_f1,
                "solved": solved,
            })
            if not solved:
                keep_idx.append(row_idx)

            if (row_idx + 1) % 5000 == 0:
                kept = len(keep_idx)
                print(f"[score]   [{row_idx + 1}/{len(df)}] kept {kept}  ({kept / (row_idx + 1):.1%})")

    out_kept = Path(os.path.expanduser(args.output_kept))
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[keep_idx].reset_index(drop=True).to_parquet(out_kept, index=False)
    print(f"[score] wrote {len(keep_idx)} kept rows -> {out_kept}")

    out_audit = Path(os.path.expanduser(args.output_audit))
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    with out_audit.open("w", encoding="utf-8") as f:
        for r in audit_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_solved = sum(1 for r in audit_rows if r["solved"])
    print(f"[score] audit: {n_solved}/{len(audit_rows)} ({n_solved / len(audit_rows):.1%}) solved -> {out_audit}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="RL train parquet from preprocess_hotpot_rl.py")
    p.add_argument("--output-kept", required=True, help="Where to write the kept (unsolved) rows parquet")
    p.add_argument("--output-audit", required=True, help="JSONL with per-row max scores")
    p.add_argument("--gen-cache", required=True,
                   help="JSONL cache of vLLM completions; reused across reruns to skip the 2h generation step")
    p.add_argument("--ckpt", required=True, help="SFT checkpoint directory (HF format)")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--f1-threshold", type=float, default=0.9)
    p.add_argument("--sf-threshold", type=float, default=0.8)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="If >0, only process this many rows (smoke test)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--stage", choices=["all", "generate", "score"], default="all",
                   help="all = generate (skip if cached) then score; generate / score run only one stage")
    args = p.parse_args()

    df = pd.read_parquet(os.path.expanduser(args.input))
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f"loaded {len(df)} rows from {args.input}")

    if args.stage in ("all", "generate"):
        stage_generate(df, args)
    if args.stage in ("all", "score"):
        stage_score(df, args)


if __name__ == "__main__":
    main()
