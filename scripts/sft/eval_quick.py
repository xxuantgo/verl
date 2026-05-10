"""Quick post-SFT generation check for the HotpotQA CoT model.

Loads N validation rows, strips the gold assistant turn, asks the model to
complete (greedy), and reports format-compliance + answer match. Optionally
runs the same eval on a baseline model so the SFT delta is visible.

Example:
    # SFT model only
    python scripts/sft/eval_quick.py \\
        --ckpt checkpoints/sft_hotpot_cot/global_step_282 \\
        --val ~/data/hotpotqa/sft_cot_val.parquet --n 20

    # Head-to-head with the base instruct model
    python scripts/sft/eval_quick.py \\
        --ckpt checkpoints/sft_hotpot_cot/global_step_282 \\
        --base Qwen/Qwen2.5-1.5B-Instruct --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
from prompt_templates import extract_answer, parse_format  # noqa: E402


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def evaluate(model_path: str, rows: list[dict], max_new_tokens: int, tag: str) -> dict:
    print(f"\n========== {tag}: {model_path} ==========", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).cuda().eval()

    n_fmt = n_em = n_substr = 0
    for i, row in enumerate(rows):
        msgs = list(row["messages"][:2])  # system + user only
        gold = normalize(row["gold"])

        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).cuda()
        with torch.no_grad():
            out = mdl.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

        fmt_ok = parse_format(completion) is not None
        pred_raw = extract_answer(completion) or ""
        pred = normalize(pred_raw)
        em = bool(pred) and pred == gold
        substr = bool(pred) and (pred == gold or gold in pred or pred in gold)

        n_fmt += fmt_ok
        n_em += em
        n_substr += substr

        flag = "OK" if substr else "XX"
        print(
            f"[{i:02d}] {flag} fmt={int(fmt_ok)} em={int(em)} | "
            f"gold={gold!r:<40} pred={pred!r}",
            flush=True,
        )
        if not fmt_ok:
            print(f"     raw[:200]={completion[:200]!r}", flush=True)

    n = len(rows)
    summary = {
        "model": model_path,
        "n": n,
        "format_rate": n_fmt / n,
        "exact_match": n_em / n,
        "substring_match": n_substr / n,
    }
    print(f"\n{tag} summary: {json.dumps(summary, indent=2)}", flush=True)

    del mdl
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint dir (HF format)")
    ap.add_argument("--base", default=None, help="Optional baseline model id for A/B")
    ap.add_argument("--val", default=str(Path.home() / "data/hotpotqa/sft_cot_val.parquet"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_parquet(args.val)
    if args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    rows: list[dict] = []
    for _, r in df.iterrows():
        info = r["extra_info"]
        info = json.loads(info) if isinstance(info, str) else info
        rows.append({"messages": r["messages"], "gold": info["raw_answer"]})

    sft_summary = evaluate(args.ckpt, rows, args.max_new_tokens, tag="SFT")
    if args.base:
        base_summary = evaluate(args.base, rows, args.max_new_tokens, tag="BASE")
        delta = {
            k: round(sft_summary[k] - base_summary[k], 4)
            for k in ("format_rate", "exact_match", "substring_match")
        }
        print(f"\nSFT - BASE delta: {json.dumps(delta, indent=2)}")


if __name__ == "__main__":
    main()
