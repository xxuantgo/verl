# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Rule-based quality filter for synthesized CoT data.

Reads raw JSONL produced by ``synthesize_cot.py`` and keeps records where:
  * the ``<think>...</think><answer>...</answer>`` format parsed cleanly;
  * the ``<think>`` body is at least ``--min-think-chars`` characters;
  * the ``<answer>`` either exact-matches or contains a normalized substring of
    the gold answer (loose recall — gold may be a span);
  * total tokens (think + answer, by tokenizer or simple word count) are
    within ``--max-tokens``.

Stats are printed to stderr; rejected reasons are written to ``--rejects``.

Example:
    python scripts/data/filter_cot.py \\
        --input  ~/data/hotpotqa/sft_cot_raw.jsonl \\
        --output ~/data/hotpotqa/sft_cot_filtered.jsonl \\
        --rejects ~/data/hotpotqa/sft_cot_rejects.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path


_PUNC = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = re.sub(r"\s+", " ", s)
    return s


def answer_matches(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    if p == g or g in p or p in g:
        return True
    # token-level F1 fallback for slight wording differences
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return False
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return False
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= 0.6


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rejects", default=None, help="Optional JSONL path for rejected records")
    p.add_argument("--min-think-chars", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name (e.g. Qwen/Qwen2.5-1.5B-Instruct); "
             "if unset, uses whitespace word count",
    )
    args = p.parse_args()
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)
    if args.rejects:
        args.rejects = os.path.expanduser(args.rejects)
    return args


def main() -> None:
    args = parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rej_fh = None
    if args.rejects:
        Path(args.rejects).parent.mkdir(parents=True, exist_ok=True)
        rej_fh = open(args.rejects, "w", encoding="utf-8")

    stats = Counter()
    kept = 0
    total = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                reason = "bad_json"
                stats[reason] += 1
                if rej_fh:
                    rej_fh.write(
                        json.dumps(
                            {
                                "reason": reason,
                                "line_no": total,
                                "error": str(e),
                                "raw_line": line,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            reason = None
            if not rec.get("format_ok"):
                reason = "bad_format"
            else:
                think = rec.get("think") or ""
                ans = rec.get("answer") or ""
                gold = rec.get("raw_answer") or ""
                if len(think) < args.min_think_chars:
                    reason = "think_too_short"
                elif not answer_matches(ans, gold):
                    reason = "answer_mismatch"
                else:
                    n_tok = count_tokens(think, tokenizer) + count_tokens(ans, tokenizer)
                    if n_tok > args.max_tokens:
                        reason = "too_long"

            if reason is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                stats["kept"] += 1
            else:
                stats[reason] += 1
                if rej_fh is not None:
                    rec["_reject_reason"] = reason
                    rej_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if rej_fh is not None:
        rej_fh.close()

    print(f"total={total} kept={kept} ({kept / max(total, 1):.1%})", file=sys.stderr)
    for k, v in stats.most_common():
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
