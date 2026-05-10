# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Convert filtered HotpotQA CoT JSONL into the parquet schema verl SFT expects.

verl's MultiTurnSFTDataset reads a ``messages`` column whose value is a list of
``{role, content}`` dicts and applies the tokenizer's chat template, masking
loss on non-assistant turns. We emit:

  messages = [
    {"role": "system", "content": <system prompt with format spec>},
    {"role": "user",   "content": <passages + question>},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"},
  ]

plus ``extra_info`` for downstream RL alignment (qid, gold answer,
supporting facts).

Example:
    python scripts/data/preprocess_hotpot_sft.py \\
        --input  ~/data/hotpotqa/sft_cot_filtered.jsonl \\
        --train-out ~/data/hotpotqa/sft_cot_train.parquet \\
        --val-out   ~/data/hotpotqa/sft_cot_val.parquet \\
        --val-size 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/data/preprocess_hotpot_sft.py ...` from the repo root
# without needing the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import STUDENT_SYSTEM_PROMPT, render_student_user  # noqa: E402


def build_row(rec: dict) -> dict:
    user_msg = render_student_user(passages=rec["passages"], question=rec["raw_question"])
    assistant_msg = f"<think>{rec['think']}</think><answer>{rec['answer']}</answer>"
    # supporting_facts is [[title, sent_idx], ...] with mixed types per element,
    # which pyarrow can't infer a stable schema for; stash extra_info as a
    # JSON string so the parquet schema stays simple (str + list[struct]).
    extra_info = {
        "qid": rec.get("qid"),
        "raw_question": rec.get("raw_question"),
        "raw_answer": rec.get("raw_answer"),
        "supporting_facts": rec.get("supporting_facts"),
        "meta": rec.get("meta"),
    }
    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "data_source": "hotpotqa",
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Filtered JSONL from filter_cot.py")
    p.add_argument("--train-out", required=True)
    p.add_argument("--val-out", default=None)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    in_path = Path(os.path.expanduser(args.input))
    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(build_row(rec))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_rows: list[dict] = []
    if args.val_out and args.val_size > 0:
        val_rows = rows[: args.val_size]
        rows = rows[args.val_size :]

    train_out = Path(os.path.expanduser(args.train_out))
    train_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(train_out, index=False)
    print(f"wrote {len(rows)} train rows -> {train_out}")

    if args.val_out and val_rows:
        val_out = Path(os.path.expanduser(args.val_out))
        val_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val_rows).to_parquet(val_out, index=False)
        print(f"wrote {len(val_rows)} val rows -> {val_out}")


if __name__ == "__main__":
    main()
