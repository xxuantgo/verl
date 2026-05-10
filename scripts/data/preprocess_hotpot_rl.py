# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""HotpotQA distractor → verl RL parquet.

Emits the schema verl's RLHFDataset + NaiveRewardManager expect:

    prompt        : list[{"role", "content"}]      — system + user
    data_source   : str                              — "hotpotqa"
    reward_model  : dict                             — {"style": "rule",
                                                        "ground_truth": {answer, gold_titles}}
    extra_info    : dict                             — {qid, candidate_titles, raw_question,
                                                        type, level}

`candidate_titles` is the per-row list of all 10 passage titles (gold + distractor),
which the Phase 3 reward function reads to filter <think>-cited titles strictly
(see PHASE3_GRPO.md §3.4).

Per PHASE3_GRPO.md §4.2 the 10 passages are shuffled **once at preprocessing
time** with a seeded RNG so the position of gold passages cannot be learned, but
remains stable across the 5 GRPO rollouts of a given prompt (group internal
comparability).

Example:
    python scripts/data/preprocess_hotpot_rl.py \\
        --input  ~/data/hotpotqa/hotpot_train_v1.1.json \\
        --output ~/data/hotpotqa/rl_distractor_train_full.parquet
    python scripts/data/preprocess_hotpot_rl.py \\
        --input  ~/data/hotpotqa/hotpot_dev_distractor_v1.json \\
        --output ~/data/hotpotqa/rl_distractor_val.parquet \\
        --val-subset 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import STUDENT_SYSTEM_PROMPT, render_student_user  # noqa: E402


def _gold_titles(sf: Any) -> list[str]:
    """Distinct gold titles from supporting_facts, regardless of dump format.

    Both Stanford-official ``[[title, sent_id], ...]`` and HuggingFace
    ``{title:[], sent_id:[]}`` are supported. We deduplicate while preserving
    insertion order — sent_ids are dropped because Phase 3 reward uses only
    title-level sf-F1 (PHASE3_GRPO.md §3.4). Storing a flat ``list[str]`` also
    avoids pyarrow's heterogeneous-tuple schema headache.
    """
    if isinstance(sf, dict):
        titles = sf.get("title", [])
    elif isinstance(sf, list):
        titles = [pair[0] for pair in sf if pair]
    else:
        titles = []
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _passages_with_titles(context: Any, rng: random.Random) -> tuple[str, list[str]]:
    """Render context as a flat string and also return the title list (used by
    the reward function to filter <think>-cited titles strictly).

    The 10 passages are shuffled with the per-sample RNG so their position
    can't be memorized; the same shuffle is reflected in both the rendered
    string and the returned titles.
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for t, sents in zip(titles, sentences):
            body = "".join(sents) if isinstance(sents, list) else str(sents)
            pairs.append((t, body))
    else:
        for t, sents in context:
            body = "".join(sents) if isinstance(sents, list) else str(sents)
            pairs.append((t, body))

    rng.shuffle(pairs)
    rendered = "\n".join(f"[{t}] {body}" for t, body in pairs)
    return rendered, [t for t, _ in pairs]


def build_row(ex: dict, rng: random.Random) -> dict:
    qid = ex.get("_id") or ex.get("id")
    question = ex["question"]
    answer = ex["answer"]
    rendered_passages, candidate_titles = _passages_with_titles(ex.get("context", []), rng)
    gold_titles = _gold_titles(ex.get("supporting_facts"))

    user_msg = render_student_user(passages=rendered_passages, question=question)

    prompt = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    reward_model = {
        "style": "rule",
        "ground_truth": {
            "answer": answer,
            "gold_titles": gold_titles,
        },
    }
    extra_info = {
        "qid": qid,
        "candidate_titles": candidate_titles,
        "raw_question": question,
        "type": ex.get("type"),
        "level": ex.get("level"),
    }
    return {
        "prompt": prompt,
        "data_source": "hotpotqa",
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="HotpotQA train or dev distractor JSON.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument(
        "--val-subset",
        type=int,
        default=0,
        help="If >0, write only this many rows (used for val parquet, e.g. 500).",
    )
    p.add_argument("--seed", type=int, default=42, help="Per-sample passage shuffle seed.")
    args = p.parse_args()

    in_path = Path(os.path.expanduser(args.input))
    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if args.val_subset > 0:
        # Stable subset selection: shuffle once with a seeded RNG, take prefix.
        # This avoids leaking a question-class skew (e.g. all "yes/no" first).
        subset_rng = random.Random(args.seed)
        subset_rng.shuffle(raw)
        raw = raw[: args.val_subset]

    rng = random.Random(args.seed)
    rows = [build_row(ex, rng) for ex in raw]

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
