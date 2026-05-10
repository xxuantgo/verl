# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""HotpotQA Phase 3 GRPO reward function.

Loaded by verl as a custom_reward_function plugin (see
verl/trainer/config/reward/reward.yaml). Signature follows
verl.workers.reward_manager.NaiveRewardManager:

    compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

The dict's "score" key feeds the RL signal; every other key is auto-logged to
wandb under critic/rewards/* (see verl/workers/reward_manager/naive.py:92-96).

Reward version is selected via the REWARD_VERSION env var ("v1" or "v2"),
weight α via ALPHA. This lets a single parquet drive all six runs in the
α ablation (v1 + v2 at α∈{0.0, 0.3, 0.5, 0.7, 1.0_F1}) without re-preprocessing.

Design rationale and references: see PHASE3_GRPO.md §3.
"""

from __future__ import annotations

import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Pull in the shared prompt-format helpers — same module SFT preprocessing used,
# so the format the policy is graded on is the format SFT trained.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data"))
from prompt_templates import classify_format, extract_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Run-level config (read once at import; launch script sets these via env)
# ---------------------------------------------------------------------------

REWARD_VERSION = os.environ.get("REWARD_VERSION", "v1").lower()
ALPHA = float(os.environ.get("ALPHA", "1.0"))   # weight on answer (vs sf) score

# v1 uses EM as answer score; v2 uses HotpotQA token-level F1. v2 with α=1.0
# is the "F1-only + format + action_penalty" variant (Search-R1 style).
ANSWER_SCORE_KIND = "em" if REWARD_VERSION == "v1" else "f1"

# Format bonus / action penalty magnitudes (PHASE3_GRPO.md §3.3).
FORMAT_BONUS_OK = 0.10
FORMAT_BONUS_MALFORMED = -0.10
ACTION_PENALTY_HEDGE = -0.50

# Length penalty (PHASE3_GRPO.md §3.6) — disabled by default, enable mid-run
# via env when length hacking is observed.
LENGTH_PENALTY_ENABLED = os.environ.get("LENGTH_PENALTY", "0") == "1"
LENGTH_PENALTY_THRESHOLD = 1024
LENGTH_PENALTY_SLOPE = 0.1
LENGTH_PENALTY_RANGE = 256


# ---------------------------------------------------------------------------
# HotpotQA official answer normalization + EM / F1
# (mirrors hotpot_evaluate_v1.py from the dataset authors)
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLE_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def _yesno_short_circuit(pred_norm: str, gold_norm: str) -> bool:
    """HotpotQA's yes/no short-circuit — if either side is yes/no/noanswer
    and they don't exactly match, F1 is forced to 0."""
    SPECIAL = {"yes", "no", "noanswer"}
    if pred_norm in SPECIAL or gold_norm in SPECIAL:
        return pred_norm != gold_norm
    return False


def compute_em(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def compute_token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 on normalized strings, HotpotQA official algorithm."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if _yesno_short_circuit(pred_norm, gold_norm):
        return 0.0
    pred_tok = pred_norm.split()
    gold_tok = gold_norm.split()
    if not pred_tok or not gold_tok:
        return 0.0
    common = Counter(pred_tok) & Counter(gold_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tok)
    recall = num_same / len(gold_tok)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Supporting-facts F1 (v2 only) — title-level, set-based
# ---------------------------------------------------------------------------

# Match titles that the model emits inside <think>: quoted strings or markdown
# bold. We deliberately do NOT do free-form substring matching against passage
# titles, because that would let the model "cite" all 10 distractor titles
# verbatim and trivially saturate sf precision.
_QUOTED_RE = re.compile(r'"([^"\n]{2,80})"|“([^”\n]{2,80})”')
_BOLD_RE = re.compile(r"\*\*([^*\n]{2,80})\*\*")


def _normalize_title(t: str) -> str:
    return " ".join(t.lower().split())


def extract_supporting_facts_from_think(think: str, candidate_titles: list[str]) -> set[str]:
    """Return the set of normalized titles the model cited inside <think>.

    Only counts titles that (a) appear as a quoted string or markdown-bold span
    AND (b) match one of the 10 candidate passage titles for this sample. This
    is a deliberately strict filter — see comment on the regexes above.
    """
    if not think or not candidate_titles:
        return set()
    candidate_norm = {_normalize_title(t): t for t in candidate_titles}

    cited: set[str] = set()
    for m in _QUOTED_RE.finditer(think):
        span = m.group(1) or m.group(2) or ""
        if _normalize_title(span) in candidate_norm:
            cited.add(_normalize_title(span))
    for m in _BOLD_RE.finditer(think):
        if _normalize_title(m.group(1)) in candidate_norm:
            cited.add(_normalize_title(m.group(1)))
    return cited


def compute_sf_f1(pred_titles: set[str], gold_titles: set[str]) -> float:
    """Set-based F1, HotpotQA update_sp algorithm."""
    if not gold_titles:
        return 0.0
    if not pred_titles:
        return 0.0
    tp = len(pred_titles & gold_titles)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_titles)
    recall = tp / len(gold_titles)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Action penalty: detect "I don't know" / empty answer (E1: anti-avoidance)
# ---------------------------------------------------------------------------

_HEDGE_PHRASES = {
    "",
    "i dont know",
    "i do not know",
    "unknown",
    "cannot determine",
    "cant determine",
    "no information",
    "not enough information",
    "insufficient information",
    "no answer",
    "n/a",
    "na",
    "none",
    "null",
}


def is_empty_or_hedge(answer: str | None) -> bool:
    if answer is None:
        return True
    norm = normalize_answer(answer)
    return norm in _HEDGE_PHRASES


# ---------------------------------------------------------------------------
# Length penalty (DAPO Soft Overlong Punishment, PHASE3_GRPO.md §3.6)
# ---------------------------------------------------------------------------


def soft_overlong_penalty(length_tokens: int) -> float:
    if not LENGTH_PENALTY_ENABLED or length_tokens <= LENGTH_PENALTY_THRESHOLD:
        return 0.0
    excess = length_tokens - LENGTH_PENALTY_THRESHOLD
    return -LENGTH_PENALTY_SLOPE * min(excess / LENGTH_PENALTY_RANGE, 1.0)


# ---------------------------------------------------------------------------
# verl-facing entry point
# ---------------------------------------------------------------------------


def _gold_titles(ground_truth: dict[str, Any]) -> set[str]:
    """Read normalized gold titles from ground_truth.

    Preferred schema: ``{"answer": str, "gold_titles": list[str]}`` (what
    preprocess_hotpot_rl.py emits). Also supports legacy
    ``{"supporting_facts": [[title, sent_id], ...]}`` as a fallback so old
    parquets keep working.
    """
    # NB: Don't write `x or []`. After parquet round-trip these fields are
    # numpy arrays, and bool(np.ndarray) raises on length > 1. Use explicit
    # None checks instead.
    titles_raw: list[Any] = []
    if "gold_titles" in ground_truth:
        gt = ground_truth.get("gold_titles")
        if gt is not None:
            titles_raw = list(gt)
    else:
        sf = ground_truth.get("supporting_facts")
        if sf is not None:
            for entry in sf:
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    titles_raw.append(entry[0])
                elif isinstance(entry, str):
                    titles_raw.append(entry)
    return {_normalize_title(str(t)) for t in titles_raw if t}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """Phase 3 GRPO reward. See PHASE3_GRPO.md §3 for design rationale."""
    extra_info = extra_info or {}

    # ground_truth is expected to be a dict from the parquet's reward_model
    # column. For robustness against single-string ground_truth (e.g. early
    # smoke tests), accept that too.
    if isinstance(ground_truth, str):
        ground_truth = {"answer": ground_truth, "gold_titles": []}

    gold_answer = str(ground_truth.get("answer", ""))
    candidate_titles = list(extra_info.get("candidate_titles", []))
    response_len = int(extra_info.get("response_len", len(solution_str.split())))

    # ---- 1. Format ------------------------------------------------------
    fmt_state = classify_format(solution_str)
    if fmt_state == "ok":
        format_bonus = FORMAT_BONUS_OK
    elif fmt_state == "malformed":
        format_bonus = FORMAT_BONUS_MALFORMED
    else:
        format_bonus = 0.0

    # ---- 2. Answer extraction & action penalty --------------------------
    pred_answer = extract_answer(solution_str)
    hedge = is_empty_or_hedge(pred_answer)
    action_penalty = ACTION_PENALTY_HEDGE if hedge else 0.0

    # ---- 3. Answer correctness (per reward version) ---------------------
    if pred_answer is None or hedge:
        answer_em = 0.0
        answer_f1 = 0.0
    else:
        answer_em = compute_em(pred_answer, gold_answer)
        answer_f1 = compute_token_f1(pred_answer, gold_answer)

    answer_score = answer_em if ANSWER_SCORE_KIND == "em" else answer_f1

    # ---- 4. Supporting-facts F1 (v2 only) -------------------------------
    sf_f1 = 0.0
    if REWARD_VERSION == "v2":
        parsed = extract_answer  # placeholder for static checkers
        # Pull <think> body via the same FORMAT_RE regex if format is ok;
        # otherwise score the full solution_str's pre-<answer> region.
        from prompt_templates import FORMAT_RE
        m = FORMAT_RE.search(solution_str)
        think_text = m.group(1) if m else solution_str
        pred_titles = extract_supporting_facts_from_think(think_text, candidate_titles)
        gold = _gold_titles(ground_truth)
        sf_f1 = compute_sf_f1(pred_titles, gold)

    # ---- 5. Compose final reward ----------------------------------------
    if REWARD_VERSION == "v1":
        correctness = answer_score                         # pure EM
    else:
        correctness = ALPHA * answer_score + (1.0 - ALPHA) * sf_f1

    length_penalty = soft_overlong_penalty(response_len)
    score = correctness + format_bonus + action_penalty + length_penalty

    return {
        "score": float(score),
        "answer_em": float(answer_em),
        "answer_f1": float(answer_f1),
        "sf_f1": float(sf_f1),
        "format_ok": float(fmt_state == "ok"),
        "format_malformed": float(fmt_state == "malformed"),
        "action_penalty": float(action_penalty),
        "length_penalty": float(length_penalty),
        "response_len": float(response_len),
        "reward_version": REWARD_VERSION,
    }
