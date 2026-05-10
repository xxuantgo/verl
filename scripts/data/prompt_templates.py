# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Shared prompt templates and format helpers for the HotpotQA pipeline.

Three pipeline stages must agree on the *student* answer format
``<think>...</think><answer>...</answer>``:

  * SFT preprocessing (``preprocess_hotpot_sft.py``) — bakes the format into
    the assistant turn the student is trained to imitate.
  * RL rollout (Phase 3) — the same system+user prompt is fed to the policy
    so it keeps producing the format learned at SFT time.
  * RL reward — extracts ``<answer>`` from the policy completion to score it.

If any of these drift apart, the model learns one format and is graded on
another. Centralising the strings + the regex here is the cheapest way to
prevent that.

The *teacher* templates are intentionally separate: the synthesizer leaks the
gold answer into the prompt to coax a faithful CoT, which the student must
never see.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Format specification (single source of truth)
# ---------------------------------------------------------------------------

# Used inside any system prompt that asks for the student-facing format.
FORMAT_SPEC = (
    "You MUST respond in EXACTLY this format:\n"
    "<think>step-by-step reasoning</think><answer>concise final answer</answer>"
)

# Full <think>...</think><answer>...</answer> match. DOTALL so reasoning may
# contain newlines. Used by synthesize_cot.py to validate teacher output and
# by anything else that wants to verify both halves are present.
FORMAT_RE = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    re.DOTALL,
)

# Just the <answer>...</answer> span. Used by the RL reward function: the
# policy may emit malformed think tags but as long as a final <answer> exists
# we can still score it.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_format(text: str) -> tuple[str, str] | None:
    """Return (think, answer) if the full format parses, else None."""
    m = FORMAT_RE.search(text)
    if m is None:
        return None
    return m.group(1).strip(), m.group(2).strip()


def extract_answer(text: str) -> str | None:
    """Return the contents of the last <answer>...</answer> span, or None.

    Uses the *last* match so a stray ``<answer>`` mentioned in the reasoning
    cannot shadow the real final answer.
    """
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# Student-facing prompts (SFT + RL rollout share these verbatim)
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = (
    "You are a multi-hop reasoning assistant. Read the passages and answer the "
    "question.\n\n" + FORMAT_SPEC
)

STUDENT_USER_TEMPLATE = "Passages:\n{passages}\n\nQuestion: {question}"


def render_student_user(passages: str, question: str) -> str:
    return STUDENT_USER_TEMPLATE.format(passages=passages, question=question)


# ---------------------------------------------------------------------------
# Teacher prompts (synthesize_cot.py only — leaks the gold answer)
# ---------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = (
    "You are a careful multi-hop reasoning assistant. Given a question and a set of "
    "passages, decompose the problem, cite the relevant facts, and arrive at the "
    "final answer.\n\n"
    "You MUST respond in EXACTLY this format and nothing else:\n"
    "<think>step-by-step reasoning that uses the passages</think>"
    "<answer>concise final answer (no full sentence, no extra punctuation)</answer>\n"
    "Do not include any text outside the two tags."
)

TEACHER_USER_TEMPLATE = (
    "Passages:\n{passages}\n\n"
    "Question: {question}\n\n"
    "The gold answer is known to be: {gold}\n"
    "Produce a faithful chain of thought that legitimately derives this answer "
    "from the passages, then output the answer. If the passages do not support "
    "the gold answer, still attempt the best reasoning you can without fabricating "
    "facts that contradict them."
)


def render_teacher_user(passages: str, question: str, gold: str) -> str:
    return TEACHER_USER_TEMPLATE.format(passages=passages, question=question, gold=gold)
