# Copyright 2026 verl multi-hop QA agentic RL project
# Licensed under the Apache License, Version 2.0
"""Synthesize chain-of-thought (CoT) supervision data for SFT cold start.

Reads a QA dataset (HotpotQA by default; pluggable adapters for NQ/2Wiki/MuSiQue)
and queries a teacher LLM (DeepSeek by default) to produce
``<think>...</think><answer>...</answer>`` traces. Output is one JSONL record per
example, written incrementally so the run is resumable.

Example:
    python scripts/data/synthesize_cot.py \\
        --dataset hotpot \\
        --input  ~/data/hotpotqa/hotpot_train_v1.1.json \\
        --output ~/data/hotpotqa/sft_cot_raw.jsonl \\
        --n 5000 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    parse_format,
    render_teacher_user,
)

try:
    import httpx
except ImportError as e:
    raise SystemExit("httpx is required: pip install httpx") from e

# Load DEEPSEEK_API_KEY etc. from a project-root .env if present. Optional dep:
# `pip install python-dotenv`. Falls back silently if not installed so the
# script still runs when the env is exported by other means.
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for _candidate in (_here.parent.parent.parent / ".env", _here.parent.parent / ".env"):
        if _candidate.is_file():
            load_dotenv(_candidate)
            break
except ImportError:
    pass


@dataclass
class Sample:
    qid: str
    question: str
    answer: str
    passages: str
    supporting_facts: list[Any]
    meta: dict[str, Any]


def _format_hotpot_passages(context: Any) -> str:
    """Render HotpotQA context to a flat string.

    Supports both schemas in the wild:
      * Stanford official dump: [[title, [sent, sent, ...]], ...]
      * HuggingFace ``hotpot_qa`` config: {"title": [...], "sentences": [[...]]}
    """
    parts: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for title, sents in zip(titles, sentences):
            body = "".join(sents) if isinstance(sents, list) else str(sents)
            parts.append(f"[{title}] {body}")
    else:
        for title, sents in context:
            body = "".join(sents) if isinstance(sents, list) else str(sents)
            parts.append(f"[{title}] {body}")
    return "\n".join(parts)


def _normalize_supporting_facts(sf: Any) -> list[list[Any]]:
    """Coerce supporting_facts into the [[title, sent_id], ...] shape so the
    downstream parquet always sees one schema. HF dumps {"title":[], "sent_id":[]}.
    """
    if isinstance(sf, dict):
        return [[t, s] for t, s in zip(sf.get("title", []), sf.get("sent_id", []))]
    return list(sf or [])


def load_hotpot(path: str) -> list[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for ex in raw:
        # Stanford dump uses "_id"; HF dump uses "id".
        qid = ex.get("_id") or ex.get("id")
        out.append(
            Sample(
                qid=qid,
                question=ex["question"],
                answer=ex["answer"],
                passages=_format_hotpot_passages(ex.get("context", [])),
                supporting_facts=_normalize_supporting_facts(ex.get("supporting_facts")),
                meta={
                    "type": ex.get("type"),
                    "level": ex.get("level"),
                },
            )
        )
    return out


DATASET_LOADERS: dict[str, Callable[[str], list[Sample]]] = {
    "hotpot": load_hotpot,
}


# ---------------------------------------------------------------------------
# Teacher API client
# ---------------------------------------------------------------------------


async def call_deepseek(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    sample: Sample,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_teacher_user(
                    passages=sample.passages,
                    question=sample.question,
                    gold=sample.answer,
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# Format parsing lives in prompt_templates.parse_format so the SFT/RL stages
# use the exact same regex.


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a server-supplied ``Retry-After`` header into seconds.

    Per RFC 7231 the header is either an integer number of seconds or an
    HTTP-date. DeepSeek (like most OpenAI-compat gateways) returns the integer
    form on 429, but we handle both so we don't silently fall back to dumb
    exponential backoff if they ever switch.
    """
    raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_done_qids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                continue
    return done


async def worker(
    name: int,
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    out_fh,
    write_lock: asyncio.Lock,
    counters: dict[str, int],
):
    while True:
        sample: Sample | None = await queue.get()
        if sample is None:
            queue.task_done()
            return
        attempt = 0
        record: dict[str, Any] | None = None
        while attempt < args.max_retries:
            attempt += 1
            try:
                text = await call_deepseek(
                    client,
                    args.base_url,
                    args.api_key,
                    args.model,
                    sample,
                    args.max_tokens,
                    args.temperature,
                )
                parsed = parse_format(text)
                record = {
                    "qid": sample.qid,
                    "raw_question": sample.question,
                    "raw_answer": sample.answer,
                    "passages": sample.passages,
                    "supporting_facts": sample.supporting_facts,
                    "meta": sample.meta,
                    "raw_completion": text,
                    "think": parsed[0] if parsed else None,
                    "answer": parsed[1] if parsed else None,
                    "format_ok": parsed is not None,
                }
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # 429 = rate-limited, 5xx = transient server error → retry.
                # Any other 4xx (401 auth, 400 bad model id, 404, …) is the
                # caller's fault and won't fix itself; fail fast so the user
                # sees the real error instead of 4× exponential backoff.
                if status == 429 or status >= 500:
                    wait = _retry_after_seconds(e.response)
                    if wait is None:
                        wait = min(2 ** attempt, 30)
                    wait += random.random()  # jitter to de-sync workers
                    print(
                        f"[worker {name}] qid={sample.qid} attempt={attempt} "
                        f"status={status} retry-after={wait:.1f}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                else:
                    body = e.response.text[:200].replace("\n", " ")
                    print(
                        f"[worker {name}] qid={sample.qid} non-retryable status={status} body={body!r}",
                        file=sys.stderr,
                    )
                    break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                wait = min(2 ** attempt, 30) + random.random()
                print(
                    f"[worker {name}] qid={sample.qid} attempt={attempt} net-error={e!r} sleep={wait:.1f}",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
            except Exception as e:
                # Unexpected (e.g. JSON parse, key error in response). Don't
                # spin on these — log loudly and move on.
                print(
                    f"[worker {name}] qid={sample.qid} attempt={attempt} unexpected={e!r}",
                    file=sys.stderr,
                )
                break
        if record is None:
            counters["failed"] += 1
            queue.task_done()
            continue
        async with write_lock:
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            counters["written"] += 1
            if counters["written"] % 50 == 0:
                print(
                    f"[progress] written={counters['written']} failed={counters['failed']}",
                    file=sys.stderr,
                )
        queue.task_done()


async def amain(args: argparse.Namespace) -> None:
    loader = DATASET_LOADERS[args.dataset]
    samples = loader(args.input)
    print(f"loaded {len(samples)} examples from {args.input}", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.n > 0:
        samples = samples[: args.n]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_qids(out_path)
    if done:
        print(f"resume: skipping {len(done)} already-written qids", file=sys.stderr)
        samples = [s for s in samples if s.qid not in done]

    if not samples:
        print("nothing to do", file=sys.stderr)
        return

    queue: asyncio.Queue = asyncio.Queue()
    for s in samples:
        queue.put_nowait(s)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    counters = {"written": 0, "failed": 0}
    write_lock = asyncio.Lock()
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as out_fh:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            workers = [
                asyncio.create_task(worker(i, queue, client, args, out_fh, write_lock, counters))
                for i in range(args.concurrency)
            ]
            await queue.join()
            for w in workers:
                await w
    dt = time.time() - t0
    print(
        f"done in {dt:.1f}s written={counters['written']} failed={counters['failed']}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_LOADERS), default="hotpot")
    p.add_argument("--input", required=True, help="Path to raw dataset (e.g. hotpot_train_v1.1.json)")
    p.add_argument("--output", required=True, help="JSONL output path (appended to; resumable)")
    p.add_argument("--n", type=int, default=5000, help="Number of examples to sample (-1 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    # 官方 V4 系列有两个正式 model ID：
    # deepseek-v4-flash（低延迟低成本）和 deepseek-v4-pro（更强推理能力，适合复杂多步工作流）
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-pro"))
    p.add_argument(
        "--base-url",
        default=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com"),
    )
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()
    if not args.api_key:
        raise SystemExit("set DEEPSEEK_API_KEY env var or pass --api-key")
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)
    return args


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
