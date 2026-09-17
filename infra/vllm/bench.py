#!/usr/bin/env python3
"""Phase 4 benchmark: tokens/sec, p50/p95 latency, and GPU memory for the
self-hosted cheap-tier vLLM server.

Usage:
    .venv/bin/python bench.py --base-url http://localhost:8001/v1 \
        --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1 4 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field

import httpx

PROMPT = (
    "Explain the difference between a stack and a queue, with a short "
    "example of each, in about 150 words."
)


@dataclass
class RunResult:
    latencies_s: list[float] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    wall_time_s: float = 0.0
    errors: int = 0


async def one_request(client: httpx.AsyncClient, base_url: str, model: str, max_tokens: int) -> tuple[float, int]:
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    latency = time.perf_counter() - start
    completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
    return latency, completion_tokens


async def run_concurrency(base_url: str, model: str, concurrency: int, n_requests: int, max_tokens: int) -> RunResult:
    result = RunResult()
    sem = asyncio.Semaphore(concurrency)

    async def worker(client: httpx.AsyncClient) -> None:
        async with sem:
            try:
                latency, tokens = await one_request(client, base_url, model, max_tokens)
                result.latencies_s.append(latency)
                result.completion_tokens.append(tokens)
            except Exception as e:  # noqa: BLE001
                result.errors += 1
                print(f"  request failed: {e}")

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(worker(client) for _ in range(n_requests)))
    result.wall_time_s = time.perf_counter() - start
    return result


def gpu_memory_mib() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def summarize(concurrency: int, result: RunResult) -> dict:
    if not result.latencies_s:
        return {"concurrency": concurrency, "errors": result.errors, "note": "all requests failed"}
    total_tokens = sum(result.completion_tokens)
    sorted_lat = sorted(result.latencies_s)
    p50 = statistics.median(sorted_lat)
    p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))]
    return {
        "concurrency": concurrency,
        "n_requests": len(result.latencies_s),
        "errors": result.errors,
        "wall_time_s": round(result.wall_time_s, 2),
        "tokens_per_sec": round(total_tokens / result.wall_time_s, 1) if result.wall_time_s else 0,
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "gpu_memory": gpu_memory_mib(),
    }


async def main_async(args: argparse.Namespace) -> None:
    results = []
    for c in args.concurrency:
        print(f"==> concurrency={c}")
        result = await run_concurrency(args.base_url, args.model, c, n_requests=c * args.requests_per_level, max_tokens=args.max_tokens)
        summary = summarize(c, result)
        print(json.dumps(summary, indent=2))
        results.append(summary)

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--requests-per-level", type=int, default=4, help="requests per concurrency level, multiplied by concurrency")
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
