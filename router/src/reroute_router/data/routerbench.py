"""Loader for RouterBench (withmartian/routerbench on Hugging Face),
normalized into the unified schema in :mod:`reroute_router.data.schema`.

RouterBench ships wide: one row per prompt with a quality score and a cost
per candidate model as separate columns. We melt it into one row per
(prompt, model) pair to match our schema.

NOTE: verify the exact column names against the dataset card before first
run (``load_dataset("withmartian/routerbench")`` and inspect
``ds["train"].column_names``) — the mapping below is the expected shape as
of RouteLLM's writeup but Hugging Face dataset revisions can rename fields.
"""

from __future__ import annotations

import pandas as pd

from reroute_router.data.schema import COLUMNS, validate

# Models included in the RouterBench release. Update this list after
# inspecting the dataset card if it differs.
DEFAULT_MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]


def load_routerbench(split: str = "train", models: list[str] | None = None) -> pd.DataFrame:
    """Downloads RouterBench via the `datasets` library and melts it into
    the unified schema. Requires the `data` optional dependency group.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "loading RouterBench requires the 'data' extra: uv sync --extra data"
        ) from e

    ds = load_dataset("withmartian/routerbench", split=split)
    df = ds.to_pandas()
    models = models or DEFAULT_MODELS

    rows = []
    for _, row in df.iterrows():
        prompt_id = row.get("sample_id", row.name)
        prompt = row.get("prompt", row.get("question", ""))
        for model in models:
            score_col = next((c for c in row.index if c.startswith(model) and "score" in c), None)
            cost_col = next((c for c in row.index if c.startswith(model) and "cost" in c), None)
            if score_col is None:
                continue
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "model": model,
                    "quality": float(row[score_col]),
                    "cost_usd": float(row[cost_col]) if cost_col else 0.0,
                    "latency_ms": float("nan"),
                    "source": "routerbench",
                }
            )

    out = pd.DataFrame(rows, columns=COLUMNS)
    validate(out)
    return out
