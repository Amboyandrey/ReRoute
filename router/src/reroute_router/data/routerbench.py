"""Loader for RouterBench (withmartian/routerbench on Hugging Face),
normalized into the unified schema in :mod:`reroute_router.data.schema`.

RouterBench ships as pickled pandas DataFrames (not an arrow/parquet
dataset, so `datasets.load_dataset` doesn't work here) with one row per
prompt and, per candidate model, three columns:

- ``{model}``                — correctness score in [0, 1]
- ``{model}|model_response`` — the model's raw response text
- ``{model}|total_cost``     — cost in USD for that response

Verified against the actual dataset (2026-09-17): 36,497 rows, 11 models,
columns confirmed via `hf_hub_download` + `pandas.read_pickle` rather than
assumed from the dataset card.
"""

from __future__ import annotations

import pandas as pd

from reroute_router.data.schema import COLUMNS, validate

# The 11 models RouterBench scored every prompt against.
MODELS = [
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


def _download(variant: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "loading RouterBench requires the 'data' extra: uv sync --extra data"
        ) from e
    filename = f"routerbench_{variant}.pkl"
    return hf_hub_download(
        repo_id="withmartian/routerbench", filename=filename, repo_type="dataset"
    )


def load_routerbench(variant: str = "0shot", models: list[str] | None = None) -> pd.DataFrame:
    """Downloads a RouterBench variant ("0shot", "5shot", or "raw") and
    melts it into the unified schema: one row per (prompt, model).
    """
    path = _download(variant)
    df = pd.read_pickle(path)
    models = models or MODELS

    rows = []
    for _, row in df.iterrows():
        prompt_id = row["sample_id"]
        prompt_field = row["prompt"]
        # RouterBench stores prompt as [instruction, content]; join into one string.
        prompt = "\n\n".join(prompt_field) if isinstance(prompt_field, list) else str(prompt_field)

        for model in models:
            if model not in row or pd.isna(row[model]):
                continue
            cost_col = f"{model}|total_cost"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "model": model,
                    "quality": float(row[model]),
                    "cost_usd": float(row[cost_col])
                    if cost_col in row and pd.notna(row[cost_col])
                    else 0.0,
                    "latency_ms": float("nan"),
                    "source": f"routerbench_{variant}",
                }
            )

    out = pd.DataFrame(rows, columns=COLUMNS)
    validate(out)
    return out
