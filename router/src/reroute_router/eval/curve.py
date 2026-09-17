"""Cost-vs-quality curve and Performance Gap Recovered (PGR), the core
metrics this whole project is judged on. Every router — rule baseline,
supervised, bandit — is scored the same way so they land on one comparable
plot.

Expects a "long" dataframe with one row per (prompt_id, model) matching
:mod:`reroute_router.data.schema`, where every candidate model has been
scored on every prompt (true for RouterBench; for the router's own traffic
this only holds under offline replay / logged-outcome data).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

RoutingPolicy = Callable[[pd.DataFrame], pd.Series]
"""A policy maps a per-prompt-id group of candidate rows to the chosen
model name for that prompt."""


def apply_policy(df: pd.DataFrame, policy: RoutingPolicy) -> pd.DataFrame:
    """Runs a policy over every prompt and returns one row per prompt: the
    model it chose, and that choice's quality/cost.
    """
    rows = []
    for _, group in df.groupby("prompt_id", sort=False):
        chosen_model = policy(group)
        match = group.loc[group["model"] == chosen_model]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)


def always_model(model: str) -> RoutingPolicy:
    def policy(g: pd.DataFrame) -> str:
        return model

    return policy


def random_policy(models: list[str], seed: int = 0) -> RoutingPolicy:
    rng = np.random.default_rng(seed)

    def policy(g: pd.DataFrame) -> str:
        return rng.choice(models)

    return policy


def length_threshold_policy(cheap: str, strong: str, char_threshold: int = 200) -> RoutingPolicy:
    def policy(g: pd.DataFrame) -> str:
        prompt = g["prompt"].iloc[0]
        return cheap if len(prompt) <= char_threshold else strong

    return policy


def summarize(chosen: pd.DataFrame) -> dict[str, float]:
    """Mean quality and mean cost for a policy's chosen rows."""
    return {
        "mean_quality": float(chosen["quality"].mean()),
        "mean_cost_usd": float(chosen["cost_usd"].mean()),
        "n_prompts": int(len(chosen)),
    }


def performance_gap_recovered(
    router_quality: float, weak_quality: float, strong_quality: float
) -> float:
    """RouteLLM's PGR: how much of the quality gap between the weak and
    strong model a router recovers. 0.0 = as good as weak, 1.0 = as good
    as strong; can exceed 1.0 or go negative.
    """
    denom = strong_quality - weak_quality
    if denom == 0:
        return 1.0 if router_quality >= strong_quality else 0.0
    return (router_quality - weak_quality) / denom


def cost_quality_curve(
    df: pd.DataFrame,
    policies: dict[str, RoutingPolicy],
) -> pd.DataFrame:
    """Runs each named policy and returns a tidy dataframe of
    (policy, mean_quality, mean_cost_usd) — the raw data for the headline
    cost-vs-quality plot.
    """
    rows = []
    for name, policy in policies.items():
        chosen = apply_policy(df, policy)
        stats = summarize(chosen)
        rows.append({"policy": name, **stats})
    return pd.DataFrame(rows)


def headline_metric(
    router_quality: float, router_cost: float, strong_quality: float, strong_cost: float
) -> str:
    """Formats the project's target headline, e.g. "95% of GPT-level
    quality at 30% of the cost."
    """
    quality_pct = 100 * router_quality / strong_quality if strong_quality else 0
    cost_pct = 100 * router_cost / strong_cost if strong_cost else 0
    return f"{quality_pct:.0f}% of strong-model quality at {cost_pct:.0f}% of the cost"
