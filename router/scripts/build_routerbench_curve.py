#!/usr/bin/env python3
"""Phase 1: pull RouterBench, cache it as parquet, and produce the first
real cost-vs-quality curve with baseline policies.

Usage:
    uv run --extra data --extra bandit python scripts/build_routerbench_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reroute_router.data.routerbench import load_routerbench  # noqa: E402
from reroute_router.eval.curve import (  # noqa: E402
    always_model,
    cost_quality_curve,
    headline_metric,
    length_threshold_policy,
    performance_gap_recovered,
    random_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "router" / "data" / "processed"
DOCS_PATH = REPO_ROOT / "docs" / "phase1_baseline_curve.md"


def per_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(mean_quality=("quality", "mean"), mean_cost_usd=("cost_usd", "mean"))
        .sort_values("mean_cost_usd")
    )


def main() -> None:
    print("Downloading + normalizing RouterBench (0shot)...")
    df = load_routerbench(variant="0shot")
    print(
        f"Loaded {len(df):,} (prompt, model) rows, {df['prompt_id'].nunique():,} unique prompts, "
        f"{df['model'].nunique()} models"
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "routerbench_0shot.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Cached to {out_path.relative_to(REPO_ROOT)}")

    stats = per_model_stats(df)
    print("\nPer-model mean quality / mean cost (sorted by cost):")
    print(stats.to_string())

    weak_model = stats.index[0]  # cheapest
    strong_model = stats.loc[stats["mean_quality"].idxmax()].name  # highest quality

    weak_quality = float(stats.loc[weak_model, "mean_quality"])
    weak_cost = float(stats.loc[weak_model, "mean_cost_usd"])
    strong_quality = float(stats.loc[strong_model, "mean_quality"])
    strong_cost = float(stats.loc[strong_model, "mean_cost_usd"])

    print(f"\nweak (cheapest) model:   {weak_model}")
    print(f"  quality={weak_quality:.3f}, cost=${weak_cost:.5f}")
    print(f"strong (best) model:     {strong_model}")
    print(f"  quality={strong_quality:.3f}, cost=${strong_cost:.5f}")

    policies = {
        "always_weak": always_model(weak_model),
        "always_strong": always_model(strong_model),
        "random": random_policy([weak_model, strong_model], seed=0),
        "length_threshold_200": length_threshold_policy(
            weak_model, strong_model, char_threshold=200
        ),
        "length_threshold_500": length_threshold_policy(
            weak_model, strong_model, char_threshold=500
        ),
    }

    # Restrict to the two-model comparison so `apply_policy` (which needs
    # the chosen model present in each prompt's candidate rows) works over
    # exactly the models each baseline can pick from.
    two_model_df = df[df["model"].isin([weak_model, strong_model])]
    curve = cost_quality_curve(two_model_df, policies)
    curve["pgr"] = curve["mean_quality"].apply(
        lambda q: performance_gap_recovered(q, weak_quality, strong_quality)
    )
    curve = curve.sort_values("mean_cost_usd")

    print("\nCost-quality curve (weak vs strong baselines):")
    print(curve.to_string(index=False))

    best_length_row = (
        curve[curve["policy"].str.startswith("length_threshold")]
        .sort_values("mean_quality", ascending=False)
        .iloc[0]
    )
    headline = headline_metric(
        router_quality=best_length_row["mean_quality"],
        router_cost=best_length_row["mean_cost_usd"],
        strong_quality=strong_quality,
        strong_cost=strong_cost,
    )
    print(f"\nHeadline ({best_length_row['policy']}): {headline}")

    write_report(df, stats, curve, weak_model, strong_model, headline, best_length_row, strong_cost)
    print(f"\nWrote {DOCS_PATH.relative_to(REPO_ROOT)}")


def write_report(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    curve: pd.DataFrame,
    weak_model: str,
    strong_model: str,
    headline: str,
    best_length_row: pd.Series,
    strong_cost: float,
) -> None:
    lines = [
        "# Phase 1 — RouterBench baseline cost-quality curve",
        "",
        f"RouterBench (0-shot), {len(df):,} (prompt, model) rows, "
        f"{df['prompt_id'].nunique():,} unique prompts, {df['model'].nunique()} models. "
        "Built with `router/scripts/build_routerbench_curve.py`; raw data cached at "
        "`router/data/processed/routerbench_0shot.parquet` (gitignored — re-run the script to "
        "regenerate).",
        "",
        "This is the **first real point on the project's headline metric** — every later router "
        "(supervised, then bandit) is judged against these same baselines on the same curve.",
        "",
        "## Per-model mean quality / cost",
        "",
        "| model | mean quality | mean cost (USD) |",
        "|---|---|---|",
    ]
    for model, row in stats.iterrows():
        lines.append(f"| `{model}` | {row['mean_quality']:.3f} | ${row['mean_cost_usd']:.5f} |")

    lines += [
        "",
        f"Weak (cheapest) model: `{weak_model}`. "
        f"Strong (highest mean quality) model: `{strong_model}`.",
        "",
        "## Baseline policies (weak vs strong two-way routing)",
        "",
        "| policy | mean quality | mean cost (USD) | n prompts | PGR |",
        "|---|---|---|---|---|",
    ]
    for _, row in curve.iterrows():
        lines.append(
            f"| {row['policy']} | {row['mean_quality']:.3f} | ${row['mean_cost_usd']:.5f} | "
            f"{int(row['n_prompts'])} | {row['pgr']:.2f} |"
        )

    savings_pct = 100 * (1 - best_length_row["mean_cost_usd"] / strong_cost)
    lines += [
        "",
        f"**Headline** ({best_length_row['policy']}, the best baseline so far): {headline}",
        "",
        "PGR (Performance Gap Recovered) is RouteLLM's metric: 0.0 = as good as the weak model, "
        "1.0 = as good as the strong model.",
        "",
        "**Honest read of this baseline:** prompt length turns out to be a weak signal here. "
        "`length_threshold_200` gets close to strong-model quality mostly because it routes "
        "*almost everything* to the strong model at that threshold — real 401K-row "
        "RouterBench prompts are long — so it only saves about "
        f"{savings_pct:.0f}% of cost versus always-strong, "
        "not because it's smart about which prompts the weak model can actually handle. "
        "`length_threshold_500` shows the flip side: raising the threshold saves more but drops "
        "quality closer to random, because plenty of long prompts are still easy and plenty of "
        "short ones are hard. This is exactly the gap Phase 2's trained router needs to close: a "
        "real win looks like meaningfully *more* cost saved at the *same or better* quality, not "
        "just landing high on this curve by routing almost everything to the strong model.",
        "",
        "## Next",
        "",
        "Phase 2 fine-tunes ModernBERT with LoRA to predict per-prompt quality directly from "
        "content, instead of using prompt length as a proxy — should land above and to the left "
        "of the length-threshold points here (same quality, less cost) using the same weak/strong "
        "pair, or the full 11-model set, for comparison.",
    ]
    DOCS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
