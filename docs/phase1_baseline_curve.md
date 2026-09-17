# Phase 1 — RouterBench baseline cost-quality curve

RouterBench (0-shot), 401,467 (prompt, model) rows, 36,497 unique prompts, 11 models. Built with `router/scripts/build_routerbench_curve.py`; raw data cached at `router/data/processed/routerbench_0shot.parquet` (gitignored — re-run the script to regenerate).

This is the **first real point on the project's headline metric** — every later router (supervised, then bandit) is judged against these same baselines on the same curve.

## Per-model mean quality / cost

| model | mean quality | mean cost (USD) |
|---|---|---|
| `mistralai/mistral-7b-chat` | 0.306 | $0.00005 |
| `WizardLM/WizardLM-13B-V1.2` | 0.431 | $0.00007 |
| `mistralai/mixtral-8x7b-chat` | 0.547 | $0.00013 |
| `meta/code-llama-instruct-34b-chat` | 0.202 | $0.00017 |
| `zero-one-ai/Yi-34B-Chat` | 0.647 | $0.00019 |
| `meta/llama-2-70b-chat` | 0.329 | $0.00020 |
| `claude-instant-v1` | 0.598 | $0.00023 |
| `gpt-3.5-turbo-1106` | 0.619 | $0.00024 |
| `claude-v1` | 0.630 | $0.00214 |
| `claude-v2` | 0.636 | $0.00242 |
| `gpt-4-1106-preview` | 0.781 | $0.00329 |

Weak (cheapest) model: `mistralai/mistral-7b-chat`. Strong (highest mean quality) model: `gpt-4-1106-preview`.

## Baseline policies (weak vs strong two-way routing)

| policy | mean quality | mean cost (USD) | n prompts | PGR |
|---|---|---|---|---|
| always_weak | 0.306 | $0.00005 | 36497 | 0.00 |
| random | 0.542 | $0.00167 | 36497 | 0.50 |
| length_threshold_500 | 0.593 | $0.00279 | 36497 | 0.60 |
| length_threshold_200 | 0.768 | $0.00312 | 36497 | 0.97 |
| always_strong | 0.781 | $0.00329 | 36497 | 1.00 |

**Headline** (length_threshold_200, the best baseline so far): 98% of strong-model quality at 95% of the cost

PGR (Performance Gap Recovered) is RouteLLM's metric: 0.0 = as good as the weak model, 1.0 = as good as the strong model.

**Honest read of this baseline:** prompt length turns out to be a weak signal here. `length_threshold_200` gets close to strong-model quality mostly because it routes *almost everything* to the strong model at that threshold — real 401K-row RouterBench prompts are long — so it only saves about 5% of cost versus always-strong, not because it's smart about which prompts the weak model can actually handle. `length_threshold_500` shows the flip side: raising the threshold saves more but drops quality closer to random, because plenty of long prompts are still easy and plenty of short ones are hard. This is exactly the gap Phase 2's trained router needs to close: a real win looks like meaningfully *more* cost saved at the *same or better* quality, not just landing high on this curve by routing almost everything to the strong model.

## Next

Phase 2 fine-tunes ModernBERT with LoRA to predict per-prompt quality directly from content, instead of using prompt length as a proxy — should land above and to the left of the length-threshold points here (same quality, less cost) using the same weak/strong pair, or the full 11-model set, for comparison.
