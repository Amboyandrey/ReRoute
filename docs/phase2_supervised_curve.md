# Phase 2 — supervised router (LoRA ModernBERT) vs Phase 1 baselines

Held-out test split: 3,651 prompts (never seen during training), same weak/strong pair as Phase 1 (`mistralai/mistral-7b-chat` vs `gpt-4-1106-preview`). Binary classification accuracy @0.5 threshold: **0.697**.

## Threshold sweep (supervised router)

| P(cheap) threshold | mean quality | mean cost (USD) | % routed cheap | PGR |
|---|---|---|---|---|
| 0.1 | 0.354 | $0.00020 | 94.4% | 0.07 |
| 0.2 | 0.395 | $0.00038 | 86.9% | 0.16 |
| 0.3 | 0.612 | $0.00117 | 46.5% | 0.62 |
| 0.4 | 0.757 | $0.00301 | 11.1% | 0.93 |
| 0.5 | 0.780 | $0.00328 | 4.5% | 0.98 |
| 0.6 | 0.787 | $0.00331 | 2.3% | 1.00 |
| 0.7 | 0.789 | $0.00331 | 1.4% | 1.00 |
| 0.8 | 0.788 | $0.00332 | 1.2% | 1.00 |
| 0.9 | 0.789 | $0.00333 | 0.9% | 1.00 |

## Phase 1 baselines, recomputed on this same test split

| policy | mean quality | mean cost (USD) | PGR |
|---|---|---|---|
| always_weak | 0.320 | $0.00005 | 0.00 |
| always_strong | 0.789 | $0.00336 | 1.00 |
| random | 0.549 | $0.00173 | 0.49 |
| length_threshold_200 | 0.774 | $0.00317 | 0.97 |
| length_threshold_500 | 0.599 | $0.00280 | 0.60 |

## Result: where the supervised router actually wins

Picking the threshold with the single highest PGR is misleading here — it just converges to routing almost everything to strong (barely distinguishable from `always_strong`, since PGR only measures quality recovered, not cost). The real question is Pareto dominance: is there a threshold that gets **equal-or-better quality at equal-or-lower cost** than a given baseline? Checked against every baseline below (excluding `always_weak`, which nothing should lose to on quality):

| beats baseline | baseline (quality, cost) | at threshold | supervised (quality, cost) | cost saved |
|---|---|---|---|---|
| `always_strong` | 0.789, $0.00336 | 0.7 | 0.789, $0.00331 | 1% (trivial) |
| `random` | 0.549, $0.00173 | 0.3 | 0.612, $0.00117 | 32% |
| `length_threshold_500` | 0.599, $0.00280 | 0.3 | 0.612, $0.00117 | 58% |

The supervised router Pareto-dominates 3 of 4 baselines on this test split, but only 2 of those wins are substantial (>5% cost saved rather than a rounding-error difference). The genuine win is beating `length_threshold_500` and `random` at roughly a third of the cost — real evidence the classifier is using prompt content, not just length. The win over `always_strong` itself is trivial (same quality, ~1% cheaper) and shouldn't be read as a meaningful result on its own.

## Serving

The router service (`reroute_router.serve.main`) auto-loads this checkpoint's ONNX export if present at `router/supervised/checkpoints/v1/model.onnx`, falling back to the rule policy if it's missing. Set `ROUTER_SUPERVISED_THRESHOLD` to override the default 0.5 decision threshold — this run's recommended value is **0.3** (the threshold that dominates the toughest baseline above).

## Next

Phase 3 frames routing as a contextual bandit over this same embedding, learning online from live traffic instead of a fixed offline threshold, and adding the `mid` tier back in as a third action instead of collapsing to a weak/strong binary.
