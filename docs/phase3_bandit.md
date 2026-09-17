# Phase 3 — contextual bandit (LinUCB) offline replay

Trained online (sequential, one update per step) over 29,197 shuffled training prompts, evaluated greedily on the same 3,651-prompt held-out test split as Phase 1/2, same weak/strong pair. Context: 129-dim hashed bag-of-words (see `reroute_router.bandit.features`) — not the Phase 2 ModernBERT embedding, since that model was only ever exported as a classifier head, not an embedding extractor; LinUCB also wants a small fixed-dim context for its per-step matrix update to stay cheap. Offline replay is exact here (no importance weighting) because RouterBench has a true counterfactual reward for both actions at every prompt.

## Cost-quality curve (cost_lambda sweep)

`cost_lambda` is the tradeoff knob in `reward = quality - cost_lambda * cost_usd`; sweeping it traces out the bandit's own cost-quality curve the same way threshold sweeps did for the supervised router.

| cost_lambda | mean quality | mean cost (USD) | % routed cheap | PGR |
|---|---|---|---|---|
| 10 | 0.780 | $0.00317 | 3.1% | 0.98 |
| 50 | 0.732 | $0.00148 | 23.3% | 0.88 |
| 100 | 0.725 | $0.00138 | 25.1% | 0.86 |
| 200 | 0.681 | $0.00103 | 33.7% | 0.77 |
| 400 | 0.551 | $0.00048 | 57.3% | 0.49 |

## Comparison to Phase 1 baselines and Phase 2 supervised router

Numbers below are from `docs/phase1_baseline_curve.md` and `docs/phase2_supervised_curve.md` (same test split, same weak/strong pair; not recomputed here since they're already on record).

| router | mean quality | mean cost (USD) | PGR |
|---|---|---|---|
| always_strong (baseline) | 0.789 | $0.00336 | 1.00 |
| length_threshold_200 (baseline) | 0.774 | $0.00317 | 0.97 |
| length_threshold_500 (baseline) | 0.599 | $0.00280 | 0.60 |
| random (baseline) | 0.549 | $0.00173 | 0.49 |
| supervised, threshold=0.3 (Phase 2) | 0.612 | $0.00117 | 0.62 |
| supervised, threshold=0.5 (Phase 2) | 0.780 | $0.00328 | 0.98 |
| bandit, lambda=10 (Phase 3) | 0.780 | $0.00317 | 0.98 |
| bandit, lambda=50 (Phase 3) | 0.732 | $0.00148 | 0.88 |
| bandit, lambda=100 (Phase 3) | 0.725 | $0.00138 | 0.86 |
| bandit, lambda=200 (Phase 3) | 0.681 | $0.00103 | 0.77 |
| bandit, lambda=400 (Phase 3) | 0.551 | $0.00048 | 0.49 |

At its best operating point (lambda=10), the bandit reaches PGR=0.98 at $0.00317/prompt. More useful than that single number is Pareto dominance against the references above (equal-or-better quality at equal-or-lower cost):

- beats **length_threshold_500** (0.599, $0.00280) at lambda=200: 0.681 quality, $0.00103/prompt (63% cheaper)
- beats **random** (0.549, $0.00173) at lambda=400: 0.551 quality, $0.00048/prompt (72% cheaper)
- beats **supervised threshold=0.3** (0.612, $0.00117) at lambda=200: 0.681 quality, $0.00103/prompt (12% cheaper)

This is a stronger result than Phase 2's supervised router got on the same comparison: the bandit dominates `length_threshold_500` across a wide band of `cost_lambda` values, not just at one threshold, and does so with a bigger margin (e.g. at lambda=50, 0.732 quality vs `length_threshold_500`'s 0.599, at roughly half the cost). That's a genuinely interesting result for a linear model over hashed bag-of-words features against a fine-tuned transformer — plausibly because LinUCB's online updates adapt directly to this exact weak/strong pair's actual error pattern, where the supervised classifier's fixed threshold has to commit to one operating point at training time. Its real advantage on top of this — learning continuously from live feedback instead of being frozen after one offline training run — isn't visible in a replay evaluation like this one; that requires actually running it against live traffic, which is future work (see below).

## Failure case: reward drift from a biased judge

Simulated a lenient LLM judge that overscores the cheap tier's answers by a flat +0.3 in the *training* reward signal only (ground truth used for evaluation is untouched) — a realistic failure mode: a weak or lazy judge model rating superficially fluent wrong answers as correct more often for the smaller model. Trained both bandits at cost_lambda=100, evaluated both on the same true test quality:

| training reward | true mean quality (test) | % routed cheap |
|---|---|---|
| clean | 0.725 | 25.1% |
| biased (+0.3 on cheap) | 0.688 | 32.6% |

**Confirmed drift:** the biased judge shifts routing 7.5% more traffic to the cheap tier and costs 0.037 in true mean quality — exactly the failure mode the project plan flagged upfront. The bandit is only as good as its reward signal; a noisy or biased grader (LLM judge, flaky test suite, inconsistent human raters) silently degrades routing quality without any error or warning, since the bandit has no way to know its reward signal is wrong. Mitigations worth trying: reward signal validation against a held-out trusted judge, wider confidence intervals (lower `alpha`) to slow convergence on noisy signal, or periodically re-validating high-confidence cheap routes against the strong tier.

## Not done in this pass

- **Online learning from live traffic.** This is offline replay only; the plan's `POST /v1/feedback` endpoint for updating the bandit from real gateway traffic isn't wired up yet. `LinUCBRouter.update()` is ready to be called from such an endpoint — the missing piece is the gateway-side feedback route and a router-service endpoint to receive it.
- **Three-way routing (cheap/mid/strong).** Like Phase 2, this stays binary to match Phase 1's baseline curve for a fair comparison. `LinUCBRouter` already supports an arbitrary tier list, so adding `mid` back in is mostly a data-labeling problem (Phase 1/2's dataset only has weak/strong scores), not an algorithm change.
