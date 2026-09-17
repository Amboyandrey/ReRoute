# ReRoute — Implementation Plan

ReRoute is an OpenAI-compatible LLM gateway that routes each request to the
cheapest model able to answer it well, using a router model trained on
outcome data rather than hand-written rules.

## Architecture

```
client ──OpenAI API──▶ Go gateway ──▶ router svc (Python, ONNX)  ──decision──┐
                          │                                                  │
                          ├──▶ cheap tier: vLLM (Qwen2.5-1.5B/3B)  ◀─────────┤
                          ├──▶ mid tier:   Nebius AI Studio (Llama/Qwen 70B) ◀┤
                          └──▶ strong tier: Claude / GPT            ◀─────────┘
                          │
                          └──▶ Prometheus (route decisions, cost, cost saved, latency) ──▶ Grafana
```

Nebius provides two things: an OpenAI-compatible hosted inference API (used
to cheaply run thousands of prompts through several open models when
building the dataset, and as a mid-tier route in production) and GPU VMs if
the local 4060 (8 GB) is too small for a given vLLM model.

## Repo layout

```
ReRoute/
├── gateway/           Go — data plane
├── router/            Python (uv) — everything ML
│   ├── data/          dataset builders (RouterBench, Arena, own set)
│   ├── eval/          cost–quality curve, PGR metric, plots
│   ├── supervised/    LoRA fine-tune of ModernBERT-base
│   ├── bandit/        contextual bandit + offline replay eval
│   └── serve/         FastAPI + ONNX Runtime, /route endpoint
├── infra/
│   ├── helm/reroute/  chart: gateway, router, vllm, prom, grafana
│   ├── k3s/           bootstrap scripts
│   └── grafana/       dashboard JSON
├── docs/              write-up, failure cases, curves
└── Makefile
```

## Phase 1 — Data + eval harness

- **RouterBench** (`withmartian/routerbench` on HF): ~36k prompts × 11
  models with quality scores + cost. Every model's response is scored, so
  this gives full counterfactual rewards — exactly what offline bandit
  evaluation needs.
- **LMSYS Arena preference data**
  (`lmsys/lmsys-arena-human-preference-55k`), used the way RouteLLM does, to
  derive "strong vs weak wins" labels.
- **Own set** (~3k prompts): mix of GSM8K, MMLU subset, HumanEval, and
  open-ended chat. Run through 3–4 models via Nebius (e.g. Qwen2.5-1.5B,
  Llama-3.1-8B, Llama-3.3-70B) plus one strong model. Graded with
  exact-match / unit tests where verifiable, LLM judge otherwise. Record
  tokens + $ per response.
- Unified schema: `(prompt, model, quality∈[0,1], cost_usd, latency_ms)` as
  parquet.
- **Eval harness** outputs the cost-vs-quality curve and RouteLLM's PGR
  (performance gap recovered) at fixed strong-model call rates. Baselines:
  always-cheap, always-strong, random, "prompt length > N → strong".
  Headline metric format: *"X% of strong-model quality at Y% of cost."*

## Phase 2 — Supervised router

- Fine-tune **ModernBERT-base** with LoRA (PEFT) as a binary/ordinal
  classifier: "will the cheap model score ≥ threshold on this prompt?" Also
  try a multi-head variant predicting each model's expected quality so one
  model covers all tiers. Fits on the 4060.
- Threshold sweep → the curve. Calibration check (reliability diagram)
  matters since the bandit later uses these probabilities.
- Export to **ONNX** so the serving path is light and Go-adjacent.

## Phase 3 — RL router (contextual bandit)

- Context = ModernBERT embedding (from Phase 2); actions = {cheap, mid,
  strong}; reward = `quality − λ·cost`.
- Algorithms: LinUCB / Linear Thompson Sampling (neural head optional).
  Warm-start from the supervised model.
- **Offline replay on RouterBench first** (all counterfactual rewards
  exist), then online on live gateway traffic via a feedback endpoint
  (`POST /v1/feedback` — thumbs, tests passed, judge score).
- Deliberately induce and document failure cases: noisy judge → drift to
  cheap tier; delayed feedback; distribution shift when a new prompt
  category appears; λ mis-tuned → all-strong collapse. Compare against
  supervised + rule baselines on the same curve.

## Phase 4 — Self-hosted cheap tier

- vLLM serving **Qwen2.5-1.5B-Instruct** (or 3B AWQ) on the local 4060 (8 GB
  is enough for that class; 7B is tight). For a 7B cheap tier, use a Nebius
  GPU VM instead.
- Benchmark script: tokens/sec at concurrency 1/4/16, p50/p95 latency, GPU
  memory — documented in `docs/`.

## Phase 5 — Go gateway

- `POST /v1/chat/completions` (+ `/v1/models`), OpenAI-compatible, **SSE
  streaming passthrough** with `usage` accounting.
- Calls the router service with a short timeout; if it's down, falls back to
  a static default route (gateway never depends on the ML path being up).
- Provider adapters: OpenAI-compatible (Nebius, vLLM, OpenAI) + Anthropic.
  Retries with backoff on 429/5xx, **fallback chain** (cheap → mid →
  strong) on provider failure.
- Per-API-key **token-bucket rate limits**, key → budget config in YAML.
- Prometheus metrics: `reroute_route_decisions_total{tier,reason}`,
  `reroute_cost_usd_total`, `reroute_cost_saved_usd_total` (vs
  always-strong), `reroute_upstream_latency_seconds` histogram,
  `reroute_fallbacks_total`.
- Tests: httptest fake upstreams for streaming, retry, fallback, rate limit.

## Phase 6 — Kubernetes

- **k3s** locally (single node, GPU passthrough via NVIDIA container
  toolkit) — Helm chart with gateway, router, vLLM, kube-prometheus-stack,
  Grafana.
- Grafana dashboard: route split over time, $ saved live, quality feedback,
  p95 per tier. Export JSON into repo.
- Optional: Nebius managed K8s for a "real" deployment with the vLLM pod on
  an L4/L40.

## Build order & milestones

1. Phase 1 → curve with baselines only (demo backbone)
2. Phase 2 → first learned point on the curve
3. Phase 5 gateway MVP (no router yet, static routing) — parallel with 2
4. Phase 4 vLLM → wire as cheap tier
5. Phase 3 bandit → offline, then online through the gateway
6. Phase 6 → dashboard showing live savings; write-up

## Open decisions

- **Strong-tier keys**: Anthropic and/or OpenAI key needed for the strong
  route and for the LLM judge (a Nebius-hosted 70B can judge in a pinch, but
  judging with the same family you route to biases the labels).
- **vLLM location**: local 4060 (1.5B–3B models) vs Nebius GPU VM (7B+).
