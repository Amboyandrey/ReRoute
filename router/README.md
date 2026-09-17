# reroute-router

Python side of [ReRoute](../README.md): dataset building, the cost-vs-quality
eval harness, the supervised (LoRA) router, the contextual-bandit router, and
the FastAPI service the Go gateway calls to get a routing decision.

## Layout

- `src/reroute_router/data/` — dataset builders (RouterBench, LMSYS Arena, our own set)
- `src/reroute_router/eval/` — cost-vs-quality curve, PGR metric, baselines
- `src/reroute_router/supervised/` — LoRA fine-tune of ModernBERT, ONNX export
- `src/reroute_router/bandit/` — contextual bandit + offline replay evaluation
- `src/reroute_router/serve/` — FastAPI `/route` service called by the gateway

## Setup

```bash
uv sync --group dev              # base + dev tools
uv sync --extra data --group dev # + dataset building
uv sync --extra train --group dev # + torch/transformers/peft for training
uv sync --extra bandit --group dev # + bandit deps
```

## Run the router service

```bash
uv run reroute-serve   # serves on :9000, see ROUTER_PORT
```

## Tests

```bash
uv run pytest
```
