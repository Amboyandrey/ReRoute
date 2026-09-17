# ReRoute

An OpenAI-compatible LLM gateway that routes each request to the cheapest
model that can answer it well. A trained router model — not a hand-written
adapter — decides where each prompt goes: easy prompts fall to a small
self-hosted model, hard prompts go to Claude or GPT, and the router learns
from outcomes where to draw the line.

See [`docs/PLAN.md`](docs/PLAN.md) for the full project plan, architecture,
and build phases.

## Results

Three routers, evaluated on the same held-out RouterBench test prompts,
routing between `mistral-7b-chat` (weak/cheap) and `gpt-4-1106-preview`
(strong) — so they land on one comparable cost-quality curve.

| Router | Best real win over the baselines |
|---|---|
| Rules (length/random thresholds) | `length_threshold_200` reaches 98% of strong-model quality, but only ~5% cheaper — length alone is a weak signal here |
| Supervised (LoRA-tuned ModernBERT) | Matches `length_threshold_500`'s quality at **58% lower cost** |
| Contextual bandit (LinUCB) | Beats `length_threshold_500` at **63% lower cost** — a bigger win than the supervised router, from a simpler model, because it adapts online instead of committing to one fixed threshold |

- **Self-hosted cheap tier** (vLLM, RTX 4060 Laptop, 8GB VRAM): 72 → 959
  tokens/sec across concurrency 1→16, p95 latency only 2.8s → 3.4s, zero
  errors. [Full numbers](docs/benchmarks.md).
- **Deliberately induced failure case**: a biased reward signal (a lenient
  judge overscoring the cheap tier) shifts 7.5% more traffic to it and
  costs 0.037 in true quality — a reproducible demonstration that the
  bandit can't detect a bad reward signal on its own.
  [Details](docs/phase3_bandit.md).

Full write-ups: [Phase 1 baselines](docs/phase1_baseline_curve.md) ·
[Phase 2 supervised router](docs/phase2_supervised_curve.md) ·
[Phase 3 bandit](docs/phase3_bandit.md) ·
[Phase 6 Kubernetes](docs/phase6_kubernetes.md)

## Repo layout

```
ReRoute/
├── gateway/    Go — OpenAI-compatible data plane (streaming, retries,
│               fallback, per-key rate limits, Prometheus metrics)
├── router/     Python (uv) — dataset building, eval harness, supervised
│               (LoRA) router, contextual-bandit router, and the FastAPI
│               service the gateway calls for routing decisions
├── infra/      Helm chart, k3s bootstrap, Grafana dashboards
└── docs/       Plan and write-up, including failure-case analysis
```

## Quickstart

Three services, three shells. Verified end-to-end on an RTX 4060 Laptop
(8GB VRAM) — see [`docs/benchmarks.md`](docs/benchmarks.md) for numbers.

```bash
# 1. Cheap tier: vLLM serving the local model on your GPU
cd infra/vllm && uv venv --python 3.12 .venv && uv pip install vllm httpx --python .venv
./run.sh

# 2. Router service (defaults to a length-based rule baseline until Phase 2
#    trains the real classifier)
cd router && uv sync --group dev && uv run reroute-serve

# 3. Gateway
cd gateway && cp config.example.yaml config.yaml && go run ./cmd/gateway -config config.yaml
```

Then:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

The `mid` and `strong` tiers need `NEBIUS_API_KEY` / `ANTHROPIC_API_KEY` set
(see `.env.example`) — without them, only short prompts routed to `cheap`
will succeed.

## Development

```bash
make test   # gateway (go vet + go test) and router (pytest)
make fmt    # gofmt + ruff format
make lint   # go vet + ruff check
```
