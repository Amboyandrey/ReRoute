# ReRoute

An OpenAI-compatible LLM gateway that routes each request to the cheapest
model that can answer it well. A trained router model — not a hand-written
adapter — decides where each prompt goes: easy prompts fall to a small
self-hosted model, hard prompts go to Claude or GPT, and the router learns
from outcomes where to draw the line.

See [`docs/PLAN.md`](docs/PLAN.md) for the full project plan, architecture,
and build phases.

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

```bash
# Router service (defaults to a length-based rule baseline until Phase 2
# trains the real classifier)
cd router && uv sync --group dev && uv run reroute-serve

# Gateway (in another shell)
cd gateway && cp config.example.yaml config.yaml && go run ./cmd/gateway -config config.yaml
```

Then, with the .env values from `.env.example` exported:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

## Development

```bash
make test   # gateway (go vet + go test) and router (pytest)
make fmt    # gofmt + ruff format
make lint   # go vet + ruff check
```
