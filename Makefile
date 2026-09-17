.PHONY: help gateway-build gateway-test gateway-run router-sync router-test router-serve test fmt lint

help:
	@echo "Targets: gateway-build gateway-test gateway-run router-sync router-test router-serve test fmt lint"

# --- Gateway (Go) ---

gateway-build:
	cd gateway && go build ./...

gateway-test:
	cd gateway && go vet ./... && go test ./...

gateway-run:
	cd gateway && go run ./cmd/gateway -config config.yaml

# --- Router (Python) ---

router-sync:
	cd router && uv sync --group dev

router-test:
	cd router && uv run pytest

router-serve:
	cd router && uv run reroute-serve

# --- Combined ---

test: gateway-test router-test

fmt:
	cd gateway && gofmt -w .
	cd router && uv run ruff format .

lint:
	cd gateway && go vet ./...
	cd router && uv run ruff check .
