# Local vLLM cheap tier

Serves the self-hosted "cheap" tier model on the laptop's own GPU (RTX 4060
Laptop, 8GB VRAM) via [vLLM](https://github.com/vllm-project/vllm)'s
OpenAI-compatible server. The gateway's `cheap` tier
(`gateway/config.example.yaml`) points at this by default.

We run vLLM natively (not in Docker) for local dev: the host's
`nvidia-container-toolkit` isn't installed, and vLLM's wheels bundle their
own CUDA runtime, so a plain venv against the host driver is simpler and
just as fast. The Helm chart (`infra/helm/reroute`) still uses the
`vllm/vllm-openai` Docker image for cluster deployment — see
`infra/k3s/bootstrap.sh` for installing the NVIDIA device plugin there.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install vllm httpx --python .venv
```

## Run

```bash
./run.sh
# or override the model/port:
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct" VLLM_PORT=8001 ./run.sh
```

Serves an OpenAI-compatible API at `http://localhost:8001/v1`.

## Benchmark

```bash
.venv/bin/python bench.py --concurrency 1 4 16
```

Reports tokens/sec, p50/p95 latency, and GPU memory at each concurrency
level. Results as of the first run on the RTX 4060 Laptop (8GB) with
Qwen2.5-1.5B-Instruct are in [`../../docs/benchmarks.md`](../../docs/benchmarks.md).
