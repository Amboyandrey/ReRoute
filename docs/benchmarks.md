# Phase 4 benchmarks — self-hosted cheap tier

Local vLLM server (`infra/vllm/`), RTX 4060 Laptop (8GB VRAM), serving
`Qwen/Qwen2.5-1.5B-Instruct` at `--max-model-len 8192`,
`--gpu-memory-utilization 0.85`. Benchmarked with `infra/vllm/bench.py`
(200 completion tokens per request, prompt: a ~25-word instruction).

## 2026-09-17 — first run

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.173.02, 8188 MiB VRAM.
vLLM 0.29.0, `VLLM_USE_FLASHINFER_SAMPLER=0` (see note below).

| Concurrency | Requests | Wall time (s) | Tokens/sec | p50 latency (s) | p95 latency (s) | GPU memory |
|---|---|---|---|---|---|---|
| 1  | 4  | 11.09 | 72.1  | 2.764 | 2.835 | 6444 / 8188 MiB |
| 4  | 16 | 13.06 | 245.0 | 3.257 | 3.304 | 6444 / 8188 MiB |
| 16 | 64 | 13.35 | 958.8 | 3.306 | 3.392 | 6444 / 8188 MiB |

Zero errors at every concurrency level. Throughput scales roughly linearly
with concurrency up to 16 concurrent requests with almost no per-request
latency penalty (p95 only goes from 2.8s to 3.4s from concurrency 1 to 16),
which is continuous batching doing its job — this GPU is nowhere near
saturated at 16 concurrent requests for a 1.5B model. GPU memory is flat
across all levels because `--gpu-memory-utilization 0.85` pre-allocates the
KV cache pool up front rather than growing it with load.

**Setup note:** `nvidia-container-toolkit` isn't installed on this machine,
so Docker GPU passthrough (`docker run --gpus`) doesn't work here — vLLM
runs natively against the host driver instead (see `infra/vllm/README.md`).
Separately, vLLM's default flashinfer sampler JIT-compiles a CUDA kernel on
first use and needs `nvcc` (the full CUDA toolkit, not just the driver),
which also isn't installed; `VLLM_USE_FLASHINFER_SAMPLER=0` falls back to
vLLM's built-in PyTorch sampler and avoids that requirement entirely, with
no observed throughput cost at this model size.
