#!/usr/bin/env bash
# Serves the cheap-tier model locally on the laptop's GPU via vLLM.
# 8GB VRAM (e.g. RTX 4060 Laptop) comfortably fits Qwen2.5-1.5B-Instruct;
# bump --gpu-memory-utilization down if you're running other GPU workloads
# alongside it.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# flashinfer's sampler JIT-compiles a CUDA kernel on first use, which needs
# nvcc (the CUDA toolkit, not just the driver). We only have the driver
# installed, so fall back to vLLM's built-in PyTorch sampler instead of
# installing the full toolkit just for this.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec .venv/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN"
