"""Online routing service.

The Go gateway calls ``POST /route`` with a prompt and gets back the tier
name that should handle it. This module starts with a length-based rule
baseline so the gateway has something real to call from day one; Phase 2
swaps in the trained ModernBERT classifier (loaded as ONNX, so serving
doesn't need torch/transformers/peft) and Phase 3 swaps that for the
contextual bandit, both behind the same HTTP contract so the gateway never
has to change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("reroute.router")

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "supervised" / "checkpoints" / "v1"

app = FastAPI(title="ReRoute router", version="0.1.0")


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    tier: str
    confidence: float


class Policy:
    """Pluggable routing policy. Swapped out as later phases land:
    RulePolicy -> SupervisedPolicy (Phase 2) -> BanditPolicy (Phase 3).
    """

    def decide(self, prompt: str) -> RouteResponse:
        raise NotImplementedError


class RulePolicy(Policy):
    """Baseline: route by prompt length as a stand-in until a trained
    model is wired in. Not meant to be a good router — it exists so the
    gateway, metrics, and dashboards all have real signal to build against
    immediately.
    """

    def __init__(self, short_chars: int = 200, long_chars: int = 1500) -> None:
        self.short_chars = short_chars
        self.long_chars = long_chars

    def decide(self, prompt: str) -> RouteResponse:
        n = len(prompt)
        if n <= self.short_chars:
            return RouteResponse(tier="cheap", confidence=0.5)
        if n <= self.long_chars:
            return RouteResponse(tier="mid", confidence=0.5)
        return RouteResponse(tier="strong", confidence=0.5)


class SupervisedPolicy(Policy):
    """Phase 2: ModernBERT (LoRA-tuned) binary classifier exported to ONNX,
    predicting P(cheap tier is good enough for this prompt). Loads
    `model.onnx` + `tokenizer.json` from a training checkpoint directory —
    no torch/transformers/peft needed at serving time.
    """

    def __init__(self, checkpoint_dir: Path, threshold: float = 0.5, max_length: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # Cap threading explicitly rather than let onnxruntime grab every
        # logical core by default — this service handles one request at a
        # time per call, and the gateway calls it with a short timeout, so
        # predictable low-thread latency matters more than raw throughput.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(os.environ.get("ROUTER_ONNX_THREADS", "4"))
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(checkpoint_dir / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.threshold = threshold

    def decide(self, prompt: str) -> RouteResponse:
        encoding = self.tokenizer.encode(prompt)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        (logits,) = self.session.run(
            ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        probs = _softmax(logits[0])
        cheap_prob = float(probs[1])  # label 1 == "route_cheap", see train.py

        tier = "cheap" if cheap_prob >= self.threshold else "strong"
        return RouteResponse(
            tier=tier, confidence=cheap_prob if tier == "cheap" else 1 - cheap_prob
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _load_policy() -> Policy:
    """Loads the supervised policy if a trained checkpoint is present,
    otherwise falls back to the rule baseline. Mirrors the gateway's own
    fallback-when-unavailable design: the serving path degrades gracefully
    rather than failing to start.
    """
    checkpoint_dir = Path(os.environ.get("ROUTER_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR)))
    onnx_path = checkpoint_dir / "model.onnx"
    if not onnx_path.exists():
        logger.info("no supervised checkpoint at %s, using rule policy", onnx_path)
        return RulePolicy()
    try:
        threshold = float(os.environ.get("ROUTER_SUPERVISED_THRESHOLD", "0.5"))
        policy = SupervisedPolicy(checkpoint_dir, threshold=threshold)
        logger.info("loaded supervised policy from %s (threshold=%.2f)", checkpoint_dir, threshold)
        return policy
    except Exception:
        logger.exception(
            "failed to load supervised policy from %s, using rule policy", checkpoint_dir
        )
        return RulePolicy()


_policy: Policy = _load_policy()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    decision = _policy.decide(req.prompt)
    logger.info("route decision", extra={"tier": decision.tier, "prompt_len": len(req.prompt)})
    return decision


def run() -> None:
    port = int(os.environ.get("ROUTER_PORT", "9000"))
    uvicorn.run("reroute_router.serve.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
