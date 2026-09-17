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

ROUTER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR = ROUTER_ROOT / "supervised" / "checkpoints" / "v1"
DEFAULT_BANDIT_STATE_PATH = ROUTER_ROOT / "bandit" / "state" / "v1.npz"
BANDIT_TIERS = ["cheap", "strong"]
BANDIT_CONTEXT_DIM_FEATURES = 128  # see reroute_router.bandit.features.DEFAULT_DIM

app = FastAPI(title="ReRoute router", version="0.1.0")


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    tier: str
    confidence: float


class FeedbackRequest(BaseModel):
    prompt: str
    tier: str
    quality: float
    cost_usd: float = 0.0


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


class BanditPolicy(Policy):
    """Phase 3: LinUCB contextual bandit over hashed bag-of-words prompt
    features (see `reroute_router.bandit`). Unlike the rule and supervised
    policies, this one learns online: `/feedback` calls update it in
    place, and state is persisted to disk after every update so learning
    survives a service restart.
    """

    def __init__(self, state_path: Path):
        from reroute_router.bandit.features import Featurizer
        from reroute_router.bandit.policy import LinUCBRouter

        self.state_path = state_path
        self.featurizer = Featurizer(n_features=BANDIT_CONTEXT_DIM_FEATURES)
        if state_path.exists():
            self.router = LinUCBRouter.load(state_path)
        else:
            cost_lambda = float(os.environ.get("ROUTER_BANDIT_COST_LAMBDA", "100"))
            self.router = LinUCBRouter(
                BANDIT_TIERS, context_dim=self.featurizer.dim, cost_lambda=cost_lambda
            )

    def decide(self, prompt: str) -> RouteResponse:
        ctx = self.featurizer.transform_one(prompt)
        scores = self.router.scores(ctx)
        tier = max(scores, key=scores.get)
        # LinUCB scores aren't probabilities; squash the margin between the
        # chosen and runner-up arm into (0.5, 1) so `confidence` stays
        # comparable in shape to the other policies' outputs.
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = float(1.0 / (1.0 + np.exp(-margin)))
        return RouteResponse(tier=tier, confidence=confidence)

    def learn(self, prompt: str, tier: str, quality: float, cost_usd: float) -> None:
        ctx = self.featurizer.transform_one(prompt)
        self.router.update(tier, ctx, quality, cost_usd)
        self.router.save(self.state_path)


def _load_policy() -> Policy:
    """Picks the best available policy, in order: an explicit
    `ROUTER_POLICY` override, then a persisted bandit state file, then a
    trained supervised checkpoint, then the rule baseline. Any load
    failure falls back to the rule policy rather than refusing to start —
    same fail-open design as the gateway's own router fallback.
    """
    checkpoint_dir = Path(os.environ.get("ROUTER_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR)))
    bandit_state_path = Path(
        os.environ.get("ROUTER_BANDIT_STATE_PATH", str(DEFAULT_BANDIT_STATE_PATH))
    )
    forced = os.environ.get("ROUTER_POLICY", "").strip().lower()

    def load_bandit() -> Policy:
        policy = BanditPolicy(bandit_state_path)
        logger.info("loaded bandit policy (state=%s)", bandit_state_path)
        return policy

    def load_supervised() -> Policy:
        threshold = float(os.environ.get("ROUTER_SUPERVISED_THRESHOLD", "0.5"))
        policy = SupervisedPolicy(checkpoint_dir, threshold=threshold)
        logger.info("loaded supervised policy from %s (threshold=%.2f)", checkpoint_dir, threshold)
        return policy

    try:
        if forced == "bandit":
            return load_bandit()
        if forced == "supervised":
            return load_supervised()
        if forced == "rule":
            return RulePolicy()
        if bandit_state_path.exists():
            return load_bandit()
        if (checkpoint_dir / "model.onnx").exists():
            return load_supervised()
    except Exception:
        logger.exception("failed to load a trained policy, falling back to rule policy")
        return RulePolicy()

    logger.info("no trained policy found, using rule policy")
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


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Online-learning hook: updates the bandit policy from an observed
    outcome (test passed, judge score, thumbs up/down mapped to quality).
    A no-op (reported, not an error) when the loaded policy can't learn
    from feedback — the rule and supervised policies are both fixed once
    loaded. The plan's `POST /v1/feedback` gateway route isn't wired up
    yet (see docs/phase3_bandit.md); this endpoint is what it would call.
    """
    if not isinstance(_policy, BanditPolicy):
        logger.info("feedback received but active policy can't learn from it, ignoring")
        return {"status": "ignored", "reason": "active policy is not a learning policy"}
    _policy.learn(req.prompt, req.tier, req.quality, req.cost_usd)
    return {"status": "ok"}


def run() -> None:
    port = int(os.environ.get("ROUTER_PORT", "9000"))
    uvicorn.run("reroute_router.serve.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
