"""Online routing service.

The Go gateway calls ``POST /route`` with a prompt and gets back the tier
name that should handle it. This module starts with a length-based rule
baseline so the gateway has something real to call from day one; Phase 2
swaps in the trained ModernBERT classifier and Phase 3 swaps that for the
contextual bandit, both behind the same HTTP contract so the gateway never
has to change.
"""

from __future__ import annotations

import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("reroute.router")

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


_policy: Policy = RulePolicy()


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
