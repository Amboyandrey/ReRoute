"""Phase 3: contextual bandit router.

Context = a fixed-dimension feature vector for the prompt (see
:mod:`reroute_router.bandit.features`), actions = tier names, reward =
quality - lambda * cost. LinUCB with an offline-replay evaluator against
RouterBench (which has full counterfactual rewards for every action, so
replay doesn't need importance weighting — see
``router/scripts/offline_replay_bandit.py``), then optionally updated
online from live gateway traffic via the router service's ``/feedback``
endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LinUCBArm:
    """One action's (tier's) ridge-regression model: reward ~ theta . context.

    Maintains A^-1 incrementally via the Sherman-Morrison formula instead
    of calling `np.linalg.inv` on every score/update. That matters here:
    with rank-1 updates, Sherman-Morrison is O(d^2) per step versus O(d^3)
    for a fresh inverse, and this arm is scored on every single routing
    decision, so an O(d^3) `score()` would make the bandit the latency
    bottleneck of the request path.
    """

    dim: int
    alpha: float = 1.0
    ridge: float = 1.0
    A_inv: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n_updates: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.A_inv = np.eye(self.dim) / self.ridge
        self.b = np.zeros(self.dim)

    def score(self, context: np.ndarray) -> float:
        theta = self.A_inv @ self.b
        mean = float(theta @ context)
        bonus = self.alpha * float(np.sqrt(max(context @ self.A_inv @ context, 0.0)))
        return mean + bonus

    def update(self, context: np.ndarray, reward: float) -> None:
        # Sherman-Morrison: (A + xx^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        Ai_x = self.A_inv @ context
        denom = 1.0 + float(context @ Ai_x)
        self.A_inv -= np.outer(Ai_x, Ai_x) / denom
        self.b += reward * context
        self.n_updates += 1


class LinUCBRouter:
    """Contextual bandit over a fixed set of tiers using LinUCB."""

    def __init__(
        self,
        tiers: list[str],
        context_dim: int,
        alpha: float = 1.0,
        cost_lambda: float = 0.1,
        ridge: float = 1.0,
    ):
        self.tiers = tiers
        self.cost_lambda = cost_lambda
        self.arms = {tier: LinUCBArm(dim=context_dim, alpha=alpha, ridge=ridge) for tier in tiers}

    def select(self, context: np.ndarray) -> str:
        scores = {tier: arm.score(context) for tier, arm in self.arms.items()}
        return max(scores, key=scores.get)

    def scores(self, context: np.ndarray) -> dict[str, float]:
        return {tier: arm.score(context) for tier, arm in self.arms.items()}

    def reward(self, quality: float, cost_usd: float) -> float:
        return quality - self.cost_lambda * cost_usd

    def update(self, tier: str, context: np.ndarray, quality: float, cost_usd: float) -> None:
        self.arms[tier].update(context, self.reward(quality, cost_usd))

    def save(self, path: str | Path) -> None:
        """Persists learned state (each arm's A_inv, b, update count) plus
        enough config to reconstruct the router, so online learning from
        live traffic survives a router-service restart.
        """
        payload = {
            "tiers": np.array(self.tiers),
            "cost_lambda": np.array(self.cost_lambda),
        }
        for tier, arm in self.arms.items():
            payload[f"{tier}__A_inv"] = arm.A_inv
            payload[f"{tier}__b"] = arm.b
            payload[f"{tier}__alpha"] = np.array(arm.alpha)
            payload[f"{tier}__n_updates"] = np.array(arm.n_updates)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> LinUCBRouter:
        with np.load(path, allow_pickle=False) as data:
            tiers = [str(t) for t in data["tiers"]]
            cost_lambda = float(data["cost_lambda"])
            dim = data[f"{tiers[0]}__b"].shape[0]
            alpha = float(data[f"{tiers[0]}__alpha"])
            router = cls(tiers, context_dim=dim, alpha=alpha, cost_lambda=cost_lambda)
            for tier in tiers:
                arm = router.arms[tier]
                arm.A_inv = data[f"{tier}__A_inv"]
                arm.b = data[f"{tier}__b"]
                arm.alpha = float(data[f"{tier}__alpha"])
                arm.n_updates = int(data[f"{tier}__n_updates"])
        return router
