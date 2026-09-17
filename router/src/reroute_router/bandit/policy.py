"""Phase 3: contextual bandit router.

Context = prompt embedding (reused from the Phase 2 encoder), actions =
tier names, reward = quality - lambda * cost. Starts as LinUCB with an
offline-replay evaluator against RouterBench (which has full counterfactual
rewards for every action), then runs online against live gateway traffic.

This module defines the interface; the learning algorithm lands once the
supervised embedding model (Phase 2) exists to provide contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinUCBArm:
    """One action's (tier's) linear model: reward ~ theta . context."""

    dim: int
    alpha: float = 1.0
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.A = np.eye(self.dim)
        self.b = np.zeros(self.dim)

    def score(self, context: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = float(theta @ context)
        bonus = self.alpha * float(np.sqrt(context @ A_inv @ context))
        return mean + bonus

    def update(self, context: np.ndarray, reward: float) -> None:
        self.A += np.outer(context, context)
        self.b += reward * context


class LinUCBRouter:
    """Contextual bandit over a fixed set of tiers using LinUCB."""

    def __init__(
        self,
        tiers: list[str],
        context_dim: int,
        alpha: float = 1.0,
        cost_lambda: float = 0.1,
    ):
        self.tiers = tiers
        self.cost_lambda = cost_lambda
        self.arms = {tier: LinUCBArm(dim=context_dim, alpha=alpha) for tier in tiers}

    def select(self, context: np.ndarray) -> str:
        scores = {tier: arm.score(context) for tier, arm in self.arms.items()}
        return max(scores, key=scores.get)

    def reward(self, quality: float, cost_usd: float) -> float:
        return quality - self.cost_lambda * cost_usd

    def update(self, tier: str, context: np.ndarray, quality: float, cost_usd: float) -> None:
        self.arms[tier].update(context, self.reward(quality, cost_usd))
