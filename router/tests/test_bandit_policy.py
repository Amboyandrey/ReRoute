import numpy as np
import pytest

from reroute_router.bandit.policy import LinUCBArm, LinUCBRouter


def test_arm_incremental_inverse_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    dim = 6
    arm = LinUCBArm(dim=dim, alpha=1.0, ridge=1.0)
    A_brute = np.eye(dim) / 1.0

    for _ in range(25):
        x = rng.normal(size=dim)
        r = rng.normal()
        arm.update(x, r)
        A_brute += np.outer(x, x)

    assert np.allclose(arm.A_inv, np.linalg.inv(A_brute), atol=1e-8)


def test_arm_learns_linear_reward() -> None:
    # theta = [1, 0, 0]; reward = context[0]. The arm should learn to
    # predict higher mean reward for contexts with a larger first
    # coordinate once its uncertainty bonus has shrunk.
    rng = np.random.default_rng(1)
    arm = LinUCBArm(dim=3, alpha=0.0, ridge=1.0)  # alpha=0: pure mean, no exploration bonus
    for _ in range(200):
        x = rng.normal(size=3)
        reward = x[0]
        arm.update(x, reward)

    high = arm.score(np.array([2.0, 0.0, 0.0]))
    low = arm.score(np.array([-2.0, 0.0, 0.0]))
    assert high > low


def test_router_selects_higher_scoring_arm() -> None:
    router = LinUCBRouter(["cheap", "strong"], context_dim=2, alpha=0.0, cost_lambda=0.0)
    ctx = np.array([1.0, 0.0])
    # Force "strong" to have a much higher mean reward for this context.
    router.arms["strong"].update(ctx, reward=1.0)
    router.arms["cheap"].update(ctx, reward=-1.0)
    assert router.select(ctx) == "strong"


def test_router_reward_penalizes_cost() -> None:
    router = LinUCBRouter(["cheap", "strong"], context_dim=2, cost_lambda=10.0)
    assert router.reward(quality=1.0, cost_usd=0.0) == pytest.approx(1.0)
    assert router.reward(quality=1.0, cost_usd=0.05) == pytest.approx(0.5)


def test_router_update_only_touches_chosen_arm() -> None:
    router = LinUCBRouter(["cheap", "strong"], context_dim=2, cost_lambda=0.0)
    ctx = np.array([1.0, 1.0])
    router.update("cheap", ctx, quality=1.0, cost_usd=0.0)
    assert router.arms["cheap"].n_updates == 1
    assert router.arms["strong"].n_updates == 0
