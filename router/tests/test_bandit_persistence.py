import numpy as np

from reroute_router.bandit.policy import LinUCBRouter


def test_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    router = LinUCBRouter(["cheap", "strong"], context_dim=5, alpha=0.7, cost_lambda=42.0)
    for _ in range(10):
        ctx = rng.normal(size=5)
        router.update("cheap", ctx, quality=0.8, cost_usd=0.001)
        router.update("strong", ctx, quality=0.9, cost_usd=0.01)

    path = tmp_path / "bandit.npz"
    router.save(path)
    loaded = LinUCBRouter.load(path)

    assert loaded.tiers == router.tiers
    assert loaded.cost_lambda == router.cost_lambda
    for tier in router.tiers:
        assert np.allclose(loaded.arms[tier].A_inv, router.arms[tier].A_inv)
        assert np.allclose(loaded.arms[tier].b, router.arms[tier].b)
        assert loaded.arms[tier].n_updates == router.arms[tier].n_updates
        assert loaded.arms[tier].alpha == router.arms[tier].alpha


def test_loaded_router_makes_same_decisions_as_original(tmp_path) -> None:
    rng = np.random.default_rng(1)
    router = LinUCBRouter(["cheap", "strong"], context_dim=4, cost_lambda=5.0)
    for _ in range(15):
        ctx = rng.normal(size=4)
        tier = "cheap" if rng.random() < 0.5 else "strong"
        router.update(tier, ctx, quality=rng.random(), cost_usd=rng.random() * 0.01)

    path = tmp_path / "bandit.npz"
    router.save(path)
    loaded = LinUCBRouter.load(path)

    for _ in range(20):
        ctx = rng.normal(size=4)
        assert loaded.select(ctx) == router.select(ctx)
