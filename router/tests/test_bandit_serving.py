import pytest
from fastapi.testclient import TestClient

import reroute_router.serve.main as main
from reroute_router.serve.main import BanditPolicy, RulePolicy, app

client = TestClient(app)


def test_bandit_policy_cold_start_decides_without_crashing(tmp_path) -> None:
    policy = BanditPolicy(state_path=tmp_path / "bandit.npz")
    decision = policy.decide("some prompt")
    assert decision.tier in ("cheap", "strong")
    assert 0.0 <= decision.confidence <= 1.0


def test_bandit_policy_learn_persists_state(tmp_path) -> None:
    state_path = tmp_path / "bandit.npz"
    policy = BanditPolicy(state_path=state_path)
    policy.learn("a prompt", tier="cheap", quality=0.9, cost_usd=0.0001)
    assert state_path.exists()

    reloaded = BanditPolicy(state_path=state_path)
    assert reloaded.router.arms["cheap"].n_updates == 1


def test_bandit_policy_learns_to_prefer_rewarding_tier(tmp_path) -> None:
    policy = BanditPolicy(state_path=tmp_path / "bandit.npz")
    prompt = "a specific repeated prompt"
    for _ in range(50):
        policy.learn(prompt, tier="cheap", quality=1.0, cost_usd=0.0)
        policy.learn(prompt, tier="strong", quality=0.0, cost_usd=0.0)
    decision = policy.decide(prompt)
    assert decision.tier == "cheap"


def test_feedback_endpoint_ignored_when_policy_cannot_learn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_policy", RulePolicy())
    resp = client.post(
        "/feedback",
        json={"prompt": "hi", "tier": "cheap", "quality": 1.0, "cost_usd": 0.0},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"


def test_feedback_endpoint_updates_bandit_policy(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = BanditPolicy(state_path=tmp_path / "bandit.npz")
    monkeypatch.setattr(main, "_policy", policy)

    resp = client.post(
        "/feedback",
        json={"prompt": "hi", "tier": "cheap", "quality": 1.0, "cost_usd": 0.0},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert policy.router.arms["cheap"].n_updates == 1
