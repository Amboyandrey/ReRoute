import pytest
from fastapi.testclient import TestClient

import reroute_router.serve.main as main
from reroute_router.serve.main import RulePolicy, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def force_rule_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    # The /route endpoint tests below exercise the *rule* baseline
    # specifically. `_load_policy()` swaps in the trained supervised
    # policy automatically whenever a checkpoint exists on disk (e.g. after
    # running Phase 2 training locally), which would make these tests
    # depend on model quality instead of the rule it's meant to test.
    monkeypatch.setattr(main, "_policy", RulePolicy())


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_route_short_prompt_goes_cheap() -> None:
    resp = client.post("/route", json={"prompt": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "cheap"


def test_route_long_prompt_goes_strong() -> None:
    resp = client.post("/route", json={"prompt": "x" * 2000})
    assert resp.status_code == 200
    assert resp.json()["tier"] == "strong"


def test_rule_policy_thresholds() -> None:
    policy = RulePolicy(short_chars=10, long_chars=20)
    assert policy.decide("short").tier == "cheap"
    assert policy.decide("x" * 15).tier == "mid"
    assert policy.decide("x" * 25).tier == "strong"
