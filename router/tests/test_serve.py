from fastapi.testclient import TestClient

from reroute_router.serve.main import RulePolicy, app

client = TestClient(app)


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
