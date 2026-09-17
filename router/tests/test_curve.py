import pandas as pd
import pytest

from reroute_router.eval.curve import (
    always_model,
    apply_policy,
    cost_quality_curve,
    headline_metric,
    length_threshold_policy,
    performance_gap_recovered,
)


def make_df() -> pd.DataFrame:
    # Two prompts, two models each: "cheap" and "strong".
    common = {"latency_ms": 50, "source": "test"}
    strong_common = {"latency_ms": 500, "source": "test"}
    return pd.DataFrame(
        [
            {
                "prompt_id": 1,
                "prompt": "hi",
                "model": "cheap",
                "quality": 0.6,
                "cost_usd": 0.001,
                **common,
            },
            {
                "prompt_id": 1,
                "prompt": "hi",
                "model": "strong",
                "quality": 0.95,
                "cost_usd": 0.05,
                **strong_common,
            },
            {
                "prompt_id": 2,
                "prompt": "x" * 500,
                "model": "cheap",
                "quality": 0.3,
                "cost_usd": 0.001,
                **common,
            },
            {
                "prompt_id": 2,
                "prompt": "x" * 500,
                "model": "strong",
                "quality": 0.9,
                "cost_usd": 0.05,
                **strong_common,
            },
        ]
    )


def test_always_model_policy() -> None:
    df = make_df()
    chosen = apply_policy(df, always_model("cheap"))
    assert len(chosen) == 2
    assert (chosen["model"] == "cheap").all()


def test_length_threshold_policy_routes_by_length() -> None:
    df = make_df()
    policy = length_threshold_policy("cheap", "strong", char_threshold=10)
    chosen = apply_policy(df, policy)
    chosen = chosen.set_index("prompt_id")
    assert chosen.loc[1, "model"] == "cheap"
    assert chosen.loc[2, "model"] == "strong"


def test_cost_quality_curve_has_one_row_per_policy() -> None:
    df = make_df()
    policies = {"always_cheap": always_model("cheap"), "always_strong": always_model("strong")}
    curve = cost_quality_curve(df, policies)
    assert set(curve["policy"]) == {"always_cheap", "always_strong"}
    assert (curve["n_prompts"] == 2).all()


def test_performance_gap_recovered_bounds() -> None:
    assert performance_gap_recovered(0.9, 0.5, 0.9) == 1.0
    assert performance_gap_recovered(0.5, 0.5, 0.9) == 0.0
    assert performance_gap_recovered(0.7, 0.5, 0.9) == pytest.approx(0.5)


def test_headline_metric_format() -> None:
    msg = headline_metric(
        router_quality=0.855, router_cost=0.03, strong_quality=0.9, strong_cost=0.10
    )
    assert "%" in msg
    assert "cost" in msg
