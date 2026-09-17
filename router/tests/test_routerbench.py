import pandas as pd
import pytest

from reroute_router.data.routerbench import load_routerbench


def fake_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "p0",
                "prompt": ["instruction", "actual question"],
                "eval_name": "test_eval",
                "model_a": 1.0,
                "model_a|model_response": "resp a",
                "model_a|total_cost": 0.001,
                "model_b": 0.0,
                "model_b|model_response": "resp b",
                "model_b|total_cost": 0.0001,
                "oracle_model_to_route_to": "model_a",
            },
            {
                "sample_id": "p1",
                "prompt": "plain string prompt",
                "eval_name": "test_eval",
                "model_a": 0.5,
                "model_a|model_response": "resp a2",
                "model_a|total_cost": 0.002,
                "model_b": 0.5,
                "model_b|model_response": "resp b2",
                "model_b|total_cost": 0.0002,
                "oracle_model_to_route_to": "model_a",
            },
        ]
    )


def test_load_routerbench_normalizes_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reroute_router.data.routerbench.pd.read_pickle", lambda _path: fake_raw_df()
    )
    monkeypatch.setattr("reroute_router.data.routerbench._download", lambda variant: "fake/path")

    df = load_routerbench(variant="0shot", models=["model_a", "model_b"])

    assert len(df) == 4  # 2 prompts x 2 models
    assert set(df.columns) == {
        "prompt_id",
        "prompt",
        "model",
        "quality",
        "cost_usd",
        "latency_ms",
        "source",
    }
    assert df["quality"].between(0.0, 1.0).all()
    assert (df["source"] == "routerbench_0shot").all()

    # list-valued prompt is joined; plain string prompt is passed through.
    row0 = df[(df["prompt_id"] == "p0") & (df["model"] == "model_a")].iloc[0]
    assert row0["prompt"] == "instruction\n\nactual question"
    row1 = df[(df["prompt_id"] == "p1") & (df["model"] == "model_a")].iloc[0]
    assert row1["prompt"] == "plain string prompt"

    a0_cost = df[(df["prompt_id"] == "p0") & (df["model"] == "model_a")]["cost_usd"].iloc[0]
    assert a0_cost == pytest.approx(0.001)
