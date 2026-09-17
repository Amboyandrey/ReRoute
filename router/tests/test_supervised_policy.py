from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reroute_router.serve.main import RulePolicy, SupervisedPolicy, _load_policy


class FakeEncoding:
    def __init__(self, ids, attention_mask):
        self.ids = ids
        self.attention_mask = attention_mask


def make_policy_with_mocks(
    logits_for_cheap: float, logits_for_strong: float, threshold: float = 0.5
):
    """Builds a SupervisedPolicy with onnxruntime/tokenizers mocked out, so
    the test exercises only our decision logic (softmax + threshold), not
    a real model.
    """
    with (
        patch("onnxruntime.InferenceSession") as mock_session_cls,
        patch("tokenizers.Tokenizer.from_file") as mock_tok_from_file,
    ):
        mock_session = MagicMock()
        mock_session.run.return_value = (np.array([[logits_for_strong, logits_for_cheap]]),)
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = FakeEncoding(ids=[1, 2, 3], attention_mask=[1, 1, 1])
        mock_tok_from_file.return_value = mock_tokenizer

        policy = SupervisedPolicy(checkpoint_dir=Path("/fake/checkpoint"), threshold=threshold)
    return policy


def test_supervised_policy_routes_cheap_when_confident() -> None:
    # Large positive logit for "cheap" relative to "strong" -> high P(cheap).
    policy = make_policy_with_mocks(logits_for_cheap=5.0, logits_for_strong=-5.0)
    decision = policy.decide("an easy prompt")
    assert decision.tier == "cheap"
    assert decision.confidence > 0.9


def test_supervised_policy_routes_strong_when_confident() -> None:
    policy = make_policy_with_mocks(logits_for_cheap=-5.0, logits_for_strong=5.0)
    decision = policy.decide("a hard prompt")
    assert decision.tier == "strong"
    assert decision.confidence > 0.9


def test_supervised_policy_respects_custom_threshold() -> None:
    # Roughly 50/50 logits but a low threshold should still tip to "cheap".
    policy = make_policy_with_mocks(logits_for_cheap=0.1, logits_for_strong=0.0, threshold=0.2)
    decision = policy.decide("borderline prompt")
    assert decision.tier == "cheap"


def test_load_policy_falls_back_to_rule_when_no_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROUTER_CHECKPOINT_DIR", str(tmp_path / "does-not-exist"))
    policy = _load_policy()
    assert isinstance(policy, RulePolicy)
