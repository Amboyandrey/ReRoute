import numpy as np

from reroute_router.bandit.features import Featurizer


def test_featurizer_output_shape() -> None:
    f = Featurizer(n_features=32)
    ctx = f.transform(["hello world", "a different prompt entirely"])
    assert ctx.shape == (2, 33)  # +1 for the bias term


def test_featurizer_bias_term_is_constant_one() -> None:
    f = Featurizer(n_features=16)
    ctx = f.transform(["one prompt", "another one", ""])
    assert np.allclose(ctx[:, -1], 1.0)


def test_featurizer_transform_one_matches_batch() -> None:
    f = Featurizer(n_features=16)
    single = f.transform_one("hello world")
    batch = f.transform(["hello world"])[0]
    assert np.allclose(single, batch)


def test_featurizer_different_prompts_get_different_vectors() -> None:
    f = Featurizer(n_features=64)
    a = f.transform_one("the quick brown fox")
    b = f.transform_one("completely unrelated text about routers")
    assert not np.allclose(a, b)
