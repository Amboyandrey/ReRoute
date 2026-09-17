"""Context featurizer for the bandit.

Phase 2's ModernBERT only ever got exported to ONNX as a classifier head
(logits out, no exposed pooled embedding), and re-plumbing that export to
also expose hidden states is more machinery than this router needs. LinUCB
also wants a *small*, fixed-dimension context (its per-arm update/score
cost is quadratic in the dimension) — a 768-dim transformer embedding is
the wrong shape for it anyway.

Instead we use a hashed bag-of-words vector (`HashingVectorizer`): fast,
stateless (no vocabulary to fit or ship), fixed-dimension by construction,
and — per the offline replay results in `docs/phase3_bandit.md` — good
enough for LinUCB to learn real structure.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_DIM = 128


def build_vectorizer(n_features: int = DEFAULT_DIM) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,  # non-negative features; simpler for a reward-regression prior
        norm="l2",
        ngram_range=(1, 2),
    )


class Featurizer:
    """Wraps a HashingVectorizer to produce dense context vectors with a
    bias term appended (standard trick so LinUCB can learn a non-zero
    intercept per arm).
    """

    def __init__(self, n_features: int = DEFAULT_DIM):
        self.n_features = n_features
        self.vectorizer = build_vectorizer(n_features)

    @property
    def dim(self) -> int:
        return self.n_features + 1  # +1 for the bias term

    def transform(self, prompts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(prompts)
        dense = sparse.toarray().astype(np.float64)
        bias = np.ones((dense.shape[0], 1))
        return np.hstack([dense, bias])

    def transform_one(self, prompt: str) -> np.ndarray:
        return self.transform([prompt])[0]
