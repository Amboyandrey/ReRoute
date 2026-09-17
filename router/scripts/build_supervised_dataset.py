#!/usr/bin/env python3
"""Phase 2 step 1: turn RouterBench into a binary classification dataset
for the supervised router — "will the weak (cheap-tier proxy) model be
good enough for this prompt?" — split into train/val/test by prompt_id.

Uses the same weak/strong pair as Phase 1's baseline curve
(docs/phase1_baseline_curve.md) so the two are directly comparable.

Usage:
    uv run --extra train python scripts/build_supervised_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTERBENCH_PATH = REPO_ROOT / "router" / "data" / "processed" / "routerbench_0shot.parquet"
OUT_PATH = REPO_ROOT / "router" / "data" / "processed" / "supervised_dataset.parquet"

WEAK_MODEL = "mistralai/mistral-7b-chat"
STRONG_MODEL = "gpt-4-1106-preview"
QUALITY_THRESHOLD = 0.5  # matches Phase 1's binary-ish quality scale (0, .25, .5, .75, 1)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 0


def main() -> None:
    if not ROUTERBENCH_PATH.exists():
        raise SystemExit(
            f"{ROUTERBENCH_PATH} not found — run "
            "`uv run python scripts/build_routerbench_curve.py` first."
        )
    df = pd.read_parquet(ROUTERBENCH_PATH)

    weak = df[df["model"] == WEAK_MODEL][["prompt_id", "prompt", "quality", "cost_usd"]].rename(
        columns={"quality": "weak_quality", "cost_usd": "weak_cost_usd"}
    )
    strong = df[df["model"] == STRONG_MODEL][["prompt_id", "quality", "cost_usd"]].rename(
        columns={"quality": "strong_quality", "cost_usd": "strong_cost_usd"}
    )
    merged = weak.merge(strong, on="prompt_id", how="inner")
    merged["label"] = (merged["weak_quality"] >= QUALITY_THRESHOLD).astype(int)

    print(f"{len(merged):,} prompts with both weak and strong scores")
    print(f"label balance: {merged['label'].value_counts(normalize=True).to_dict()}")

    # merged is one row per prompt (weak/strong joined), so a random shuffle
    # followed by a contiguous train/val/test split has no leakage risk.
    shuffled = merged.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    shuffled["split"] = split

    for name in SPLITS:
        part = shuffled[shuffled["split"] == name]
        print(f"  {name}: {len(part):,} rows, label==1 rate={part['label'].mean():.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
