"""Phase 2: LoRA fine-tune of a small encoder (ModernBERT-base by default)
to predict whether the cheap model will be good enough for a given prompt.

This is a stub that defines the training contract; the actual training loop
lands once the dataset from :mod:`reroute_router.data` is available. Kept
here so the module layout and CLI entry point exist from day one.

Usage (once implemented):
    uv run --extra train python -m reroute_router.supervised.train \
        --dataset router/data/processed/train.parquet \
        --base-model answerdotai/ModernBERT-base \
        --output router/supervised/checkpoints/v1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_path: str
    base_model: str = "answerdotai/ModernBERT-base"
    output_dir: str = "router/supervised/checkpoints/v1"
    quality_threshold: float = 0.8
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-4


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", dest="dataset_path", required=True)
    p.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    p.add_argument("--output", dest="output_dir", default="router/supervised/checkpoints/v1")
    p.add_argument("--quality-threshold", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    ns = p.parse_args()
    return TrainConfig(
        dataset_path=ns.dataset_path,
        base_model=ns.base_model,
        output_dir=ns.output_dir,
        quality_threshold=ns.quality_threshold,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        learning_rate=ns.learning_rate,
    )


def main() -> None:
    cfg = parse_args()
    raise NotImplementedError(
        "Phase 2 training loop not implemented yet. "
        f"Would train on {cfg.dataset_path} from base model {cfg.base_model}."
    )


if __name__ == "__main__":
    main()
