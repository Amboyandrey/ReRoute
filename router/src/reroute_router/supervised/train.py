"""Phase 2: LoRA fine-tune of a small encoder (ModernBERT-base by default)
to predict whether the cheap (weak) model will be good enough for a given
prompt — a binary classifier over the same weak/strong pair used for
Phase 1's baseline curve, so the two are directly comparable.

Usage:
    uv run --extra train python -m reroute_router.supervised.train \
        --dataset router/data/processed/supervised_dataset.parquet \
        --output router/supervised/checkpoints/v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrainConfig:
    dataset_path: str
    base_model: str = "answerdotai/ModernBERT-base"
    output_dir: str = "router/supervised/checkpoints/v1"
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    # False by default: verified to fit in 8GB VRAM at batch=8/max_length=512
    # without it, and it's ~35% faster than with checkpointing on.
    gradient_checkpointing: bool = False
    learning_rate: float = 2e-4
    seed: int = 0


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", dest="dataset_path", required=True)
    p.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    p.add_argument("--output", dest="output_dir", default="router/supervised/checkpoints/v1")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--gradient-checkpointing", action="store_true")
    ns = p.parse_args()
    return TrainConfig(
        dataset_path=ns.dataset_path,
        base_model=ns.base_model,
        output_dir=ns.output_dir,
        max_length=ns.max_length,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        grad_accum_steps=ns.grad_accum_steps,
        gradient_checkpointing=ns.gradient_checkpointing,
        learning_rate=ns.learning_rate,
    )


def load_splits(dataset_path: str) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(dataset_path)
    return {
        name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")
    }


def build_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["prompt", "label"]])

    def tokenize(batch):
        return tokenizer(batch["prompt"], truncation=True, max_length=max_length, padding=False)

    return ds.map(tokenize, batched=True, remove_columns=["prompt"])


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    logits, labels = eval_pred
    probs = _softmax(logits)[:, 1]
    preds = probs >= 0.5
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def export_onnx(model, tokenizer, output_dir: Path, max_length: int) -> None:
    """Merges the LoRA adapter into the base model and exports it to ONNX
    so the serving path (`reroute_router.serve.main`) only needs
    `onnxruntime` + `tokenizers`, not the full torch/transformers/peft
    training stack.
    """
    import torch

    merged = model.merge_and_unload()
    merged.eval().to("cpu")

    dummy = tokenizer(
        "warmup example for onnx export",
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        merged,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        # torch's exporter targets opset 18 regardless of what's requested
        # here for this model; asking for 17 triggers a lossy fallback
        # conversion that produces an invalid graph (a Split node using an
        # opset-18-only attribute) onnxruntime then refuses to load.
        opset_version=18,
    )
    print(f"Exported ONNX model to {onnx_path}")


def train(cfg: TrainConfig) -> Path:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    splits = load_splits(cfg.dataset_path)
    print(f"train/val/test sizes: {[len(splits[s]) for s in ('train', 'val', 'test')]}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=2,
        id2label={0: "route_strong", 1: "route_cheap"},
        label2id={"route_strong": 0, "route_cheap": 1},
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        # ModernBERT's sequence-classification head ("classifier") is
        # randomly initialized (not part of the pretrained checkpoint), so
        # it needs full fine-tuning rather than a low-rank adapter.
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if cfg.gradient_checkpointing:
        # Required for gradient checkpointing to work with frozen
        # base-model params under LoRA (otherwise the checkpointed
        # segments have no tensor with requires_grad=True to build a
        # backward graph from).
        model.enable_input_require_grads()

    train_ds = build_hf_dataset(splits["train"], tokenizer, cfg.max_length)
    val_ds = build_hf_dataset(splits["val"], tokenizer, cfg.max_length)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_metrics = trainer.evaluate()
    print("Final val metrics:", val_metrics)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_config.json").write_text(
        json.dumps({**cfg.__dict__, "val_metrics": val_metrics}, indent=2, default=str)
    )
    print(f"Saved adapter + tokenizer to {output_dir}")

    try:
        export_onnx(model, tokenizer, output_dir, cfg.max_length)
    except Exception as e:  # noqa: BLE001
        # ONNX export is a nice-to-have for the lean serving path; a
        # failure here shouldn't discard an otherwise-successful training
        # run (the router service falls back to the rule policy if
        # model.onnx isn't present).
        print(f"ONNX export failed (non-fatal): {e}")

    return output_dir


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
