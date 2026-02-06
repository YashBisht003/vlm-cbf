from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise ImportError("peft is required. Install with: pip install peft") from exc

from vlm_llava_utils import parse_training_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA with LoRA for formation prediction")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Base LLaVA model id")
    parser.add_argument("--train-jsonl", default="vlm_dataset/train.jsonl", help="Train JSONL path")
    parser.add_argument("--val-jsonl", default="vlm_dataset/val.jsonl", help="Validation JSONL path")
    parser.add_argument("--image-root", default="vlm_dataset", help="Root path for image references in JSONL")
    parser.add_argument("--out", default="llava_lora_out", help="Output folder for adapter and logs")
    parser.add_argument("--epochs", type=float, default=3.0, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max-length", type=int, default=1024, help="Token truncation length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (paper: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (paper: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--bf16", action="store_true", help="Use bf16")
    parser.add_argument("--use-4bit", action="store_true", help="Enable 4-bit QLoRA load")
    parser.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable grad checkpointing")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint save cadence")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval cadence")
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging cadence")
    parser.set_defaults(use_4bit=True)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class LlavaFormationDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], image_root: Path) -> None:
        self.rows = rows
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.rows[idx]
        image_rel = sample.get("image", "")
        image_path = self.image_root / image_rel
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        prompt, answer = parse_training_record(sample)
        if not prompt:
            prompt = (
                "Given the object geometry and dimensions, output 4 robot waypoints in the object frame. "
                "Return JSON with fields: waypoints[{x,y,load}], confidence."
            )
        return {"image": image, "prompt": prompt, "answer": answer}


class LlavaSupervisedCollator:
    def __init__(self, processor: AutoProcessor, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _messages_user(self, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _messages_full(self, prompt: str, answer: str) -> List[Dict[str, Any]]:
        return self._messages_user(prompt) + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]

        full_texts = [
            self.processor.apply_chat_template(
                self._messages_full(prompt, answer), tokenize=False, add_generation_prompt=False
            )
            for prompt, answer in zip(prompts, answers)
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._messages_user(prompt), tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for i, plen in enumerate(prompt_lens):
            labels[i, : int(plen)] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels
        return model_inputs


def _build_model(args: argparse.Namespace, use_cuda: bool):
    torch_dtype = torch.float16 if use_cuda else torch.float32
    load_kwargs = {"torch_dtype": torch_dtype}
    use_4bit = bool(args.use_4bit and use_cuda)
    if args.use_4bit and not use_cuda:
        print("Warning: --use-4bit requested but CUDA is unavailable. Falling back to non-quantized load.")
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
        )
        load_kwargs["quantization_config"] = bnb
        load_kwargs["device_map"] = "auto"
    elif use_cuda:
        load_kwargs["device_map"] = "auto"

    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    image_root = Path(args.image_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path)
    if not train_rows:
        raise RuntimeError("No training samples loaded.")
    if not val_rows:
        raise RuntimeError("No validation samples loaded.")

    processor = AutoProcessor.from_pretrained(args.model_id)
    use_cuda = torch.cuda.is_available()
    model = _build_model(args, use_cuda=use_cuda)

    train_ds = LlavaFormationDataset(train_rows, image_root=image_root)
    val_ds = LlavaFormationDataset(val_rows, image_root=image_root)
    collator = LlavaSupervisedCollator(processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        remove_unused_columns=False,
        dataloader_num_workers=args.workers,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    processor.save_pretrained(str(out_dir / "adapter"))
    metrics = trainer.evaluate()
    metrics_path = out_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved adapter to: {out_dir / 'adapter'}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
