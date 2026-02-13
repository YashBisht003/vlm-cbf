from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover
    raise ImportError("peft is required. Install with: pip install peft") from exc

from vlm_llava_utils import parse_output_json


DEFAULT_PROMPT = (
    "Given the object geometry and dimensions, output K=3 ranked formation hypotheses in object frame. "
    "Return strict JSON: hypotheses[{confidence,waypoints[{x,y,load}],load_fractions[4]}], "
    "plus top-level confidence and waypoints for the best hypothesis."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with LLaVA LoRA model for formation prediction")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Base LLaVA model id")
    parser.add_argument("--adapter", default="", help="LoRA adapter path (if empty, load base model directly)")
    parser.add_argument("--model-path", default="", help="Merged model path alternative to --model-id")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--out", default="llava_formation.json", help="Output JSON file")
    return parser.parse_args()


def load_model_and_processor(model_id: str, adapter: str, model_path: str):
    model_ref = model_path if model_path else model_id
    processor = AutoProcessor.from_pretrained(model_ref)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        base = LlavaForConditionalGeneration.from_pretrained(model_ref, torch_dtype=dtype, device_map="auto")
    else:
        base = LlavaForConditionalGeneration.from_pretrained(model_ref, torch_dtype=dtype)

    if adapter:
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = base
    model.eval()
    return model, processor


def infer_with_loaded(
    model,
    processor,
    image_path: Path,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = generated[:, prompt_len:]
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    parsed = parse_output_json(text_out)
    if parsed is None:
        parsed = {"confidence": 0.0, "waypoints": [], "hypotheses": [], "raw_text": text_out}
    else:
        parsed["raw_text"] = text_out
    return parsed


def infer_image(
    image_path: Path,
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    adapter: str = "",
    model_path: str = "",
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    model, processor = load_model_and_processor(model_id=model_id, adapter=adapter, model_path=model_path)
    return infer_with_loaded(
        model=model,
        processor=processor,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def main() -> None:
    args = _parse_args()
    output = infer_image(
        image_path=Path(args.image),
        model_id=args.model_id,
        adapter=args.adapter,
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
