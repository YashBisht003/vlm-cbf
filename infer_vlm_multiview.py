from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from vlm_multiview import infer_multiview_cpu, infer_multiview_llava


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-view VLM inference and hypothesis fusion")
    parser.add_argument("--backend", choices=("cpu", "llava"), default="llava", help="Inference backend")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths (e.g., 4 robot views)")
    parser.add_argument("--k", type=int, default=3, help="Number of fused ranked hypotheses")
    parser.add_argument("--out", default="vlm_multiview.json", help="Output JSON path")

    # CPU backend
    parser.add_argument("--model", default="cpu_vlm_model.joblib", help="CPU model path")

    # LLaVA backend
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Base LLaVA model id")
    parser.add_argument("--adapter", default="", help="LoRA adapter path")
    parser.add_argument("--model-path", default="", help="Merged model path alternative")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus top-p")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_paths: List[Path] = [Path(p) for p in args.images]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input images: {missing}")

    if args.backend == "cpu":
        output = infer_multiview_cpu(model_path=Path(args.model), image_paths=image_paths, k=args.k)
    else:
        output = infer_multiview_llava(
            model_id=args.model_id,
            adapter=args.adapter,
            model_path=args.model_path,
            image_paths=image_paths,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            k=args.k,
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
