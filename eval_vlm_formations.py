from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from infer_vlm_llava import infer_with_loaded, load_model_and_processor
from vlm_llava_utils import extract_ground_truth, parse_waypoints


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formation prediction accuracy for VLM model")
    parser.add_argument("--jsonl", default="vlm_dataset/val.jsonl", help="Validation JSONL path")
    parser.add_argument("--image-root", default="vlm_dataset", help="Image root for JSONL image refs")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Base model id")
    parser.add_argument("--adapter", default="", help="LoRA adapter path")
    parser.add_argument("--model-path", default="", help="Merged model path alternative")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Fallback threshold tau")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.12,
        help="Mean waypoint distance threshold for formation correctness (m)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--out-csv", default="vlm_eval_samples.csv", help="Per-sample output CSV")
    parser.add_argument("--out-json", default="vlm_eval_metrics.json", help="Aggregate metrics JSON")
    return parser.parse_args()


def _load_jsonl(path: Path, max_rows: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _assignment_error(pred: List[Dict], gt: List[Dict]) -> Tuple[float, float]:
    pred_xy = np.array([[float(w["x"]), float(w["y"])] for w in pred], dtype=np.float64)
    gt_xy = np.array([[float(w["x"]), float(w["y"])] for w in gt], dtype=np.float64)
    cost = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    mean_dist = float(cost[row, col].mean())

    pred_load = [str(pred[r].get("load", "low")).lower() for r in row]
    gt_load = [str(gt[c].get("load", "low")).lower() for c in col]
    load_acc = float(np.mean([1.0 if p == g else 0.0 for p, g in zip(pred_load, gt_load)]))
    return mean_dist, load_acc


def main() -> None:
    args = _parse_args()
    rows = _load_jsonl(Path(args.jsonl), max_rows=args.max_samples)
    image_root = Path(args.image_root)
    if not rows:
        raise RuntimeError("No rows to evaluate.")

    per_sample: List[Dict[str, float | str | int]] = []
    correct_count = 0
    fallback_count = 0
    valid_count = 0
    dist_vals: List[float] = []
    load_vals: List[float] = []

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        adapter=args.adapter,
        model_path=args.model_path,
    )

    for idx, sample in enumerate(rows):
        gt = extract_ground_truth(sample)
        if gt is None:
            continue
        image_rel = sample.get("image", "")
        image_path = image_root / image_rel
        if not image_path.exists():
            continue

        pred_obj = infer_with_loaded(
            model=model,
            processor=processor,
            image_path=image_path,
        )
        pred_waypoints = parse_waypoints(pred_obj if isinstance(pred_obj, dict) else {})
        confidence = float((pred_obj or {}).get("confidence", 0.0)) if isinstance(pred_obj, dict) else 0.0
        fallback = int(confidence < args.confidence_threshold or pred_waypoints is None)
        if fallback:
            fallback_count += 1

        valid = int(pred_waypoints is not None)
        valid_count += valid
        mean_dist = np.nan
        load_acc = np.nan
        correct = 0
        if pred_waypoints is not None:
            mean_dist, load_acc = _assignment_error(pred_waypoints, gt["waypoints"])
            dist_vals.append(float(mean_dist))
            load_vals.append(float(load_acc))
            correct = int((mean_dist <= args.distance_threshold) and (load_acc >= 0.75))
            correct_count += correct

        per_sample.append(
            {
                "idx": idx,
                "image": image_rel,
                "confidence": confidence,
                "fallback": fallback,
                "valid": valid,
                "mean_waypoint_dist": float(mean_dist) if not np.isnan(mean_dist) else "",
                "load_acc": float(load_acc) if not np.isnan(load_acc) else "",
                "correct": correct,
            }
        )

    total = len(per_sample)
    formation_acc = float(correct_count / max(total, 1))
    fallback_rate = float(fallback_count / max(total, 1))
    mean_dist = float(np.mean(dist_vals)) if dist_vals else float("nan")
    mean_load_acc = float(np.mean(load_vals)) if load_vals else float("nan")
    valid_rate = float(valid_count / max(total, 1))

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "idx",
                "image",
                "confidence",
                "fallback",
                "valid",
                "mean_waypoint_dist",
                "load_acc",
                "correct",
            ],
        )
        writer.writeheader()
        for row in per_sample:
            writer.writerow(row)

    metrics = {
        "samples": total,
        "formation_accuracy": formation_acc,
        "fallback_rate": fallback_rate,
        "valid_prediction_rate": valid_rate,
        "mean_waypoint_distance": mean_dist,
        "mean_load_accuracy": mean_load_acc,
        "distance_threshold": args.distance_threshold,
        "confidence_threshold": args.confidence_threshold,
    }
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote sample CSV: {out_csv}")
    print(f"Wrote metrics JSON: {out_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
