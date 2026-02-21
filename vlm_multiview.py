from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from vlm_llava_utils import parse_hypotheses


def _normalize_label(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if "high" in text or text.startswith("h"):
            return "high"
    if isinstance(value, (int, float)):
        return "high" if float(value) >= 0.5 else "low"
    return "low"


def _pad_waypoints(waypoints: List[Dict[str, Any]], size: int = 4) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for wp in waypoints[:size]:
        out.append(
            {
                "x": float(wp.get("x", 0.0)),
                "y": float(wp.get("y", 0.0)),
                "load": _normalize_label(wp.get("load", "low")),
            }
        )
    while len(out) < size:
        out.append({"x": 0.0, "y": 0.0, "load": "low"})
    return out


def _load_fractions(waypoints: List[Dict[str, Any]]) -> List[float]:
    raw = np.array([2.0 if str(wp.get("load", "low")).lower() == "high" else 1.0 for wp in waypoints], dtype=np.float32)
    raw = np.clip(raw, 1e-6, None)
    frac = raw / float(np.sum(raw))
    return [float(v) for v in frac]


def _extract_hypotheses(pred: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    hyps = parse_hypotheses(pred, k=max(1, int(k)))
    if not hyps:
        return []
    out: List[Dict[str, Any]] = []
    for h in hyps:
        conf = float(h.get("confidence", pred.get("confidence", 0.0)))
        wps = _pad_waypoints(h.get("waypoints", []), size=4)
        out.append({"confidence": conf, "waypoints": wps})
    return out


def fuse_multiview_predictions(view_predictions: Sequence[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
    """
    Fuse per-view hypothesis outputs into one ranked JSON:
      hypotheses[{confidence, waypoints[{x,y,load}], load_fractions[4]}]
    and top-level best hypothesis fields.
    """
    if not view_predictions:
        return {"confidence": 0.0, "waypoints": [], "hypotheses": []}

    per_view_hyps: List[List[Dict[str, Any]]] = []
    view_weights: List[float] = []
    for pred in view_predictions:
        hyps = _extract_hypotheses(pred, k=max(1, int(k)))
        if not hyps:
            continue
        per_view_hyps.append(hyps)
        view_weights.append(max(1e-3, float(hyps[0]["confidence"])))

    if not per_view_hyps:
        return {"confidence": 0.0, "waypoints": [], "hypotheses": []}

    fused_hypotheses: List[Dict[str, Any]] = []
    rank_count = max(1, int(k))
    for rank in range(rank_count):
        rank_waypoints: List[Dict[str, Any]] = []
        rank_conf_num = 0.0
        rank_conf_den = 0.0

        for wp_idx in range(4):
            x_num = 0.0
            y_num = 0.0
            w_den = 0.0
            high_vote = 0.0
            low_vote = 0.0

            for view_idx, hyps in enumerate(per_view_hyps):
                chosen = hyps[rank] if rank < len(hyps) else hyps[-1]
                w = max(1e-3, float(chosen["confidence"])) * view_weights[view_idx]
                wp = chosen["waypoints"][wp_idx]
                x_num += w * float(wp["x"])
                y_num += w * float(wp["y"])
                w_den += w
                if _normalize_label(wp.get("load", "low")) == "high":
                    high_vote += w
                else:
                    low_vote += w

            if w_den <= 1e-9:
                rank_waypoints.append({"x": 0.0, "y": 0.0, "load": "low"})
            else:
                rank_waypoints.append(
                    {
                        "x": float(x_num / w_den),
                        "y": float(y_num / w_den),
                        "load": "high" if high_vote >= low_vote else "low",
                    }
                )

        for view_idx, hyps in enumerate(per_view_hyps):
            chosen = hyps[rank] if rank < len(hyps) else hyps[-1]
            w = view_weights[view_idx]
            rank_conf_num += w * float(chosen["confidence"])
            rank_conf_den += w
        rank_conf = float(rank_conf_num / max(rank_conf_den, 1e-9))
        fused_hypotheses.append(
            {
                "confidence": rank_conf,
                "waypoints": rank_waypoints,
                "load_fractions": _load_fractions(rank_waypoints),
            }
        )

    fused_hypotheses = sorted(fused_hypotheses, key=lambda h: float(h.get("confidence", 0.0)), reverse=True)
    fused_hypotheses = fused_hypotheses[: max(1, int(k))]
    best = fused_hypotheses[0]
    return {
        "confidence": float(best["confidence"]),
        "waypoints": best["waypoints"],
        "hypotheses": fused_hypotheses,
        "fusion": "confidence_weighted_rank",
        "views_used": len(per_view_hyps),
    }


def infer_multiview_cpu(model_path: Path, image_paths: Sequence[Path], k: int = 3) -> Dict[str, Any]:
    from infer_vlm_cpu import infer_image as infer_cpu_image

    preds: List[Dict[str, Any]] = []
    for image_path in image_paths:
        preds.append(infer_cpu_image(Path(model_path), Path(image_path)))
    return fuse_multiview_predictions(preds, k=k)


def infer_multiview_llava(
    model_id: str,
    adapter: str,
    model_path: str,
    image_paths: Sequence[Path],
    prompt: str = (
        "Given the object geometry and dimensions, output K=3 ranked formation hypotheses in object frame. "
        "Return strict JSON: hypotheses[{confidence,waypoints[{x,y,load}],load_fractions[4]}], "
        "plus top-level confidence and waypoints for the best hypothesis."
    ),
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    k: int = 3,
) -> Dict[str, Any]:
    from infer_vlm_llava import infer_with_loaded, load_model_and_processor

    model, processor = load_model_and_processor(model_id=model_id, adapter=adapter, model_path=model_path)
    preds: List[Dict[str, Any]] = []
    for image_path in image_paths:
        preds.append(
            infer_with_loaded(
                model=model,
                processor=processor,
                image_path=Path(image_path),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
    return fuse_multiview_predictions(preds, k=k)
