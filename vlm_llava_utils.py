from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def strip_image_token(text: str) -> str:
    cleaned = (text or "").replace("<image>", "")
    return cleaned.strip()


def parse_training_record(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Parse one JSONL sample into (prompt, assistant_json_string).
    Supports both:
    - {"conversations":[{"from":"human","value":"<image> ..."}, {"from":"gpt","value":"..."}]}
    - {"prompt":"...", "output": {...}}
    """
    conversations = sample.get("conversations")
    if isinstance(conversations, list) and conversations:
        prompt = ""
        answer = ""
        for item in conversations:
            role = str(item.get("from", "")).strip().lower()
            value = str(item.get("value", ""))
            if role == "human" and not prompt:
                prompt = strip_image_token(value)
            elif role in ("gpt", "assistant") and not answer:
                answer = value
        return prompt, answer

    prompt = str(sample.get("prompt", "")).strip()
    output = sample.get("output", {})
    if isinstance(output, str):
        return prompt, output
    return prompt, json.dumps(output, separators=(",", ":"))


def parse_output_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Try to salvage JSON object from free-form generation.
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None


def normalize_load_label(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if "high" in text or text.startswith("h"):
            return "high"
    if isinstance(value, (int, float)):
        return "high" if float(value) >= 0.5 else "low"
    return "low"


def parse_waypoints(output: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    waypoints = output.get("waypoints", [])
    if not isinstance(waypoints, list):
        return None
    parsed: List[Dict[str, Any]] = []
    for wp in waypoints:
        if not isinstance(wp, dict):
            continue
        try:
            x = float(wp.get("x", 0.0))
            y = float(wp.get("y", 0.0))
        except Exception:
            continue
        load = normalize_load_label(wp.get("load", "low"))
        parsed.append({"x": x, "y": y, "load": load})
    if len(parsed) < 4:
        return None
    return parsed[:4]


def parse_hypotheses(output: Dict[str, Any], k: int = 3) -> Optional[List[Dict[str, Any]]]:
    """
    Parse multi-hypothesis format:
      {"hypotheses":[{"confidence":c,"waypoints":[...]} ...]}
    Falls back to single-hypothesis format if "hypotheses" is missing.
    """
    hyps = output.get("hypotheses", None)
    parsed: List[Dict[str, Any]] = []
    if isinstance(hyps, list) and hyps:
        for entry in hyps[: max(1, int(k))]:
            if not isinstance(entry, dict):
                continue
            wps = parse_waypoints(entry)
            if wps is None:
                continue
            conf = float(entry.get("confidence", output.get("confidence", 0.0)))
            parsed.append({"confidence": conf, "waypoints": wps})
        if parsed:
            parsed.sort(key=lambda h: float(h.get("confidence", 0.0)), reverse=True)
            return parsed

    wps = parse_waypoints(output)
    if wps is None:
        return None
    conf = float(output.get("confidence", 0.0))
    return [{"confidence": conf, "waypoints": wps}]


def extract_ground_truth(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt, answer = parse_training_record(sample)
    _ = prompt  # prompt is not needed for metric extraction
    obj = parse_output_json(answer)
    if obj is None:
        return None
    hypotheses = parse_hypotheses(obj, k=3)
    if hypotheses is None:
        return None
    top = hypotheses[0]
    confidence = float(top.get("confidence", obj.get("confidence", 1.0)))
    return {"confidence": confidence, "waypoints": top["waypoints"], "hypotheses": hypotheses}
