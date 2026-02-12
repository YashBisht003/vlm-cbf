from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise ImportError("joblib is required for residual corrector runtime") from exc


FEATURE_NAMES: List[str] = [
    "measured_share",
    "target_share",
    "delta_share",
    "posterior_max",
    "selected_confidence",
    "payload_norm",
    "is_heavy",
    "mass_norm",
    "dim_l_norm",
    "dim_w_norm",
    "dim_h_norm",
    "belief_unc_pos",
    "radial_x",
    "radial_y",
]


@dataclass
class ResidualPredict:
    dx: float
    dy: float


class ResidualCorrectorRuntime:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Residual model not found: {path}")
        payload = joblib.load(path)
        self.model = payload["model"]
        self.scaler = payload.get("scaler", None)
        self.feature_names = payload.get("feature_names", FEATURE_NAMES)

    def predict(self, features: Sequence[float]) -> ResidualPredict:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Residual feature dim mismatch: got {x.shape[1]}, expected {len(self.feature_names)}"
            )
        if self.scaler is not None:
            x = self.scaler.transform(x)
        y = self.model.predict(x)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.shape[0] < 2:
            raise ValueError("Residual model output must have at least 2 values (dx, dy)")
        return ResidualPredict(dx=float(y[0]), dy=float(y[1]))


def synthetic_residual_dataset(
    samples: int,
    seed: int,
    gain: float = 0.12,
    max_shift: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for _ in range(int(samples)):
        measured = float(rng.uniform(0.05, 0.65))
        target = float(rng.uniform(0.05, 0.65))
        delta = measured - target
        posterior_max = float(rng.uniform(0.35, 0.99))
        selected_conf = float(rng.uniform(0.35, 0.99))
        is_heavy = 1.0 if rng.uniform() < 0.5 else 0.0
        payload = 150.0 if is_heavy > 0.5 else 50.0
        payload_norm = payload / 150.0
        mass_norm = float(rng.uniform(50.0, 280.0) / 280.0)
        dim_l = float(rng.uniform(0.4, 1.8) / 2.0)
        dim_w = float(rng.uniform(0.3, 1.4) / 2.0)
        dim_h = float(rng.uniform(0.2, 0.9) / 1.0)
        belief_unc = float(rng.uniform(0.0, 0.25))

        theta = float(rng.uniform(-np.pi, np.pi))
        radial = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        tangential = np.array([-radial[1], radial[0]], dtype=np.float32)

        base = gain * delta
        conf_scale = 0.6 + 0.4 * posterior_max
        unc_scale = 1.0 - 0.4 * np.clip(belief_unc / 0.25, 0.0, 1.0)
        radial_shift = base * conf_scale * unc_scale
        tangent_shift = 0.25 * base * float(rng.uniform(-1.0, 1.0))
        noise = rng.normal(0.0, 0.004, size=2).astype(np.float32)
        shift = radial_shift * radial + tangent_shift * tangential + noise
        shift = np.clip(shift, -max_shift, max_shift).astype(np.float32)

        features = np.array(
            [
                measured,
                target,
                delta,
                posterior_max,
                selected_conf,
                payload_norm,
                is_heavy,
                mass_norm,
                dim_l,
                dim_w,
                dim_h,
                belief_unc,
                float(radial[0]),
                float(radial[1]),
            ],
            dtype=np.float32,
        )
        x_rows.append(features)
        y_rows.append(np.array([float(shift[0]), float(shift[1])], dtype=np.float32))

    return np.stack(x_rows, axis=0), np.stack(y_rows, axis=0)
