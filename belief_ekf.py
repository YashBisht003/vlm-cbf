from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BeliefEKF:
    dt: float
    q_pos: float = 1e-3
    q_vel: float = 1e-2
    r_meas: float = 5e-3

    def __post_init__(self) -> None:
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.05

    def predict(self) -> None:
        dt = self.dt
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_xy: Tuple[float, float]) -> None:
        z = np.array(meas_xy, dtype=np.float32)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * self.r_meas
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P

    def mean(self) -> np.ndarray:
        return self.x.copy()

    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def uncertainty(self) -> float:
        return float(np.sqrt(np.trace(self.P[:2, :2])))

