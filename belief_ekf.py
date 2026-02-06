from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BeliefEKF:
    dt: float
    q_pos: float = 1e-3
    q_vel: float = 1e-2
    q_theta: float = 5e-4
    q_omega: float = 2e-3
    r_meas: float = 5e-3
    r_yaw: float = 2e-3

    def __post_init__(self) -> None:
        # State: [x, y, theta, vx, vy, omega]
        self.x = np.zeros(6, dtype=np.float32)
        self.P = np.diag([0.08, 0.08, 0.05, 0.1, 0.1, 0.08]).astype(np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return float(angle)

    def predict(self, accel: Optional[Tuple[float, float, float]] = None) -> None:
        dt = self.dt
        ax, ay, aomega = (0.0, 0.0, 0.0) if accel is None else accel
        x, y, theta, vx, vy, omega = [float(v) for v in self.x]

        x_next = x + vx * dt
        y_next = y + vy * dt
        theta_next = self._wrap_angle(theta + omega * dt)
        vx_next = vx + float(ax) * dt
        vy_next = vy + float(ay) * dt
        omega_next = omega + float(aomega) * dt
        self.x = np.array([x_next, y_next, theta_next, vx_next, vy_next, omega_next], dtype=np.float32)

        F = np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        Q = np.diag(
            [self.q_pos, self.q_pos, self.q_theta, self.q_vel, self.q_vel, self.q_omega]
        ).astype(np.float32)
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_pose: Tuple[float, float, float]) -> None:
        z = np.array(meas_pose, dtype=np.float32)
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        R = np.diag([self.r_meas, self.r_meas, self.r_yaw]).astype(np.float32)
        y = z - H @ self.x
        y[2] = self._wrap_angle(float(y[2]))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(float(self.x[2]))
        self.P = (np.eye(6, dtype=np.float32) - K @ H) @ self.P
        self.P = (0.5 * (self.P + self.P.T)).astype(np.float32)

    def mean(self) -> np.ndarray:
        return self.x.copy()

    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def uncertainty(self) -> float:
        return float(np.sqrt(np.trace(self.P[:2, :2])))

    def uncertainty_full(self) -> float:
        # Position and yaw joint uncertainty.
        return float(np.sqrt(max(0.0, np.trace(self.P[:3, :3]))))

    def risk_components(self) -> Tuple[float, float]:
        pos_u = float(np.sqrt(max(0.0, np.trace(self.P[:2, :2]))))
        yaw_u = float(np.sqrt(max(0.0, self.P[2, 2])))
        return pos_u, yaw_u
