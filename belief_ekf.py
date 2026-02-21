from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class BeliefEKF:
    dt: float
    q_pos: float = 1e-3  # reused as mass process noise
    q_vel: float = 1e-2  # reused as COM process noise
    q_theta: float = 5e-4  # reused as inertia process noise
    q_omega: float = 2e-3  # auxiliary regularization noise
    r_meas: float = 5e-3  # per-force measurement noise
    r_yaw: float = 2e-3  # total-force measurement noise

    def __post_init__(self) -> None:
        # State: [mass, com_x, com_y, com_z, Ixx, Iyy, Izz]
        self.x = np.zeros(7, dtype=np.float32)
        self.P = np.diag([225.0, 0.02, 0.02, 0.02, 5.0, 5.0, 5.0]).astype(np.float32)
        self._g = 9.81
        self._state_min = np.array([1.0, -2.0, -2.0, -2.0, 1e-4, 1e-4, 1e-4], dtype=np.float32)
        self._state_max = np.array([500.0, 2.0, 2.0, 2.0, 1e6, 1e6, 1e6], dtype=np.float32)
        self._var_min = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], dtype=np.float32)
        self._var_max = np.array([1e6, 1.0, 1.0, 1.0, 1e8, 1e8, 1e8], dtype=np.float32)

    def initialize(
        self,
        mass_kg: float,
        com_offset_xyz: Sequence[float],
        dims_xyz: Sequence[float],
    ) -> None:
        mass = float(max(1.0, mass_kg))
        dims = np.asarray(dims_xyz, dtype=np.float32).reshape(3)
        com = np.asarray(com_offset_xyz, dtype=np.float32).reshape(3)
        lx, ly, lz = [float(max(1e-3, v)) for v in dims]
        ixx = mass * (ly * ly + lz * lz) / 12.0
        iyy = mass * (lx * lx + lz * lz) / 12.0
        izz = mass * (lx * lx + ly * ly) / 12.0
        self.x = np.array([mass, com[0], com[1], com[2], ixx, iyy, izz], dtype=np.float32)

        mass_sigma = max(5.0, 0.20 * mass)
        com_sigma = max(0.02, 0.20 * max(lx, ly))
        inertia_sigma = max(1.0, 0.15 * max(ixx, iyy, izz))
        self.P = np.diag(
            [
                mass_sigma * mass_sigma,
                com_sigma * com_sigma,
                com_sigma * com_sigma,
                com_sigma * com_sigma,
                inertia_sigma * inertia_sigma,
                inertia_sigma * inertia_sigma,
                inertia_sigma * inertia_sigma,
            ]
        ).astype(np.float32)

    def predict(self) -> None:
        # Random-walk dynamics for latent properties.
        self.x = self.x.astype(np.float32)
        mass_q = max(1e-8, float(self.q_pos))
        com_q = max(1e-8, float(self.q_vel))
        inertia_q = max(1e-8, float(self.q_theta))
        aux_q = max(1e-8, float(self.q_omega))
        Q = np.diag(
            [
                mass_q,
                com_q,
                com_q,
                com_q,
                inertia_q + aux_q,
                inertia_q + aux_q,
                inertia_q + aux_q,
            ]
        ).astype(np.float32)
        self.P = (self.P + Q).astype(np.float32)
        self._stabilize()

    def _measurement_model(
        self,
        state: np.ndarray,
        robot_positions_xy: np.ndarray,
        object_center_xy: np.ndarray,
    ) -> np.ndarray:
        mass = float(max(1.0, state[0]))
        com_xy = np.asarray(state[1:3], dtype=np.float32)
        inertia_vec = np.asarray(state[4:7], dtype=np.float32)

        rel = np.asarray(robot_positions_xy, dtype=np.float32) - np.asarray(object_center_xy, dtype=np.float32).reshape(1, 2)
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        dirs = rel / np.clip(dist, 1e-4, None)
        r_scale = float(max(0.1, np.mean(np.linalg.norm(rel, axis=1))))
        beta = 1.5
        score = 1.0 + beta * (dirs @ com_xy.reshape(2, 1)).reshape(-1) / max(1e-4, r_scale)
        score = np.clip(score, 0.05, 4.0)
        share = score / max(1e-6, float(np.sum(score)))
        inertia_scale = float(np.clip(np.mean(inertia_vec) / max(1.0, np.mean(np.abs(inertia_vec))), -2.0, 2.0))
        inertia_gain = 1.0 + 0.02 * inertia_scale
        force_pred = mass * self._g * inertia_gain * share
        total_pred = mass * self._g
        return np.concatenate([force_pred.astype(np.float32), np.array([total_pred], dtype=np.float32)], axis=0)

    def _jacobian_numeric(
        self,
        state: np.ndarray,
        robot_positions_xy: np.ndarray,
        object_center_xy: np.ndarray,
    ) -> np.ndarray:
        z0 = self._measurement_model(state, robot_positions_xy, object_center_xy)
        m = z0.shape[0]
        n = state.shape[0]
        H = np.zeros((m, n), dtype=np.float32)
        for i in range(n):
            dx = np.zeros_like(state, dtype=np.float32)
            eps = 1e-3 * max(1.0, abs(float(state[i])))
            dx[i] = eps
            z1 = self._measurement_model(state + dx, robot_positions_xy, object_center_xy)
            z2 = self._measurement_model(state - dx, robot_positions_xy, object_center_xy)
            H[:, i] = ((z1 - z2) / max(1e-6, 2.0 * eps)).astype(np.float32)
        return H

    def _stabilize(self) -> None:
        # Keep state and covariance finite and bounded.
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        self.x = np.clip(self.x, self._state_min, self._state_max).astype(np.float32)
        self.x[0] = max(1.0, float(self.x[0]))

        P = np.nan_to_num(self.P, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)
        P = (0.5 * (P + P.T)).astype(np.float32)
        d = np.diag(P).astype(np.float32)
        d = np.clip(d, self._var_min, self._var_max)
        np.fill_diagonal(P, d)
        # Project to PSD to avoid negative variances from numerical drift.
        try:
            w, v = np.linalg.eigh(P.astype(np.float64))
            w = np.clip(w, 1e-9, 1e12)
            P = (v @ np.diag(w) @ v.T).astype(np.float32)
        except np.linalg.LinAlgError:
            P = np.diag(d).astype(np.float32)
        self.P = (0.5 * (P + P.T)).astype(np.float32)

    def update(
        self,
        forces_z: Sequence[float] | Tuple[float, float, float],
        robot_positions_xy: Optional[Sequence[Sequence[float]]] = None,
        object_center_xy: Optional[Sequence[float]] = None,
    ) -> None:
        # Backward-compatible no-op for older pose-only call sites.
        if robot_positions_xy is None or object_center_xy is None:
            return

        z_forces = np.asarray(forces_z, dtype=np.float32).reshape(-1)
        if z_forces.size == 0:
            return
        z_forces = np.clip(z_forces, 0.0, None)
        z_total = float(np.sum(z_forces))
        z = np.concatenate([z_forces, np.array([z_total], dtype=np.float32)], axis=0)

        robot_xy = np.asarray(robot_positions_xy, dtype=np.float32).reshape(-1, 2)
        obj_xy = np.asarray(object_center_xy, dtype=np.float32).reshape(2)
        if robot_xy.shape[0] != z_forces.shape[0]:
            return

        self._stabilize()
        h = self._measurement_model(self.x, robot_xy, obj_xy)
        H = self._jacobian_numeric(self.x, robot_xy, obj_xy)

        force_var = max(1e-6, float(self.r_meas))
        total_var = max(1e-6, float(self.r_yaw))
        R = np.diag([force_var] * z_forces.shape[0] + [total_var]).astype(np.float32)

        y = (z - h).astype(np.float32)
        S = H @ self.P @ H.T + R
        if not np.all(np.isfinite(S)):
            self._stabilize()
            return
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        if not np.all(np.isfinite(K)):
            self._stabilize()
            return

        self.x = (self.x + K @ y).astype(np.float32)
        I = np.eye(7, dtype=np.float32)
        KH = K @ H
        # Joseph form is numerically more stable than (I-KH)P.
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
        self._stabilize()

    def mean(self) -> np.ndarray:
        return self.x.copy()

    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def uncertainty(self) -> float:
        return float(np.sqrt(max(0.0, np.trace(self.P[1:3, 1:3]))))

    def uncertainty_full(self) -> float:
        # COM and mass joint uncertainty.
        return float(np.sqrt(max(0.0, np.trace(self.P[:4, :4]))))

    def risk_components(self) -> Tuple[float, float]:
        com_u = float(np.sqrt(max(0.0, np.trace(self.P[1:3, 1:3]))))
        mass_rel = float(np.sqrt(max(0.0, self.P[0, 0])) / max(1e-3, float(self.x[0])))
        return com_u, mass_rel
