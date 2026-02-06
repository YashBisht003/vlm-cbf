from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import osqp
    from scipy import sparse
except Exception:  # pragma: no cover
    osqp = None
    sparse = None


@dataclass
class CbfResult:
    v_safe: np.ndarray
    success: bool
    slack: float = 0.0


def solve_cbf_qp(
    v_des: np.ndarray,
    pos_i: np.ndarray,
    neighbor_pos: List[np.ndarray],
    neighbor_vel: List[np.ndarray],
    v_max: float,
    d_min: float,
    alpha: float,
    slack_weight: float = 100.0,
    slack_max: float = 3.0,
    extra_linear: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
) -> CbfResult:
    """
    Solve min ||v - v_des||^2 + w*s^2 subject to:
      |vx| <= v_max, |vy| <= v_max
      s >= 0
      2 (p_i - p_j)^T v_i + s >= -alpha * B_ij + 2 (p_i - p_j)^T v_j
      a_k^T v_i + s >= b_k, for each extra linearized CBF constraint
    """
    v_des = np.asarray(v_des, dtype=np.float64).reshape(2)
    pos_i = np.asarray(pos_i, dtype=np.float64).reshape(2)
    slack_weight = max(float(slack_weight), 1e-6)
    slack_max = max(float(slack_max), 1e-6)

    def _clip_speed(v: np.ndarray, vmax: float) -> np.ndarray:
        vn = float(np.linalg.norm(v))
        if vn <= float(vmax) + 1e-9:
            return v
        return v * (float(vmax) / max(vn, 1e-9))

    # Fallback if solver missing.
    if osqp is None or sparse is None:
        v_clip = _clip_speed(np.clip(v_des, -v_max, v_max), v_max)
        return CbfResult(v_safe=v_clip, success=False, slack=0.0)

    # Decision variable x = [vx, vy, s]
    P = sparse.csc_matrix(np.diag([1.0, 1.0, slack_weight]))
    q = np.array([-v_des[0], -v_des[1], 0.0], dtype=np.float64)

    A_rows = []
    l = []
    u = []

    # Box constraints.
    A_rows.append([1.0, 0.0, 0.0]); l.append(-v_max); u.append(v_max)
    A_rows.append([0.0, 1.0, 0.0]); l.append(-v_max); u.append(v_max)
    A_rows.append([0.0, 0.0, 1.0]); l.append(0.0); u.append(slack_max)

    # Separation constraints.
    for pj, vj in zip(neighbor_pos, neighbor_vel):
        pj = np.asarray(pj, dtype=np.float64).reshape(2)
        vj = np.asarray(vj, dtype=np.float64).reshape(2)
        delta = pos_i - pj
        B = float(np.dot(delta, delta) - d_min * d_min)
        rhs = -alpha * B + 2.0 * float(np.dot(delta, vj))
        A_rows.append([2.0 * delta[0], 2.0 * delta[1], 1.0])
        l.append(rhs)
        u.append(np.inf)

    if extra_linear is not None:
        for coeff, rhs in extra_linear:
            coeff = np.asarray(coeff, dtype=np.float64).reshape(2)
            A_rows.append([float(coeff[0]), float(coeff[1]), 1.0])
            l.append(float(rhs))
            u.append(np.inf)

    A = sparse.csc_matrix(np.array(A_rows, dtype=np.float64))
    l = np.array(l, dtype=np.float64)
    u = np.array(u, dtype=np.float64)

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=False, warm_start=True)
    result = solver.solve()
    status = str(result.info.status).lower()
    if "solved" in status and result.x is not None and len(result.x) >= 3:
        v = np.array([result.x[0], result.x[1]], dtype=np.float64)
        v = _clip_speed(v, v_max)
        return CbfResult(v_safe=v, success=True, slack=float(max(0.0, result.x[2])))
    return CbfResult(v_safe=_clip_speed(np.clip(v_des, -v_max, v_max), v_max), success=False, slack=0.0)
