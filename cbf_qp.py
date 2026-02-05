from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def solve_cbf_qp(
    v_des: np.ndarray,
    pos_i: np.ndarray,
    neighbor_pos: List[np.ndarray],
    neighbor_vel: List[np.ndarray],
    v_max: float,
    d_min: float,
    alpha: float,
) -> CbfResult:
    """
    Solve min ||v - v_des||^2 subject to:
      |vx| <= v_max, |vy| <= v_max
      2 (p_i - p_j)^T v_i >= -alpha * B_ij + 2 (p_i - p_j)^T v_j
    """
    v_des = np.asarray(v_des, dtype=np.float64).reshape(2)
    pos_i = np.asarray(pos_i, dtype=np.float64).reshape(2)

    # Fallback if solver missing.
    if osqp is None or sparse is None:
        v_clip = np.clip(v_des, -v_max, v_max)
        return CbfResult(v_safe=v_clip, success=False)

    P = sparse.csc_matrix(np.eye(2))
    q = -v_des

    A_rows = []
    l = []
    u = []

    # Box constraints.
    A_rows.append([1.0, 0.0]); l.append(-v_max); u.append(v_max)
    A_rows.append([0.0, 1.0]); l.append(-v_max); u.append(v_max)

    # Separation constraints.
    for pj, vj in zip(neighbor_pos, neighbor_vel):
        pj = np.asarray(pj, dtype=np.float64).reshape(2)
        vj = np.asarray(vj, dtype=np.float64).reshape(2)
        delta = pos_i - pj
        B = float(np.dot(delta, delta) - d_min * d_min)
        rhs = -alpha * B + 2.0 * float(np.dot(delta, vj))
        A_rows.append([2.0 * delta[0], 2.0 * delta[1]])
        l.append(rhs)
        u.append(np.inf)

    A = sparse.csc_matrix(np.array(A_rows, dtype=np.float64))
    l = np.array(l, dtype=np.float64)
    u = np.array(u, dtype=np.float64)

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True, warm_start=True)
    result = solver.solve()
    status = str(result.info.status).lower()
    if "solved" in status:
        return CbfResult(v_safe=result.x, success=True)
    return CbfResult(v_safe=np.clip(v_des, -v_max, v_max), success=False)
