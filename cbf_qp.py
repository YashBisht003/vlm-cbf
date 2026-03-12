from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    slack_exceeded: bool = False
    recovery_active: bool = False


_OSQP_CACHE: Dict[tuple[int, int], object] = {}


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
    slack_threshold: float = np.inf,
    recovery_gain: float = 0.5,
) -> CbfResult:
    """
    Solve min ||v - v_des||^2 + w*s^2 subject to:
      |vx| <= v_max, |vy| <= v_max
      s >= 0
      2 (p_i - p_j)^T v_i >= -alpha * B_ij + 2 (p_i - p_j)^T v_j
      a_k^T v_i + s >= b_k, for each extra linearized non-separation constraint
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

    def _project_halfspace(v: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
        denom = float(np.dot(a, a))
        if denom <= 1.0e-12:
            return v
        violation = float(b - np.dot(a, v))
        if violation <= 0.0:
            return v
        return v + (violation / denom) * a

    def _analytic_safe_fallback(v: np.ndarray) -> np.ndarray:
        v_out = _clip_speed(np.clip(v, -v_max, v_max), v_max)
        constraints: List[tuple[np.ndarray, float]] = []
        for pj, vj in zip(neighbor_pos, neighbor_vel):
            pj = np.asarray(pj, dtype=np.float64).reshape(2)
            vj = np.asarray(vj, dtype=np.float64).reshape(2)
            delta = pos_i - pj
            B = float(np.dot(delta, delta) - d_min * d_min)
            rhs = -alpha * B + 2.0 * float(np.dot(delta, vj))
            a = np.array([2.0 * delta[0], 2.0 * delta[1]], dtype=np.float64)
            constraints.append((a, rhs))
        # Cyclic projection over separation half-spaces to handle multiple violated constraints.
        for _ in range(20):
            changed = False
            for a, rhs in constraints:
                v_next = _project_halfspace(v_out, a, rhs)
                if float(np.linalg.norm(v_next - v_out)) > 1.0e-10:
                    changed = True
                v_out = v_next
            v_out = _clip_speed(np.clip(v_out, -v_max, v_max), v_max)
            if not changed:
                break
        return _clip_speed(np.clip(v_out, -v_max, v_max), v_max)

    def _dense_csc(mat: np.ndarray):
        rows, cols = mat.shape
        data = np.asarray(mat, dtype=np.float64).reshape(-1, order="F")
        indices = np.tile(np.arange(rows, dtype=np.int32), cols)
        indptr = np.arange(0, rows * cols + 1, rows, dtype=np.int32)
        return sparse.csc_matrix((data, indices, indptr), shape=(rows, cols))

    recovery_bias = np.zeros(2, dtype=np.float64)
    recovery_active = False
    for pj in neighbor_pos:
        pj = np.asarray(pj, dtype=np.float64).reshape(2)
        delta = pos_i - pj
        dist = float(np.linalg.norm(delta))
        B = float(np.dot(delta, delta) - d_min * d_min)
        if B < 0.0 and dist > 1.0e-9 and recovery_gain > 0.0:
            recovery_active = True
            penetration = float(d_min - dist)
            recovery_bias += (recovery_gain * penetration / dist) * delta
    v_ref = v_des + recovery_bias

    # Fallback if solver missing.
    if osqp is None or sparse is None:
        v_clip = _analytic_safe_fallback(v_ref)
        return CbfResult(v_safe=v_clip, success=False, slack=0.0, recovery_active=recovery_active)

    # Decision variable x = [vx, vy, s]
    P = sparse.csc_matrix(np.diag([1.0, 1.0, slack_weight]))
    q = np.array([-v_ref[0], -v_ref[1], 0.0], dtype=np.float64)

    A_rows = []
    l = []
    u = []

    # Box constraints.
    A_rows.append([1.0, 0.0, 0.0]); l.append(-v_max); u.append(v_max)
    A_rows.append([0.0, 1.0, 0.0]); l.append(-v_max); u.append(v_max)
    A_rows.append([0.0, 0.0, 1.0]); l.append(0.0); u.append(slack_max)

    # Separation constraints are hard constraints: no slack term.
    for pj, vj in zip(neighbor_pos, neighbor_vel):
        pj = np.asarray(pj, dtype=np.float64).reshape(2)
        vj = np.asarray(vj, dtype=np.float64).reshape(2)
        delta = pos_i - pj
        B = float(np.dot(delta, delta) - d_min * d_min)
        rhs = -alpha * B + 2.0 * float(np.dot(delta, vj))
        A_rows.append([2.0 * delta[0], 2.0 * delta[1], 0.0])
        l.append(rhs)
        u.append(np.inf)

    if extra_linear is not None:
        for coeff, rhs in extra_linear:
            coeff = np.asarray(coeff, dtype=np.float64).reshape(2)
            A_rows.append([float(coeff[0]), float(coeff[1]), 1.0])
            l.append(float(rhs))
            u.append(np.inf)

    A_dense = np.array(A_rows, dtype=np.float64)
    A = _dense_csc(A_dense)
    l = np.array(l, dtype=np.float64)
    u = np.array(u, dtype=np.float64)

    cache_key = (A.shape[0], A.shape[1])
    solver = _OSQP_CACHE.get(cache_key)
    if solver is None:
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=False, warm_start=True)
        _OSQP_CACHE[cache_key] = solver
    else:
        solver.update(q=q, l=l, u=u)
        solver.update(Px=P.data, Ax=A.data)
    result = solver.solve()
    status = str(result.info.status).lower()
    if "solved" in status and result.x is not None and len(result.x) >= 3:
        v = np.array([result.x[0], result.x[1]], dtype=np.float64)
        v = _clip_speed(v, v_max)
        slack = float(max(0.0, result.x[2]))
        return CbfResult(
            v_safe=v,
            success=True,
            slack=slack,
            slack_exceeded=bool(slack > float(slack_threshold)),
            recovery_active=recovery_active,
        )
    return CbfResult(
        v_safe=_analytic_safe_fallback(v_ref),
        success=False,
        slack=0.0,
        slack_exceeded=False,
        recovery_active=recovery_active,
    )
