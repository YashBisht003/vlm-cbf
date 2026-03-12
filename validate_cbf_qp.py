from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cbf_qp


@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str


def _sep_margin(pos_i: np.ndarray, pj: np.ndarray, v: np.ndarray, vj: np.ndarray, d_min: float, alpha: float) -> float:
    delta = pos_i - pj
    B = float(np.dot(delta, delta) - d_min * d_min)
    rhs = -alpha * B + 2.0 * float(np.dot(delta, vj))
    lhs = 2.0 * float(np.dot(delta, v))
    return lhs - rhs


def test_cache_survives_zero_crossing() -> ValidationResult:
    if cbf_qp.osqp is None or cbf_qp.sparse is None:
        return ValidationResult(
            name="cache_zero_crossing",
            passed=True,
            detail="skipped: osqp unavailable in current Python",
        )
    cbf_qp._OSQP_CACHE.clear()
    cases = [
        np.array([0.30, 0.10]),
        np.array([0.00, 0.10]),
        np.array([-0.30, 0.10]),
        np.array([0.00, -0.10]),
    ]
    for pj in cases:
        _ = cbf_qp.solve_cbf_qp(
            v_des=np.array([0.1, 0.0]),
            pos_i=np.array([0.0, 0.0]),
            neighbor_pos=[pj],
            neighbor_vel=[np.array([0.0, 0.0])],
            v_max=0.25,
            d_min=0.5,
            alpha=2.0,
        )
    cache_size = len(cbf_qp._OSQP_CACHE)
    return ValidationResult(
        name="cache_zero_crossing",
        passed=cache_size == 1,
        detail=f"cache_size={cache_size}",
    )


def test_fallback_projects_multiple_constraints() -> ValidationResult:
    orig_osqp = cbf_qp.osqp
    orig_sparse = cbf_qp.sparse
    cbf_qp.osqp = None
    cbf_qp.sparse = None
    try:
        pos_i = np.array([0.0, 0.0])
        neighbors = [np.array([0.30, 0.20]), np.array([0.30, -0.20])]
        neighbor_vel = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        res = cbf_qp.solve_cbf_qp(
            v_des=np.array([0.2, 0.0]),
            pos_i=pos_i,
            neighbor_pos=neighbors,
            neighbor_vel=neighbor_vel,
            v_max=0.25,
            d_min=0.45,
            alpha=2.0,
            recovery_gain=0.5,
        )
    finally:
        cbf_qp.osqp = orig_osqp
        cbf_qp.sparse = orig_sparse
    margins = [
        _sep_margin(pos_i, pj, res.v_safe, vj, d_min=0.45, alpha=2.0)
        for pj, vj in zip(neighbors, neighbor_vel)
    ]
    passed = bool(min(margins) >= -1.0e-5)
    return ValidationResult(
        name="fallback_multi_constraint",
        passed=passed,
        detail=f"margins={np.round(np.asarray(margins), 6).tolist()}, v_safe={np.round(res.v_safe, 6).tolist()}",
    )


def test_recovery_default_active() -> ValidationResult:
    res = cbf_qp.solve_cbf_qp(
        v_des=np.array([0.2, 0.0]),
        pos_i=np.array([0.0, 0.0]),
        neighbor_pos=[np.array([0.1, 0.0])],
        neighbor_vel=[np.array([0.0, 0.0])],
        v_max=0.25,
        d_min=0.5,
        alpha=2.0,
    )
    return ValidationResult(
        name="recovery_default_active",
        passed=bool(res.recovery_active),
        detail=f"recovery_active={res.recovery_active}, v_safe={np.round(res.v_safe, 6).tolist()}",
    )


def main() -> int:
    results = [
        test_cache_survives_zero_crossing(),
        test_fallback_projects_multiple_constraints(),
        test_recovery_default_active(),
    ]
    passed = sum(int(r.passed) for r in results)
    print("CbfQP validation")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    print(f"summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
