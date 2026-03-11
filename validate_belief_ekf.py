from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from belief_ekf import BeliefEKF


@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str


def _robot_layout(asymmetric: bool = False) -> np.ndarray:
    if asymmetric:
        return np.asarray(
            [
                [1.2, 0.1],
                [-1.0, 0.9],
                [-1.3, -0.7],
                [0.8, -1.1],
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [
            [1.2, 0.0],
            [0.0, 1.2],
            [-1.2, 0.0],
            [0.0, -1.2],
        ],
        dtype=np.float32,
    )


def _make_truth_filter(mass: float, com_xyz: np.ndarray, dims_xyz: np.ndarray) -> BeliefEKF:
    ekf = BeliefEKF(dt=0.1)
    ekf.initialize(mass_kg=mass, com_offset_xyz=com_xyz, dims_xyz=dims_xyz)
    ekf.x[0] = float(mass)
    ekf.x[1:4] = np.asarray(com_xyz, dtype=np.float32)
    return ekf


def _sample_measurement(
    truth: BeliefEKF,
    robot_xy: np.ndarray,
    object_center_xy: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    h = truth._measurement_model(truth.x, robot_xy, object_center_xy)
    R = truth._measurement_noise_cov(h[:-1]) * float(noise_scale * noise_scale)
    noise = rng.multivariate_normal(mean=np.zeros(h.shape[0], dtype=np.float64), cov=R.astype(np.float64))
    z = np.clip(h + noise.astype(np.float32), a_min=0.0, a_max=None)
    z[:-1] = np.clip(z[:-1], 0.0, None)
    return z[:-1].astype(np.float32), R.astype(np.float32)


def _run_filter(
    mass: float,
    com_xyz: np.ndarray,
    dims_xyz: np.ndarray,
    robot_xy: np.ndarray,
    steps: int,
    seed: int,
    prior_mass_scale: float = 2.0,
    noise_scale: float = 1.0,
) -> tuple[BeliefEKF, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = _make_truth_filter(mass=mass, com_xyz=com_xyz, dims_xyz=dims_xyz)
    ekf = BeliefEKF(dt=0.1)
    ekf.initialize(mass_kg=mass * prior_mass_scale, com_offset_xyz=np.zeros(3, dtype=np.float32), dims_xyz=dims_xyz)
    object_center_xy = np.zeros(2, dtype=np.float32)
    nis_vals = []
    for _ in range(steps):
        ekf.predict()
        forces_z, R = _sample_measurement(truth, robot_xy, object_center_xy, rng=rng, noise_scale=noise_scale)
        ekf.update(forces_z=forces_z, robot_positions_xy=robot_xy, object_center_xy=object_center_xy)
        nis_vals.append(float(ekf.diagnostics()["nis"]))
    return ekf, np.asarray(nis_vals, dtype=np.float32)


def test_mass_and_com_convergence() -> ValidationResult:
    dims = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)
    com = np.asarray([0.22, -0.14, 0.0], dtype=np.float32)
    ekf, _ = _run_filter(
        mass=80.0,
        com_xyz=com,
        dims_xyz=dims,
        robot_xy=_robot_layout(asymmetric=True),
        steps=120,
        seed=7,
        prior_mass_scale=2.0,
    )
    mass_err = abs(float(ekf.mean()[0]) - 80.0)
    com_err = float(np.linalg.norm(ekf.mean()[1:3] - com[:2]))
    passed = mass_err < 1.0 and com_err < 0.03
    return ValidationResult(
        name="mass_com_convergence",
        passed=passed,
        detail=f"mass_err={mass_err:.3f} kg, com_err={com_err:.4f} m",
    )


def test_noise_robustness() -> ValidationResult:
    dims = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)
    com = np.asarray([0.15, 0.10, 0.0], dtype=np.float32)
    mass_err = []
    com_err = []
    for seed in range(10):
        ekf, _ = _run_filter(
            mass=120.0,
            com_xyz=com,
            dims_xyz=dims,
            robot_xy=_robot_layout(asymmetric=True),
            steps=100,
            seed=seed,
            prior_mass_scale=1.5,
            noise_scale=1.5,
        )
        mu = ekf.mean()
        mass_err.append(abs(float(mu[0]) - 120.0))
        com_err.append(float(np.linalg.norm(mu[1:3] - com[:2])))
    mass_rmse = float(math.sqrt(np.mean(np.square(mass_err))))
    com_rmse = float(math.sqrt(np.mean(np.square(com_err))))
    passed = mass_rmse < 2.0 and com_rmse < 0.05
    return ValidationResult(
        name="noise_robustness",
        passed=passed,
        detail=f"mass_rmse={mass_rmse:.3f} kg, com_rmse={com_rmse:.4f} m",
    )


def test_nis_consistency() -> ValidationResult:
    dims = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)
    com = np.asarray([0.18, -0.08, 0.0], dtype=np.float32)
    _, nis = _run_filter(
        mass=100.0,
        com_xyz=com,
        dims_xyz=dims,
        robot_xy=_robot_layout(asymmetric=True),
        steps=150,
        seed=13,
        prior_mass_scale=1.8,
    )
    nis_mean = float(np.mean(nis[50:]))
    passed = 2.5 <= nis_mean <= 8.0
    return ValidationResult(
        name="nis_consistency",
        passed=passed,
        detail=f"mean_nis={nis_mean:.3f}, target_dof=5",
    )


def test_inertia_is_unobserved_but_bounded() -> ValidationResult:
    dims = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)
    com = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    ekf0 = BeliefEKF(dt=0.1)
    ekf0.initialize(mass_kg=80.0, com_offset_xyz=com, dims_xyz=dims)
    H0 = ekf0._jacobian_numeric(ekf0.mean(), _robot_layout(asymmetric=False), np.zeros(2, dtype=np.float32))
    inertia_obs0 = np.linalg.norm(H0[:, 4:7], axis=0).astype(np.float32)
    ekf, _ = _run_filter(
        mass=80.0,
        com_xyz=com,
        dims_xyz=dims,
        robot_xy=_robot_layout(asymmetric=False),
        steps=250,
        seed=3,
        prior_mass_scale=1.0,
    )
    inertia_var = np.diag(ekf.covariance())[4:7]
    passed = bool(np.all(inertia_obs0 < 1e-6) and np.all(inertia_var < 100.0))
    return ValidationResult(
        name="inertia_unobserved_bounded",
        passed=passed,
        detail=(
            f"init_obs_norms={np.round(inertia_obs0, 8).tolist()}, "
            f"var_diag={np.round(inertia_var, 4).tolist()}"
        ),
    )


def test_asymmetric_izz_observability() -> ValidationResult:
    dims = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)
    com = np.asarray([0.18, -0.10, 0.0], dtype=np.float32)
    ekf = BeliefEKF(dt=0.1)
    ekf.initialize(mass_kg=80.0, com_offset_xyz=com, dims_xyz=dims)
    robot_xy = _robot_layout(asymmetric=True)
    H = ekf._jacobian_numeric(ekf.mean(), robot_xy, np.zeros(2, dtype=np.float32))
    inertia_obs = np.linalg.norm(H[:, 4:7], axis=0).astype(np.float32)
    passed = bool(inertia_obs[2] > 1.0e-3)
    return ValidationResult(
        name="asymmetric_izz_observability",
        passed=passed,
        detail=f"obs_norms={np.round(inertia_obs, 8).tolist()}",
    )


def main() -> int:
    results = [
        test_mass_and_com_convergence(),
        test_noise_robustness(),
        test_nis_consistency(),
        test_inertia_is_unobserved_but_bounded(),
        test_asymmetric_izz_observability(),
    ]
    passed = sum(int(r.passed) for r in results)
    print("BeliefEKF validation")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    print(f"summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
