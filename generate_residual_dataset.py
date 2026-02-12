from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from residual_corrector import FEATURE_NAMES
from vlm_cbf_env import TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate residual-correction dataset from PyBullet scenes")
    parser.add_argument("--out", default="residual_dataset.csv", help="Output CSV path")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of randomized scenes")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.03, help="Noise std on measured load fractions")
    parser.add_argument("--gain", type=float, default=0.12, help="Target shift gain")
    parser.add_argument("--max-shift", type=float, default=0.12, help="Target shift clip")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = TaskConfig(gui=not args.headless, random_seed=args.seed)
    env = VlmCbfEnv(cfg)
    rows = []
    try:
        for _ in range(int(args.episodes)):
            env.reset()
            env._plan_formation()
            if not env.vlm_hypotheses or not env.vlm_hypothesis_assignments:
                continue

            hyp_idx = int(rng.integers(0, len(env.vlm_hypotheses)))
            env._set_active_hypothesis(hyp_idx)
            hypothesis = env.vlm_hypotheses[hyp_idx]
            assignment = env.vlm_hypothesis_assignments[hyp_idx]

            obj_pos, _ = env._get_object_pose(noisy=False)
            obj_center = np.array(obj_pos[:2], dtype=np.float32)
            dims = env.object_spec.dims if env.object_spec is not None else (1.0, 1.0, 1.0)
            mass = float(env.object_spec.mass) if env.object_spec is not None else 50.0
            com = np.array(env.object_spec.com_offset[:2], dtype=np.float32) if env.object_spec is not None else np.zeros(2, dtype=np.float32)
            com_world = obj_center + com

            # Force-share proxy from COM offset: closer waypoint to COM gets larger share.
            inv_dist = []
            for wp in assignment:
                d = float(np.linalg.norm(np.asarray(wp[:2], dtype=np.float32) - com_world) + 1e-3)
                inv_dist.append(1.0 / d)
            measured = np.asarray(inv_dist, dtype=np.float32)
            measured = measured / float(np.sum(measured))
            measured = measured + rng.normal(0.0, float(args.noise), size=4).astype(np.float32)
            measured = np.clip(measured, 1e-4, None)
            measured = measured / float(np.sum(measured))

            target = np.asarray(hypothesis.load_fractions, dtype=np.float32)
            target = target / float(np.sum(target))
            posterior_max = float(rng.uniform(0.4, 0.99))
            selected_conf = float(hypothesis.confidence)

            for idx_robot, robot in enumerate(env.robots):
                wp = np.asarray(assignment[idx_robot], dtype=np.float32)
                radial = wp[:2] - obj_center
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm < 1e-6:
                    radial = np.array([1.0, 0.0], dtype=np.float32)
                    radial_norm = 1.0
                dir_xy = radial / radial_norm
                tangential = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float32)

                delta_share = float(measured[idx_robot] - target[idx_robot])
                shift = float(np.clip(float(args.gain) * delta_share, -float(args.max_shift), float(args.max_shift)))
                tangent_shift = float(rng.uniform(-0.3, 0.3) * shift)
                target_xy = shift * dir_xy + tangent_shift * tangential

                row = {
                    "measured_share": float(measured[idx_robot]),
                    "target_share": float(target[idx_robot]),
                    "delta_share": float(delta_share),
                    "posterior_max": posterior_max,
                    "selected_confidence": selected_conf,
                    "payload_norm": float(robot.spec.payload / 150.0),
                    "is_heavy": 1.0 if robot.spec.payload >= 100.0 else 0.0,
                    "mass_norm": float(mass / 280.0),
                    "dim_l_norm": float(dims[0] / 2.0),
                    "dim_w_norm": float(dims[1] / 2.0),
                    "dim_h_norm": float(dims[2] / 1.0),
                    "belief_unc_pos": float(rng.uniform(0.0, 0.2)),
                    "radial_x": float(dir_xy[0]),
                    "radial_y": float(dir_xy[1]),
                    "target_dx": float(target_xy[0]),
                    "target_dy": float(target_xy[1]),
                }
                rows.append(row)
    finally:
        env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FEATURE_NAMES) + ["target_dx", "target_dy"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved residual dataset: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
