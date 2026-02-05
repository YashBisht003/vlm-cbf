import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pybullet as p

from vlm_cbf_env import TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM-CBF PyBullet environment")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--carry-mode",
        choices=("auto", "constraint", "kinematic"),
        default="auto",
        help="Select carry mode",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--out", default="eval_results.csv", help="CSV output path")
    parser.add_argument(
        "--video-every",
        type=int,
        default=0,
        help="Record an MP4 every N episodes (0 = disabled)",
    )
    parser.add_argument("--video-dir", default="videos", help="Output directory for MP4 files")

    parser.add_argument("--pos-noise", type=float, default=0.0, help="Position noise std (m)")
    parser.add_argument("--yaw-noise", type=float, default=0.0, help="Yaw noise std (rad)")
    parser.add_argument("--force-noise", type=float, default=0.0, help="Force noise std (N)")
    parser.add_argument("--noisy-obs", action="store_true", help="Apply noise to observations")
    parser.add_argument(
        "--noisy-control",
        action="store_true",
        help="Use noisy object pose for contact control",
    )
    parser.add_argument(
        "--noisy-plan",
        action="store_true",
        help="Use noisy object pose for formation planning",
    )
    parser.add_argument(
        "--vlm-json",
        default=None,
        help="Path to VLM output JSON with waypoints and load labels",
    )
    parser.add_argument("--heavy-urdf", default=None, help="URDF path for heavy robots")
    parser.add_argument("--agile-urdf", default=None, help="URDF path for agile robots")
    parser.add_argument("--heavy-ee-link", type=int, default=None, help="End effector link index for heavy robots")
    parser.add_argument("--agile-ee-link", type=int, default=None, help="End effector link index for agile robots")
    parser.add_argument(
        "--heavy-ee-link-name",
        default=None,
        help="End effector link name for heavy robots (e.g. tool0, flange)",
    )
    parser.add_argument(
        "--agile-ee-link-name",
        default=None,
        help="End effector link name for agile robots (e.g. tool0, flange)",
    )
    parser.add_argument(
        "--robot-size-mode",
        choices=("base", "full"),
        default="base",
        help="Robot size mode for object scaling (base or full)",
    )
    parser.add_argument("--size-ratio-min", type=float, default=1.0, help="Min object size ratio to robot")
    parser.add_argument("--size-ratio-max", type=float, default=3.0, help="Max object size ratio to robot")
    return parser.parse_args()


def _run_episode(env: VlmCbfEnv, max_steps: int) -> dict:
    env.reset()
    phase_times = defaultdict(float)
    prev_phase = env.phase.value
    prev_time = 0.0
    steps = 0
    info = {"phase": prev_phase, "time": 0.0, "violations": {}}
    while steps < max_steps:
        _obs, info = env.step()
        steps += 1
        if info["phase"] != prev_phase:
            phase_times[prev_phase] += info["time"] - prev_time
            prev_phase = info["phase"]
            prev_time = info["time"]
        if info["phase"] == "done":
            break
    phase_times[prev_phase] += info["time"] - prev_time
    cbf_stats = info.get("cbf", {})
    return {
        "success": int(info["phase"] == "done"),
        "steps": steps,
        "time": info["time"],
        "phase": info["phase"],
        "carry_mode": info.get("carry_mode", "unknown"),
        "violations": info["violations"],
        "phase_times": dict(phase_times),
        "cbf": {
            "calls": cbf_stats.get("calls", 0),
            "modified": cbf_stats.get("modified", 0),
            "fallback": cbf_stats.get("fallback", 0),
            "force_stop": cbf_stats.get("force_stop", 0),
        },
    }


def main() -> None:
    args = _parse_args()
    cfg = TaskConfig(
        gui=not args.headless,
        random_seed=args.seed,
        carry_mode=args.carry_mode,
        sensor_pos_noise=args.pos_noise,
        sensor_yaw_noise=args.yaw_noise,
        sensor_force_noise=args.force_noise,
        use_noisy_obs=args.noisy_obs,
        use_noisy_control=args.noisy_control,
        use_noisy_plan=args.noisy_plan,
        vlm_json_path=args.vlm_json,
        heavy_urdf=args.heavy_urdf,
        agile_urdf=args.agile_urdf,
        heavy_ee_link=args.heavy_ee_link,
        agile_ee_link=args.agile_ee_link,
        heavy_ee_link_name=args.heavy_ee_link_name,
        agile_ee_link_name=args.agile_ee_link_name,
        object_size_ratio=(args.size_ratio_min, args.size_ratio_max),
        robot_size_mode=args.robot_size_mode,
    )
    env = VlmCbfEnv(cfg)
    results = []
    try:
        video_dir = Path(args.video_dir)
        if args.video_every and not cfg.gui:
            print("Warning: video logging works best with GUI; continuing anyway.")
        if args.video_every:
            video_dir.mkdir(parents=True, exist_ok=True)

        for ep in range(args.episodes):
            log_id = None
            if args.video_every and ep % args.video_every == 0:
                video_path = video_dir / f"episode_{ep:04d}.mp4"
                log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, str(video_path))
            results.append(_run_episode(env, args.max_steps))
            if log_id is not None:
                p.stopStateLogging(log_id)
    finally:
        env.close()

    out_path = Path(args.out)
    phase_keys = sorted({k for r in results for k in r["phase_times"].keys()})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "episode",
            "success",
            "steps",
            "time",
            "final_phase",
            "carry_mode",
            "speed_viol",
            "separation_viol",
            "force_viol",
            "cbf_calls",
            "cbf_modified",
            "cbf_fallback",
            "cbf_force_stop",
        ] + [f"time_{k}" for k in phase_keys]
        writer.writerow(header)
        for idx, res in enumerate(results):
            viol = res["violations"]
            cbf = res.get("cbf", {})
            row = [
                idx,
                res["success"],
                res["steps"],
                f"{res['time']:.3f}",
                res["phase"],
                res["carry_mode"],
                viol.get("speed", 0),
                viol.get("separation", 0),
                viol.get("force", 0),
                cbf.get("calls", 0),
                cbf.get("modified", 0),
                cbf.get("fallback", 0),
                cbf.get("force_stop", 0),
            ]
            row.extend([f"{res['phase_times'].get(k, 0.0):.3f}" for k in phase_keys])
            writer.writerow(row)

    success_rate = sum(r["success"] for r in results) / max(len(results), 1)
    mean_time = sum(r["time"] for r in results) / max(len(results), 1)
    total_viol = {
        "speed": sum(r["violations"].get("speed", 0) for r in results),
        "separation": sum(r["violations"].get("separation", 0) for r in results),
        "force": sum(r["violations"].get("force", 0) for r in results),
    }
    cbf_total = {
        "calls": sum(r.get("cbf", {}).get("calls", 0) for r in results),
        "modified": sum(r.get("cbf", {}).get("modified", 0) for r in results),
        "fallback": sum(r.get("cbf", {}).get("fallback", 0) for r in results),
        "force_stop": sum(r.get("cbf", {}).get("force_stop", 0) for r in results),
    }
    cbf_rate = (
        cbf_total["modified"] / max(cbf_total["calls"], 1)
        if cbf_total["calls"] > 0
        else 0.0
    )

    print(f"Saved: {out_path}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Mean time: {mean_time:.2f}s")
    print(f"Total violations: {total_viol}")
    print(
        "CBF stats: calls={calls} modified={modified} fallback={fallback} force_stop={force_stop} rate={rate:.3f}".format(
            rate=cbf_rate, **cbf_total
        )
    )


if __name__ == "__main__":
    main()
