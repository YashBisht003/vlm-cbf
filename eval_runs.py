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
    parser.add_argument("--no-cbf", action="store_true", help="Disable CBF/QP safety filter")
    parser.add_argument("--no-neural-cbf", action="store_true", help="Disable neural force barrier in safety layer")
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
    parser.add_argument(
        "--constraint-force-scale",
        type=float,
        default=1.5,
        help="Vacuum constraint force scale against payload",
    )
    parser.add_argument("--vacuum-attach-dist", type=float, default=0.18, help="Vacuum attach distance (m)")
    parser.add_argument("--vacuum-break-dist", type=float, default=0.30, help="Vacuum break distance (m)")
    parser.add_argument(
        "--vacuum-force-margin",
        type=float,
        default=1.05,
        help="Required force margin multiplier vs object weight",
    )
    parser.add_argument(
        "--base-drive-mode",
        choices=("velocity", "wheel"),
        default="velocity",
        help="Base drive model (velocity is robust; wheel is full wheel dynamics)",
    )
    parser.add_argument("--phase-approach-dist", type=float, default=0.25, help="Approach ready distance (m)")
    parser.add_argument(
        "--phase-approach-timeout-s",
        type=float,
        default=20.0,
        help="Approach timeout for optional quorum fallback (s)",
    )
    parser.add_argument(
        "--phase-approach-min-ready",
        type=int,
        default=4,
        help="Ready quorum count when quorum fallback is enabled",
    )
    parser.add_argument(
        "--phase-allow-quorum-fallback",
        dest="phase_allow_quorum_fallback",
        action="store_true",
        help="Allow timeout-based quorum fallback in approach phase",
    )
    parser.add_argument(
        "--no-phase-allow-quorum-fallback",
        dest="phase_allow_quorum_fallback",
        action="store_false",
        help="Disable timeout-based quorum fallback (strict consensus)",
    )
    parser.add_argument(
        "--udp-phase",
        dest="udp_phase",
        action="store_true",
        help="Enable UDP distributed phase coordination (default: enabled)",
    )
    parser.add_argument(
        "--no-udp-phase",
        dest="udp_phase",
        action="store_false",
        help="Disable UDP distributed phase coordination",
    )
    parser.add_argument(
        "--udp-neighbor-state",
        dest="udp_neighbor_state",
        action="store_true",
        help="Use UDP neighbor state in CBF (default: enabled)",
    )
    parser.add_argument(
        "--no-udp-neighbor-state",
        dest="udp_neighbor_state",
        action="store_false",
        help="Disable UDP neighbor state in CBF",
    )
    parser.add_argument("--udp-base-port", type=int, default=39000, help="Base UDP port for robot peers")
    parser.set_defaults(udp_phase=True, udp_neighbor_state=True, phase_allow_quorum_fallback=False)
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
    grasp_stats = info.get("grasp", {})
    phase_sync = info.get("phase_sync", {})
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
        "grasp": {
            "attach_attempts": grasp_stats.get("attach_attempts", 0),
            "attach_success": grasp_stats.get("attach_success", 0),
            "detach_events": grasp_stats.get("detach_events", 0),
            "overload_drop": grasp_stats.get("overload_drop", 0),
            "stretch_drop": grasp_stats.get("stretch_drop", 0),
        },
        "phase_sync": {
            "last_delay_ms": phase_sync.get("last_delay_ms", 0.0),
            "mean_delay_ms": phase_sync.get("mean_delay_ms", 0.0),
            "events": phase_sync.get("events", 0),
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
        constraint_force_scale=args.constraint_force_scale,
        vacuum_attach_dist=args.vacuum_attach_dist,
        vacuum_break_dist=args.vacuum_break_dist,
        vacuum_force_margin=args.vacuum_force_margin,
        base_drive_mode=args.base_drive_mode,
        phase_approach_dist=args.phase_approach_dist,
        phase_approach_timeout_s=args.phase_approach_timeout_s,
        phase_approach_min_ready=args.phase_approach_min_ready,
        phase_allow_quorum_fallback=args.phase_allow_quorum_fallback,
        use_udp_phase=args.udp_phase,
        use_udp_neighbor_state=args.udp_neighbor_state,
        udp_base_port=args.udp_base_port,
        use_cbf=not args.no_cbf,
        use_neural_cbf=not args.no_neural_cbf,
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
            "grasp_attach_attempts",
            "grasp_attach_success",
            "grasp_detach_events",
            "grasp_overload_drop",
            "grasp_stretch_drop",
            "phase_sync_last_ms",
            "phase_sync_mean_ms",
            "phase_sync_events",
        ] + [f"time_{k}" for k in phase_keys]
        writer.writerow(header)
        for idx, res in enumerate(results):
            viol = res["violations"]
            cbf = res.get("cbf", {})
            grasp = res.get("grasp", {})
            psync = res.get("phase_sync", {})
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
                grasp.get("attach_attempts", 0),
                grasp.get("attach_success", 0),
                grasp.get("detach_events", 0),
                grasp.get("overload_drop", 0),
                grasp.get("stretch_drop", 0),
                psync.get("last_delay_ms", 0.0),
                psync.get("mean_delay_ms", 0.0),
                psync.get("events", 0),
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
    grasp_total = {
        "attach_attempts": sum(r.get("grasp", {}).get("attach_attempts", 0) for r in results),
        "attach_success": sum(r.get("grasp", {}).get("attach_success", 0) for r in results),
        "detach_events": sum(r.get("grasp", {}).get("detach_events", 0) for r in results),
        "overload_drop": sum(r.get("grasp", {}).get("overload_drop", 0) for r in results),
        "stretch_drop": sum(r.get("grasp", {}).get("stretch_drop", 0) for r in results),
    }
    cbf_rate = (
        cbf_total["modified"] / max(cbf_total["calls"], 1)
        if cbf_total["calls"] > 0
        else 0.0
    )
    grasp_attach_rate = grasp_total["attach_success"] / max(grasp_total["attach_attempts"], 1)

    print(f"Saved: {out_path}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Mean time: {mean_time:.2f}s")
    print(f"Total violations: {total_viol}")
    print(
        "CBF stats: calls={calls} modified={modified} fallback={fallback} force_stop={force_stop} rate={rate:.3f}".format(
            rate=cbf_rate, **cbf_total
        )
    )
    print(
        "Grasp stats: attempts={attach_attempts} success={attach_success} detach={detach_events} "
        "overload_drop={overload_drop} stretch_drop={stretch_drop} attach_rate={attach_rate:.3f}".format(
            attach_rate=grasp_attach_rate, **grasp_total
        )
    )


if __name__ == "__main__":
    main()
