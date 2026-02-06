from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch

from gnn_policy import GnnPolicy
from marl_obs import build_observation, obs_dim
from vlm_cbf_env import RobotAction, TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO policy over many episodes")
    parser.add_argument("--model", required=True, help="Policy checkpoint path")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
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
    parser.add_argument("--out", default="eval_policy_results.csv", help="CSV output path")

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
    parser.add_argument("--vacuum-attach-dist", type=float, default=0.1, help="Vacuum attach distance (m)")
    parser.add_argument("--vacuum-break-dist", type=float, default=0.2, help="Vacuum break distance (m)")
    parser.add_argument(
        "--vacuum-force-margin",
        type=float,
        default=1.05,
        help="Required force margin multiplier vs object weight",
    )
    return parser.parse_args()


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_actions(action, env: VlmCbfEnv):
    actions = {}
    for idx, robot in enumerate(env.robots):
        vx = float(action[idx, 0]) * env.cfg.speed_limit
        vy = float(action[idx, 1]) * env.cfg.speed_limit
        yaw_rate = float(action[idx, 2]) * env.cfg.yaw_rate_max
        grip = bool(action[idx, 3] > 0.5)
        actions[robot.base_id] = RobotAction(base_vel=(vx, vy, yaw_rate), grip=grip)
    return actions


def _run_episode(env: VlmCbfEnv, policy: GnnPolicy, device: torch.device, deterministic: bool, max_steps: int) -> dict:
    env.reset()
    phase_times = defaultdict(float)
    prev_phase = env.phase.value
    prev_time = 0.0
    steps = 0
    info = {"phase": prev_phase, "time": 0.0, "violations": {}}
    while steps < max_steps:
        obs, pos = build_observation(env)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_out = policy.act(obs_t, pos_t, deterministic=deterministic)
        actions = _to_actions(act_out.action.cpu().numpy(), env)
        _obs, info = env.step(actions)
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
    }


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy = GnnPolicy(obs_dim(), hidden=128, msg_dim=128, layers=3).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

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
        use_cbf=not args.no_cbf,
        use_neural_cbf=not args.no_neural_cbf,
    )
    env = VlmCbfEnv(cfg)
    results = []
    try:
        for _ep in range(args.episodes):
            results.append(_run_episode(env, policy, device, args.deterministic, args.max_steps))
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
        ] + [f"time_{k}" for k in phase_keys]
        writer.writerow(header)
        for idx, res in enumerate(results):
            viol = res["violations"]
            cbf = res.get("cbf", {})
            grasp = res.get("grasp", {})
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
    cbf_rate = cbf_total["modified"] / max(cbf_total["calls"], 1)
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
