from __future__ import annotations

import argparse
from pathlib import Path
import time

import pybullet as p
import torch

from gnn_policy import GnnPolicy
from marl_obs import build_observation, obs_dim
from vlm_cbf_env import RobotAction, TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained GNN policy in the PyBullet env")
    parser.add_argument("--model", default="mappo_policy.pt", help="Policy checkpoint")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 = until done)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--no-sleep", action="store_true", help="Run as fast as possible")
    parser.add_argument(
        "--carry-mode",
        choices=("auto", "constraint", "kinematic"),
        default="auto",
        help="Object carry mode in environment",
    )
    parser.add_argument(
        "--robot-size-mode",
        choices=("base", "full"),
        default="base",
        help="Robot size reference mode for object scaling",
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
    parser.add_argument("--phase-approach-timeout-s", type=float, default=20.0, help="Approach timeout (s)")
    parser.add_argument("--phase-approach-min-ready", type=int, default=2, help="Approach timeout quorum")
    parser.add_argument("--udp-base-port", type=int, default=39000, help="Base UDP port for robot peers")
    parser.add_argument("--video", action="store_true", help="Record MP4 video")
    parser.add_argument("--video-path", default="policy_demo.mp4", help="Output MP4 path")
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


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)

    ckpt = torch.load(args.model, map_location=device)
    policy = GnnPolicy(obs_dim(), hidden=128, msg_dim=128, layers=3).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    cfg = TaskConfig(
        gui=not args.headless,
        carry_mode=args.carry_mode,
        robot_size_mode=args.robot_size_mode,
        object_size_ratio=(args.size_ratio_min, args.size_ratio_max),
        constraint_force_scale=args.constraint_force_scale,
        vacuum_attach_dist=args.vacuum_attach_dist,
        vacuum_break_dist=args.vacuum_break_dist,
        vacuum_force_margin=args.vacuum_force_margin,
        base_drive_mode=args.base_drive_mode,
        phase_approach_dist=args.phase_approach_dist,
        phase_approach_timeout_s=args.phase_approach_timeout_s,
        phase_approach_min_ready=args.phase_approach_min_ready,
        udp_base_port=args.udp_base_port,
    )
    env = VlmCbfEnv(cfg)
    env.reset()

    log_id = None
    if args.video:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.video_path)

    steps = 0
    try:
        while True:
            obs, pos = build_observation(env)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
            with torch.no_grad():
                act_out = policy.act(obs_t, pos_t, deterministic=args.deterministic)
            actions = _to_actions(act_out.action.cpu().numpy(), env)
            _obs, info = env.step(actions)
            steps += 1
            if info["phase"] == "done":
                break
            if args.steps and steps >= args.steps:
                break
            if not args.no_sleep:
                time.sleep(env.control_dt)
    finally:
        if log_id is not None:
            p.stopStateLogging(log_id)
        env.close()
    print(f"Finished after {steps} steps | phase={info['phase']}")


if __name__ == "__main__":
    main()
