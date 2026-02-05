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

    cfg = TaskConfig(gui=not args.headless)
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
