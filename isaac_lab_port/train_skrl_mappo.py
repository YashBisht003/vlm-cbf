from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SKRL MAPPO bring-up for Isaac no-VLM cooperative transport task.")
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Max training timesteps (for metadata only).")
    parser.add_argument("--num-envs", type=int, default=128, help="Requested env count (for metadata only).")
    parser.add_argument("--print-only", action="store_true", help="Print resolved setup and exit.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short random-policy smoke test and print shape/contact diagnostics.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=500,
        help="Number of environment steps for smoke test mode.",
    )
    parser.add_argument(
        "--smoke-log-every",
        type=int,
        default=50,
        help="Smoke test logging interval (steps).",
    )
    parser.add_argument(
        "--smoke-contact-threshold",
        type=float,
        default=1.0e-3,
        help="Threshold for considering contact force as non-zero.",
    )
    parser.add_argument(
        "--smoke-require-contact",
        action="store_true",
        help="Fail smoke test if no non-zero contact force is observed.",
    )
    parser.add_argument(
        "--smoke-force-contact-probe",
        action="store_true",
        help="Teleport payload onto robot_0 panda_hand before smoke stepping to verify contact sensing.",
    )
    parser.add_argument(
        "--smoke-policy",
        choices=("random", "scripted_approach", "scripted_contact_hold", "scripted_phase"),
        default="random",
        help="Action source for smoke test.",
    )
    parser.add_argument(
        "--run-official-script",
        action="store_true",
        help="Run Isaac Lab official SKRL training script via isaaclab.sh if available.",
    )
    return parser.parse_args()


def _build_official_command(task_id: str) -> list[str]:
    return [
        "isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/skrl/train.py",
        "--task",
        task_id,
        "--algorithm",
        "MAPPO",
    ]


def _launch_app(headless: bool):
    # Keep AppLauncher import and launch before importing isaaclab env modules.
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError(
            "Failed to import isaaclab.app.AppLauncher. "
            "Install Isaac Lab 2.3.2 with Isaac Sim 5.1 and run via isaaclab.sh."
        ) from exc

    launcher = AppLauncher(headless=headless)
    return launcher, launcher.app


def _load_env(load_isaaclab_env, task_id: str, num_envs: int):
    try:
        return load_isaaclab_env(task_name=task_id, num_envs=int(num_envs))
    except TypeError:
        return load_isaaclab_env(task_name=task_id)


def _build_env_cfg(num_envs: int):
    try:
        from .direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
    except ImportError:
        from direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg

    device = "cuda:0"
    return NoVlmCoopTransportEnvCfg(
        scene=NoVlmSceneCfg(num_envs=int(num_envs), env_spacing=8.0),
        device=device,
        sim=SimulationCfg(dt=0.02, device=device),
    )


def _random_actions(wrapped):
    import torch

    actions = {}
    for agent_name, action_space in wrapped.action_spaces.items():
        if not hasattr(action_space, "shape") or len(action_space.shape) != 1:
            raise RuntimeError(f"Unsupported action space shape for {agent_name}: {getattr(action_space, 'shape', None)}")
        act_dim = int(action_space.shape[0])
        actions[agent_name] = (2.0 * torch.rand((wrapped.num_envs, act_dim), device=wrapped.device) - 1.0).to(
            dtype=torch.float32
        )
    return actions


def _scripted_approach_actions(wrapped, raw_env):
    import torch

    payload_xy = raw_env._payload_features()[:, 0:2]
    actions = {}
    for agent_name in wrapped.action_spaces:
        ego = raw_env._robot_base_features(raw_env._robot_entities[agent_name])
        delta = payload_xy - ego[:, 0:2]
        cmd_xy = torch.clamp(delta * 2.0, min=-1.0, max=1.0)
        yaw = torch.zeros((wrapped.num_envs, 1), dtype=torch.float32, device=wrapped.device)
        arm = torch.zeros((wrapped.num_envs, 3), dtype=torch.float32, device=wrapped.device)
        actions[agent_name] = torch.cat([cmd_xy, yaw, arm], dim=1).to(dtype=torch.float32)
    return actions


def _hand_xy(raw_env, agent_name: str):
    import torch

    robot = raw_env._robot_entities[agent_name]
    body_ids, _ = robot.find_bodies("panda_hand")
    if not body_ids:
        return raw_env._robot_base_features(robot)[:, 0:2]
    hand_idx = int(body_ids[0])
    body_state = robot.data.body_state_w
    if not isinstance(body_state, torch.Tensor):
        return raw_env._robot_base_features(robot)[:, 0:2]
    return body_state[:, hand_idx, 0:2].to(device=raw_env.device, dtype=torch.float32)


def _scripted_contact_hold_actions(wrapped, raw_env):
    import torch

    payload_xy = raw_env._payload_features()[:, 0:2]
    speed_limit = float(raw_env.cfg.speed_limit_mps)
    contact_threshold = float(raw_env.cfg.contact_force_threshold_n)
    actions = {}
    for agent_name in wrapped.action_spaces:
        hand_xy = _hand_xy(raw_env, agent_name)
        delta = payload_xy - hand_xy
        force_n = raw_env._safe_contact_force_norm(raw_env._robot_entities[agent_name])
        in_contact = force_n >= contact_threshold

        # Approach aggressively before contact, then switch to a damped hold controller.
        vel_approach = torch.clamp(delta * 3.0, min=-speed_limit, max=speed_limit)
        vel_hold = torch.clamp(delta * 0.75, min=-0.05, max=0.05)
        vel_cmd = torch.where(in_contact.view(-1, 1), vel_hold, vel_approach)

        cmd_xy = torch.clamp(vel_cmd / max(speed_limit, 1.0e-6), min=-1.0, max=1.0)
        yaw = torch.zeros((wrapped.num_envs, 1), dtype=torch.float32, device=wrapped.device)
        arm = torch.zeros((wrapped.num_envs, 3), dtype=torch.float32, device=wrapped.device)
        actions[agent_name] = torch.cat([cmd_xy, yaw, arm], dim=1).to(dtype=torch.float32)
    return actions


def _scripted_phase_policy_actions(wrapped, raw_env):
    import torch

    payload_xy = raw_env._payload_features()[:, 0:2]
    speed_limit = float(raw_env.cfg.speed_limit_mps)
    contact_threshold = float(raw_env.cfg.contact_force_threshold_n)
    fine_radius = float(raw_env.cfg.fine_approach_radius_m)
    approach_radius = float(raw_env.cfg.approach_radius_m)

    actions = {}
    for agent_name in wrapped.action_spaces:
        ego = raw_env._robot_base_features(raw_env._robot_entities[agent_name])
        base_xy = ego[:, 0:2]
        hand_xy = _hand_xy(raw_env, agent_name)
        base_to_payload = payload_xy - base_xy
        hand_to_payload = payload_xy - hand_xy

        base_dist = torch.linalg.vector_norm(base_to_payload, dim=1, keepdim=True)
        hand_dist = torch.linalg.vector_norm(hand_to_payload, dim=1, keepdim=True)
        force_n = raw_env._safe_contact_force_norm(raw_env._robot_entities[agent_name]).view(-1, 1)
        in_contact = force_n >= contact_threshold
        near_payload = hand_dist <= fine_radius
        in_approach_band = base_dist <= approach_radius

        # Stage 1: far away, drive harder with base centerline.
        vel_far = torch.clamp(base_to_payload * 2.5, min=-speed_limit, max=speed_limit)
        # Stage 2: near payload, use hand-relative correction and slow down.
        vel_near = torch.clamp(hand_to_payload * 1.0, min=-0.08, max=0.08)
        # Stage 3: once contact exists, keep a small inward bias to avoid drifting away.
        inward_dir = torch.where(
            hand_dist > 1.0e-6,
            hand_to_payload / torch.clamp(hand_dist, min=1.0e-6),
            torch.zeros_like(hand_to_payload),
        )
        vel_hold = torch.clamp(inward_dir * 0.03 + hand_to_payload * 0.4, min=-0.05, max=0.05)

        vel_cmd = torch.where(in_contact, vel_hold, vel_far)
        vel_cmd = torch.where((~in_contact) & (near_payload | in_approach_band), vel_near, vel_cmd)

        cmd_xy = torch.clamp(vel_cmd / max(speed_limit, 1.0e-6), min=-1.0, max=1.0)
        yaw = torch.zeros((wrapped.num_envs, 1), dtype=torch.float32, device=wrapped.device)
        arm = torch.zeros((wrapped.num_envs, 3), dtype=torch.float32, device=wrapped.device)
        actions[agent_name] = torch.cat([cmd_xy, yaw, arm], dim=1).to(dtype=torch.float32)
    return actions


def _contact_stats(raw_env, threshold: float):
    import torch

    force = getattr(raw_env, "_base_contact_force_n", None)
    if not isinstance(force, torch.Tensor):
        return None
    vals = force.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if vals.numel() == 0:
        return None
    nz = int(torch.count_nonzero(vals > float(threshold)).item())
    return {
        "mean": float(torch.mean(vals).item()),
        "max": float(torch.max(vals).item()),
        "nonzero": nz,
        "total": int(vals.numel()),
    }


def _force_contact_probe_pose(raw_env) -> None:
    import torch

    robot = raw_env._robot_entities.get("robot_0")
    payload = getattr(raw_env, "_payload_entity", None)
    if robot is None or payload is None:
        raise RuntimeError("Force-contact probe requires robot_0 and payload entities.")
    body_ids, _body_names = robot.find_bodies("panda_hand")
    if not body_ids:
        raise RuntimeError("Could not find panda_hand body for robot_0.")
    hand_idx = int(body_ids[0])
    hand_pose = robot.data.body_state_w[:, hand_idx, :7].clone()
    hand_pose[:, 2] = hand_pose[:, 2] - 0.02
    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    payload.write_root_pose_to_sim(hand_pose, env_ids=env_ids)
    payload.write_root_velocity_to_sim(
        torch.zeros((raw_env.num_envs, 6), dtype=torch.float32, device=raw_env.device),
        env_ids=env_ids,
    )
    if hasattr(payload, "write_data_to_sim"):
        payload.write_data_to_sim()
    raw_env.scene.write_data_to_sim()


def _run_smoke_test(wrapped, raw_env, args: argparse.Namespace) -> None:
    if int(args.smoke_steps) <= 0:
        raise ValueError("--smoke-steps must be > 0")
    if int(args.smoke_log_every) <= 0:
        raise ValueError("--smoke-log-every must be > 0")

    obs, _info = wrapped.reset()
    if args.smoke_force_contact_probe:
        _force_contact_probe_pose(raw_env)
    state = wrapped.state()
    first_agent = next(iter(obs.keys()))
    print(f"[smoke] policy obs shape ({first_agent}): {tuple(obs[first_agent].shape)}")
    print(f"[smoke] value state shape: {tuple(state.shape)}")

    nonzero_seen = False
    max_force_seen = 0.0
    t0 = time.time()
    for step in range(int(args.smoke_steps)):
        if args.smoke_policy == "scripted_phase":
            actions = _scripted_phase_policy_actions(wrapped, raw_env)
        elif args.smoke_policy == "scripted_contact_hold":
            actions = _scripted_contact_hold_actions(wrapped, raw_env)
        elif args.smoke_policy == "scripted_approach":
            actions = _scripted_approach_actions(wrapped, raw_env)
        else:
            actions = _random_actions(wrapped)
        wrapped.step(actions)

        stats = _contact_stats(raw_env, threshold=float(args.smoke_contact_threshold))
        if stats is not None:
            nonzero_seen = nonzero_seen or (stats["nonzero"] > 0)
            max_force_seen = max(max_force_seen, float(stats["max"]))
            if (step == 0) or ((step + 1) % int(args.smoke_log_every) == 0) or (step + 1 == int(args.smoke_steps)):
                print(
                    "[smoke] step {}/{} | contact mean {:.6f} | contact max {:.6f} | nonzero {}/{}".format(
                        step + 1,
                        int(args.smoke_steps),
                        stats["mean"],
                        stats["max"],
                        stats["nonzero"],
                        stats["total"],
                    )
                )
    elapsed = time.time() - t0
    print(f"[smoke] finished in {elapsed:.2f}s")
    print(f"[smoke] contact non-zero seen: {nonzero_seen} (max={max_force_seen:.6f})")
    if not nonzero_seen:
        print(
            "[smoke][warn] contact forces remained zero. Verify USD contact setup and wheel/contact sensor configuration."
        )
        if args.smoke_require_contact:
            raise RuntimeError(
                "Smoke test failed: no non-zero contact force observed. "
                "Use --smoke-require-contact only when contact is expected in this rollout."
            )


def main() -> None:
    args = _parse_args()

    try:
        from .check_skrl_train_script import check_script
        from .task_registry import TASK_ID, registration_summary
    except ImportError:
        from check_skrl_train_script import check_script
        from task_registry import TASK_ID, registration_summary

    script_check = check_script()
    official_cmd = _build_official_command(TASK_ID)
    reg = (
        registration_summary(force=True)
        if args.print_only
        else {
            "task_id": TASK_ID,
            "registered": "deferred_until_app_launch",
        }
    )

    payload = {
        "python": sys.version.split()[0],
        "task_registration": reg,
        "official_script_check": script_check,
        "task_id": TASK_ID,
        "requested_num_envs": int(args.num_envs),
        "requested_max_steps": int(args.max_steps),
        "official_train_command": " ".join(official_cmd + (["--headless"] if args.headless else [])),
        "cwd": str(Path.cwd()),
    }
    print(json.dumps(payload, indent=2))

    if args.print_only:
        return

    # Preferred path: official Isaac Lab SKRL script.
    if args.run_official_script:
        if bool(script_check.get("algorithm_nameerror_suspect", False)):
            raise RuntimeError(
                "Preflight failed: installed Isaac Lab SKRL train.py appears susceptible to '--algorithm' NameError. "
                "Patch train.py first or update installation."
            )
        cmd = official_cmd + (["--headless"] if args.headless else [])
        subprocess.run(cmd, check=True, cwd=os.getcwd())
        return

    # Local bootstrap path for quick validation.
    _launcher = None
    sim_app = None
    try:
        _launcher, sim_app = _launch_app(headless=bool(args.headless))
        print("[bootstrap] AppLauncher started.")
        import gymnasium as gym
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        registration_summary(force=True)
        print("[bootstrap] Imported gymnasium and SkrlVecEnvWrapper.")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Isaac Lab SKRL runtime modules. "
            "Use --run-official-script or repair the Isaac Lab installation."
        ) from exc

    try:
        env_cfg = _build_env_cfg(num_envs=int(args.num_envs))
        print(f"[bootstrap] Built env cfg for {int(args.num_envs)} envs.")
        try:
            env = gym.make(TASK_ID, cfg=env_cfg)
            print("[bootstrap] gym.make succeeded.")
            wrapped = SkrlVecEnvWrapper(env, ml_framework="torch", wrapper="isaaclab-multi-agent")
            try:
                print(f"[ok] Loaded task '{TASK_ID}' via gym.make and wrapped env as {type(wrapped).__name__}.")
                if args.smoke_test:
                    raw_env = getattr(wrapped, "_unwrapped", env)
                    _run_smoke_test(wrapped, raw_env, args)
            finally:
                wrapped.close()
        except BaseException as exc:
            print(f"[bootstrap] env pipeline exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise
    finally:
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    main()
