from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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


def _load_env(load_isaaclab_env, task_id: str, num_envs: int):
    try:
        return load_isaaclab_env(task_name=task_id, num_envs=int(num_envs))
    except TypeError:
        return load_isaaclab_env(task_name=task_id)


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


def _run_smoke_test(wrapped, raw_env, args: argparse.Namespace) -> None:
    if int(args.smoke_steps) <= 0:
        raise ValueError("--smoke-steps must be > 0")
    if int(args.smoke_log_every) <= 0:
        raise ValueError("--smoke-log-every must be > 0")

    obs, _info = wrapped.reset()
    state = wrapped.state()
    first_agent = next(iter(obs.keys()))
    print(f"[smoke] policy obs shape ({first_agent}): {tuple(obs[first_agent].shape)}")
    print(f"[smoke] value state shape: {tuple(state.shape)}")

    nonzero_seen = False
    max_force_seen = 0.0
    t0 = time.time()
    for step in range(int(args.smoke_steps)):
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

    reg = registration_summary(force=True)
    script_check = check_script()
    official_cmd = _build_official_command(TASK_ID)

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

    # Local bootstrap path for quick validation:
    # load_isaaclab_env + wrap_env is the supported SKRL integration path.
    try:
        from isaaclab_rl.skrl import load_isaaclab_env
    except Exception as exc:
        raise RuntimeError(
            "Failed to import isaaclab_rl.skrl.load_isaaclab_env. "
            "Use --run-official-script or install Isaac Lab RL integration modules."
        ) from exc
    try:
        from skrl.envs.wrappers.torch import wrap_env
    except Exception as exc:
        raise RuntimeError(
            "Failed to import skrl wrapper. Install skrl in the Isaac Lab environment."
        ) from exc

    env = _load_env(load_isaaclab_env, TASK_ID, num_envs=int(args.num_envs))
    wrapped = wrap_env(env, wrapper="isaaclab-multi-agent")
    try:
        print(f"[ok] Loaded task '{TASK_ID}' via load_isaaclab_env and wrapped env as {type(wrapped).__name__}.")
        if args.smoke_test:
            raw_env = getattr(wrapped, "_unwrapped", env)
            _run_smoke_test(wrapped, raw_env, args)
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
