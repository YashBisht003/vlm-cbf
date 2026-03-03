from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from .no_vlm_task_spec import NoVlmTaskSpec, RobotPreset, VacuumBackend
except ImportError:
    from no_vlm_task_spec import NoVlmTaskSpec, RobotPreset, VacuumBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isaac Lab MAPPO bootstrap (no-VLM cooperative transport)."
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="RL device.")
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel environments.")
    parser.add_argument("--num-robots", type=int, default=4, help="Robots per environment.")
    parser.add_argument("--robot-preset", choices=[p.value for p in RobotPreset], default=RobotPreset.RIDGEBACK_FRANKA.value)
    parser.add_argument(
        "--vacuum-backend",
        choices=[v.value for v in VacuumBackend],
        default=VacuumBackend.GPU_FIXED_JOINT.value,
        help="Vacuum attach backend.",
    )
    parser.add_argument(
        "--print-spec-only",
        action="store_true",
        help="Print resolved task spec and exit.",
    )
    parser.add_argument(
        "--print-env-contract",
        action="store_true",
        help="Print DirectMARLEnv contract fields (agents/spaces/prim paths) and exit.",
    )
    parser.add_argument(
        "--print-task-registration",
        action="store_true",
        help="Register task and print registry summary (without launching Isaac).",
    )
    parser.add_argument(
        "--out-spec-json",
        default="",
        help="Optional path to write resolved task spec JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed placeholder for future trainer integration.",
    )
    return parser.parse_args()


def _build_spec(args: argparse.Namespace) -> NoVlmTaskSpec:
    spec = NoVlmTaskSpec(
        robot_preset=RobotPreset(args.robot_preset),
        num_robots=int(args.num_robots),
        num_envs=int(args.num_envs),
        device=str(args.device),
        vacuum_backend=VacuumBackend(args.vacuum_backend),
    )
    spec.validate()
    return spec


def _print_env_contract(spec: NoVlmTaskSpec) -> None:
    try:
        from .direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
    except ImportError:
        from direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
    cfg = NoVlmCoopTransportEnvCfg(
        scene=NoVlmSceneCfg(num_envs=int(spec.num_envs), env_spacing=8.0),
        num_robots=int(spec.num_robots),
        device="cuda:0" if str(spec.device) == "cuda" else "cpu",
        sim=SimulationCfg(dt=0.02, device=("cuda:0" if str(spec.device) == "cuda" else "cpu")),
    )
    print("[contract] possible_agents:", cfg.possible_agents)
    print("[contract] prim_paths:", cfg.robot_prim_paths)
    print("[contract] observation_spaces keys:", tuple(cfg.observation_spaces.keys()))
    print("[contract] action_spaces keys:", tuple(cfg.action_spaces.keys()))
    print("[contract] shared_observation_spaces keys:", tuple(cfg.shared_observation_spaces.keys()))
    print("[contract] state_space:", int(cfg.state_space))
    print("[contract] scene.num_envs:", int(cfg.scene.num_envs))
    print("[contract] scene.env_spacing:", float(cfg.scene.env_spacing))
    print("[contract] scene.replicate_physics:", bool(cfg.scene.replicate_physics))
    print("[contract] sim.dt:", float(cfg.sim.dt))
    print("[contract] sim.device:", str(cfg.sim.device))
    print("[contract] robot_0.prim_path:", str(cfg.robot_0.prim_path))
    print("[contract] robot_1.prim_path:", str(cfg.robot_1.prim_path))
    print("[contract] robot_2.prim_path:", str(cfg.robot_2.prim_path))
    print("[contract] robot_3.prim_path:", str(cfg.robot_3.prim_path))
    print("[contract] payload.prim_path:", str(cfg.payload.prim_path))


def _print_task_registration() -> None:
    try:
        from .task_registry import registration_summary
    except ImportError:
        from task_registry import registration_summary
    print(json.dumps(registration_summary(force=True), indent=2))


def _write_spec_if_needed(spec: NoVlmTaskSpec, out_spec_json: str) -> None:
    if not out_spec_json:
        return
    out_path = Path(out_spec_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    print(f"[spec] wrote: {out_path}")


def _launch_app(headless: bool):
    # IMPORTANT:
    # Keep AppLauncher import and launch before torch imports in this process.
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError(
            "Failed to import isaaclab.app.AppLauncher. "
            "Install Isaac Lab 2.3.2 with Isaac Sim 5.1 and run this script in that environment."
        ) from exc

    launcher = AppLauncher(headless=headless)
    sim_app = launcher.app
    return launcher, sim_app


def _post_launch_runtime_banner(spec: NoVlmTaskSpec) -> None:
    # Deliberately imported after AppLauncher to avoid conda/libstdc++ ordering issues.
    import torch

    print("[runtime] Python:", sys.version.split()[0])
    print("[runtime] Torch:", torch.__version__)
    print("[runtime] CUDA available:", torch.cuda.is_available())
    print("[runtime] PID:", os.getpid())
    print("[task] robot preset:", spec.robot_preset.value)
    print("[task] robot usd:", spec.to_dict()["robot_asset"]["usd_path"])
    print("[task] vacuum backend:", spec.vacuum_backend.value)
    print("[task] phases:", " -> ".join(spec.phase_sequence()))


def _run_training_stub(spec: NoVlmTaskSpec, args: argparse.Namespace) -> None:
    # This is the first executable scaffold. Next step is wiring a real DirectMARLEnv + SKRL MAPPO.
    print(
        "[todo] Build Direct MARL env with fixed-joint vacuum backend, then connect SKRL MAPPO trainer."
    )
    print(
        f"[todo] Requested setup: envs={spec.num_envs}, robots={spec.num_robots}, "
        f"device={spec.device}, seed={int(args.seed)}"
    )


def main() -> None:
    args = _parse_args()
    spec = _build_spec(args)
    _write_spec_if_needed(spec, args.out_spec_json)

    if args.print_spec_only:
        print(json.dumps(spec.to_dict(), indent=2))
        return
    if args.print_env_contract:
        _print_env_contract(spec)
        return
    if args.print_task_registration:
        _print_task_registration()
        return

    _launcher = None
    sim_app = None
    try:
        _launcher, sim_app = _launch_app(headless=bool(args.headless))
        _post_launch_runtime_banner(spec)
        _run_training_stub(spec, args)
    finally:
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    main()
