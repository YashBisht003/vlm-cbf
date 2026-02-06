from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import torch
import torch.nn as nn
import torch.optim as optim

from gnn_policy import CentralCritic, GnnPolicy
from marl_obs import build_global_state, build_observation, global_state_dim, obs_dim
from neural_cbf import neural_cbf_temporal_loss
from train_logger import TrainLogger
from vlm_cbf_env import Phase, RobotAction, TaskConfig, VlmCbfEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPPO-style training (GNN policy, centralized critic)")
    parser.add_argument("--updates", type=int, default=1600, help="Number of PPO updates")
    parser.add_argument("--steps-per-update", type=int, default=512, help="Rollout steps per update")
    parser.add_argument("--epochs", type=int, default=5, help="PPO epochs per update")
    parser.add_argument("--minibatch", type=int, default=32, help="Minibatch size (time steps)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--torch-threads", type=int, default=0, help="Torch CPU threads (0 = default)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
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
        help="Vacuum constraint force scale against robot payload (x payload*9.81)",
    )
    parser.add_argument("--vacuum-break-dist", type=float, default=0.30, help="Vacuum break distance (m)")
    parser.add_argument("--vacuum-attach-dist", type=float, default=0.18, help="Vacuum attach distance (m)")
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
        help="Approach timeout before quorum fallback (s)",
    )
    parser.add_argument(
        "--phase-approach-min-ready",
        type=int,
        default=2,
        help="Ready quorum for approach timeout fallback",
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
    parser.add_argument("--cbf-risk-s-speed", type=float, default=1.0, help="Risk severity for speed CBF")
    parser.add_argument("--cbf-risk-s-sep", type=float, default=1.0, help="Risk severity for separation CBF")
    parser.add_argument("--cbf-risk-s-force", type=float, default=1.0, help="Risk severity for force CBF")
    parser.add_argument("--cbf-risk-s-neural", type=float, default=1.0, help="Risk severity for neural CBF")
    parser.add_argument(
        "--neural-cbf",
        dest="use_neural_cbf",
        action="store_true",
        help="Enable neural force barrier h_phi(s,b) (default: enabled)",
    )
    parser.add_argument(
        "--no-neural-cbf",
        dest="use_neural_cbf",
        action="store_false",
        help="Disable neural force barrier",
    )
    parser.add_argument("--neural-cbf-model", default="", help="Path to neural CBF model checkpoint")
    parser.add_argument("--neural-cbf-hidden", type=int, default=64, help="Neural CBF hidden size")
    parser.add_argument("--neural-cbf-device", default="cpu", help="Device for neural CBF runtime")
    parser.add_argument("--neural-cbf-alpha", type=float, default=1.0, help="Alpha for neural CBF inequality")
    parser.add_argument("--neural-cbf-fd-eps", type=float, default=1e-3, help="Finite diff epsilon for neural CBF")
    parser.add_argument(
        "--neural-cbf-force-vel-gain",
        type=float,
        default=4.0,
        help="Coupling gain from approach velocity to normalized force in neural CBF model",
    )

    parser.add_argument("--cbf-epsilon", type=float, default=0.2, help="Barrier shaping epsilon")
    parser.add_argument("--w-cbf-prox", type=float, default=0.2, help="Weight for CBF proximity penalty")
    parser.add_argument("--w-int", type=float, default=0.05, help="Weight for CBF intervention penalty")
    parser.add_argument("--w-force-balance", type=float, default=0.05, help="Weight for force balance penalty")
    parser.add_argument("--w-load-share", type=float, default=0.05, help="Weight for payload-share coordination penalty")
    parser.add_argument("--w-tilt", type=float, default=0.1, help="Weight for object tilt penalty")
    parser.add_argument("--w-neural", type=float, default=0.05, help="Weight for neural barrier term")
    parser.add_argument(
        "--w-belief-int",
        type=float,
        default=0.03,
        help="Extra intervention penalty scaled by belief uncertainty",
    )
    parser.add_argument("--train-neural-cbf", action="store_true", help="Train neural CBF online from rollout data")
    parser.add_argument("--neural-cbf-lr", type=float, default=1e-3, help="Neural CBF optimizer learning rate")
    parser.add_argument("--neural-cbf-epochs", type=int, default=2, help="Neural CBF update epochs per PPO update")
    parser.add_argument("--neural-cbf-margin", type=float, default=0.0, help="Margin for temporal neural CBF residual loss")
    parser.add_argument(
        "--neural-cbf-train-alpha",
        type=float,
        default=1.0,
        help="Alpha used in temporal neural CBF residual labels",
    )

    parser.add_argument("--out", default="mappo_policy.pt", help="Final checkpoint path")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for periodic checkpoints")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint cadence (updates)")
    parser.add_argument(
        "--save-latest",
        dest="save_latest",
        action="store_true",
        help="Save a rolling latest checkpoint each update (default: enabled)",
    )
    parser.add_argument(
        "--no-save-latest",
        dest="save_latest",
        action="store_false",
        help="Disable rolling latest checkpoint",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest/best checkpoint")
    parser.add_argument("--resume-path", default="", help="Explicit checkpoint path to resume")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoint if available")
    parser.add_argument(
        "--best-metric",
        choices=("success_rate", "mean_reward"),
        default="success_rate",
        help="Metric for best checkpoint tracking",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Print status every N updates")
    parser.add_argument(
        "--verify-checkpoints",
        dest="verify_checkpoints",
        action="store_true",
        help="Load and validate checkpoints right after saving (default: enabled)",
    )
    parser.add_argument(
        "--no-verify-checkpoints",
        dest="verify_checkpoints",
        action="store_false",
        help="Disable checkpoint load-back verification",
    )
    parser.add_argument("--max-steps", type=int, default=4000, help="Max steps per episode")
    parser.add_argument("--log-csv", default="train_metrics.csv", help="CSV log output")
    parser.set_defaults(
        save_latest=True,
        verify_checkpoints=True,
        udp_phase=True,
        udp_neighbor_state=True,
        use_neural_cbf=True,
    )
    return parser.parse_args()


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_actions(action: np.ndarray, env: VlmCbfEnv) -> Dict[int, RobotAction]:
    actions: Dict[int, RobotAction] = {}
    for idx, robot in enumerate(env.robots):
        vx = float(action[idx, 0]) * env.cfg.speed_limit
        vy = float(action[idx, 1]) * env.cfg.speed_limit
        yaw_rate = float(action[idx, 2]) * env.cfg.yaw_rate_max
        grip = bool(action[idx, 3] > 0.5)
        actions[robot.base_id] = RobotAction(base_vel=(vx, vy, yaw_rate), grip=grip)
    return actions


def _mean_distance(points_a: List[np.ndarray], points_b: List[np.ndarray]) -> float:
    if not points_a or not points_b:
        return 0.0
    dists = [float(np.linalg.norm(a[:2] - b[:2])) for a, b in zip(points_a, points_b)]
    return float(np.mean(dists)) if dists else 0.0


def _compute_reward(env: VlmCbfEnv, info: Dict, prev_viol: Dict[str, int], args: argparse.Namespace) -> Tuple[float, Dict[str, int]]:
    obj_pos, _ = env._get_object_pose(noisy=False)
    obj_pos = np.array(obj_pos, dtype=np.float32)
    reward = -0.01

    robot_positions = []
    for robot in env.robots:
        pos, _ = env._get_robot_pose(robot, noisy=False)
        robot_positions.append(np.array(pos, dtype=np.float32))

    if env.phase in (Phase.PLAN, Phase.APPROACH):
        waypoints = [r.waypoint if r.waypoint is not None else pos for r, pos in zip(env.robots, robot_positions)]
        reward -= _mean_distance(robot_positions, waypoints)
    elif env.phase == Phase.CONTACT:
        targets = [obj_pos for _ in robot_positions]
        reward -= _mean_distance(robot_positions, targets)
    elif env.phase == Phase.LIFT:
        reward += float(env.current_lift) * 2.0
    elif env.phase == Phase.TRANSPORT:
        goal_dist = float(np.linalg.norm(obj_pos[:2] - env.goal_pos[:2]))
        reward -= goal_dist
    elif env.phase == Phase.PLACE:
        reward -= abs(float(obj_pos[2]) - float(env.podium_height))
    elif env.phase == Phase.DONE:
        reward += 10.0

    delta_speed = env.violations["speed"] - prev_viol.get("speed", 0)
    delta_sep = env.violations["separation"] - prev_viol.get("separation", 0)
    delta_force = env.violations["force"] - prev_viol.get("force", 0)
    reward -= 0.05 * delta_speed + 0.1 * delta_sep + 0.1 * delta_force

    cbf_info = info.get("cbf", {})
    eps = max(float(args.cbf_epsilon), 1e-6)
    neural_barrier_raw = float(cbf_info.get("neural_barrier_min", 1.0))
    neural_barrier = float(np.tanh(neural_barrier_raw))
    barrier_vals = [
        float(cbf_info.get("sep_barrier_norm_min", 1.0)),
        float(cbf_info.get("speed_barrier_norm_min", 1.0)),
        float(cbf_info.get("force_barrier_norm_min", 1.0)),
        neural_barrier,
    ]
    prox_penalty = 0.0
    for value in barrier_vals:
        prox_penalty += (max(0.0, eps - value) / eps) ** 2
    neural_margin = float(cbf_info.get("neural_constraint_margin_min", 1.0))
    prox_penalty += (max(0.0, -neural_margin) / eps) ** 2
    reward -= float(args.w_cbf_prox) * prox_penalty

    intervention_pen = float(cbf_info.get("intervention_l2", 0.0))
    reward -= float(args.w_int) * intervention_pen
    belief_unc = float(info.get("belief", {}).get("uncertainty_pos", 0.0))
    reward -= float(args.w_belief_int) * belief_unc * intervention_pen
    reward += float(args.w_neural) * neural_barrier

    if env.phase in (Phase.CONTACT, Phase.LIFT, Phase.TRANSPORT):
        forces = np.array(info.get("contact_forces", []), dtype=np.float32)
        if forces.size > 0:
            force_norm = forces / max(env.cfg.contact_force_max, 1e-6)
            reward -= float(args.w_force_balance) * float(np.var(force_norm))
            total_force = float(np.sum(np.maximum(forces, 0.0)))
            if total_force > 1e-6 and len(env.robots) == int(forces.shape[0]):
                measured_share = np.maximum(forces, 0.0) / total_force
                payload = np.array([max(1e-6, float(robot.spec.payload)) for robot in env.robots], dtype=np.float32)
                target_share = payload / float(np.sum(payload))
                reward -= float(args.w_load_share) * float(np.mean((measured_share - target_share) ** 2))

    if env.phase in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
        reward -= float(args.w_tilt) * float(info.get("object_tilt_rad", 0.0))

    return float(reward), dict(env.violations)


def _compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _latest_update_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    latest_path: Optional[Path] = None
    latest_update = -1
    for path in checkpoint_dir.glob("mappo_policy_update_*.pt"):
        match = re.search(r"mappo_policy_update_(\d+)\.pt$", path.name)
        if not match:
            continue
        update_id = int(match.group(1))
        if update_id > latest_update:
            latest_update = update_id
            latest_path = path
    return latest_path


def _verify_checkpoint(path: Path, expected_update: int) -> None:
    if not path.exists():
        raise RuntimeError(f"Checkpoint verification failed: missing file {path}")
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Checkpoint verification failed while loading {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint verification failed: payload is not a dict in {path}")
    required = ("policy", "critic", "update", "best_metric")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Checkpoint verification failed: missing keys {missing} in {path}")
    saved_update = int(payload.get("update", -1))
    if saved_update != int(expected_update):
        raise RuntimeError(
            f"Checkpoint verification failed: expected update {expected_update}, got {saved_update} in {path}"
        )


def main() -> None:
    args = _parse_args()
    if args.updates <= 0:
        raise ValueError("--updates must be > 0")
    if args.steps_per_update <= 0:
        raise ValueError("--steps-per-update must be > 0")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0")
    device = _select_device(args.device)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    seed = args.seed if args.seed is not None else int(time.time())
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = TaskConfig(
        gui=not args.headless,
        random_seed=args.seed,
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
        use_udp_phase=args.udp_phase,
        use_udp_neighbor_state=args.udp_neighbor_state,
        udp_base_port=args.udp_base_port,
        cbf_risk_s_speed=args.cbf_risk_s_speed,
        cbf_risk_s_sep=args.cbf_risk_s_sep,
        cbf_risk_s_force=args.cbf_risk_s_force,
        cbf_risk_s_neural=args.cbf_risk_s_neural,
        use_neural_cbf=args.use_neural_cbf,
        neural_cbf_model_path=(args.neural_cbf_model if args.neural_cbf_model else None),
        neural_cbf_hidden=args.neural_cbf_hidden,
        neural_cbf_device=args.neural_cbf_device,
        neural_cbf_alpha=args.neural_cbf_alpha,
        neural_cbf_fd_eps=args.neural_cbf_fd_eps,
        neural_cbf_force_vel_gain=args.neural_cbf_force_vel_gain,
        cbf_eps=args.cbf_epsilon,
    )
    env = VlmCbfEnv(cfg)
    env.reset()
    neural_cbf_opt: Optional[optim.Optimizer] = None
    if args.train_neural_cbf and cfg.use_neural_cbf and env.neural_cbf is not None:
        neural_cbf_opt = optim.Adam(env.neural_cbf.model.parameters(), lr=args.neural_cbf_lr)

    n_agents = len(env.robots)
    policy = GnnPolicy(obs_dim(), hidden=128, msg_dim=128, layers=3).to(device)
    critic = CentralCritic(obs_dim(), n_agents=n_agents, global_dim=global_state_dim()).to(device)

    policy_opt = optim.Adam(policy.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    logger = TrainLogger(args.log_csv)

    checkpoint_path = Path(args.out)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_update = 1
    best_metric = -1e9

    def _load_checkpoint(path: Path) -> bool:
        nonlocal start_update, best_metric
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return False
        ckpt = torch.load(path, map_location=device)
        try:
            policy.load_state_dict(ckpt["policy"])
            critic.load_state_dict(ckpt["critic"])
        except Exception as exc:
            print(f"Checkpoint shape mismatch for {path}: {exc}")
            return False
        if "policy_opt" in ckpt:
            policy_opt.load_state_dict(ckpt["policy_opt"])
        if "critic_opt" in ckpt:
            critic_opt.load_state_dict(ckpt["critic_opt"])
        start_update = int(ckpt.get("update", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"Resumed from: {path} (next update {start_update})")
        return True

    if args.resume or args.resume_path or args.resume_best:
        resume_candidates: List[Path] = []
        if args.resume_path:
            resume_candidates.append(Path(args.resume_path))
        elif args.resume_best:
            resume_candidates.append(checkpoint_dir / "mappo_policy_best.pt")
        else:
            resume_candidates.append(checkpoint_dir / "mappo_policy_latest.pt")
            latest_update_ckpt = _latest_update_checkpoint(checkpoint_dir)
            if latest_update_ckpt is not None:
                resume_candidates.append(latest_update_ckpt)
            resume_candidates.append(checkpoint_path)

        loaded = False
        seen = set()
        for candidate in resume_candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            loaded = _load_checkpoint(candidate)
            if loaded:
                break
        if not loaded:
            print("Resume requested but no checkpoint was loaded; starting from update 1.")

    if start_update > args.updates:
        print(
            f"Checkpoint update already reached target: start_update={start_update}, updates={args.updates}. Nothing to run."
        )
        env.close()
        return

    for update in range(start_update, args.updates + 1):
        update_start_t = time.time()
        env.reset()
        rollout_obs = []
        rollout_global = []
        rollout_pos = []
        rollout_pre = []
        rollout_grip = []
        rollout_logp = []
        rollout_vals = []
        rollout_rewards = []
        rollout_dones = []
        rollout_cbf_calls = 0
        rollout_cbf_modified = 0
        rollout_cbf_fallback = 0
        rollout_cbf_force_stop = 0
        rollout_int_l2 = 0.0
        rollout_cbf_prox = 0.0
        rollout_force_balance = 0.0
        rollout_load_share = 0.0
        rollout_tilt = 0.0
        rollout_belief_unc = 0.0
        rollout_neural_x_prev: List[np.ndarray] = []
        rollout_neural_x_next: List[np.ndarray] = []
        prev_neural_by_robot: Dict[str, Tuple[np.ndarray, float]] = {}
        neural_loss_value = 0.0
        episode_successes = 0
        episode_count = 0
        prev_viol = dict(env.violations)
        steps = 0
        episode_steps = 0

        while steps < args.steps_per_update:
            obs, pos = build_observation(env)
            global_state = build_global_state(env)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
            global_t = torch.tensor(global_state, dtype=torch.float32, device=device)
            with torch.no_grad():
                act_out = policy.act(obs_t, pos_t, deterministic=False)
                value = critic(obs_t, global_t)

            actions = _to_actions(act_out.action.cpu().numpy(), env)
            _obs, info = env.step(actions)
            reward, prev_viol = _compute_reward(env, info, prev_viol, args)
            done = float(info["phase"] == "done")
            cbf = info.get("cbf", {})
            rollout_cbf_calls += int(cbf.get("calls", 0))
            rollout_cbf_modified += int(cbf.get("modified", 0))
            rollout_cbf_fallback += int(cbf.get("fallback", 0))
            rollout_cbf_force_stop += int(cbf.get("force_stop", 0))
            rollout_int_l2 += float(cbf.get("intervention_l2", 0.0))
            eps = max(float(args.cbf_epsilon), 1e-6)
            rollout_cbf_prox += (max(0.0, eps - float(cbf.get("sep_barrier_norm_min", 1.0))) / eps) ** 2
            rollout_cbf_prox += (max(0.0, eps - float(cbf.get("speed_barrier_norm_min", 1.0))) / eps) ** 2
            rollout_cbf_prox += (max(0.0, eps - float(cbf.get("force_barrier_norm_min", 1.0))) / eps) ** 2
            rollout_cbf_prox += (max(0.0, eps - float(np.tanh(float(cbf.get("neural_barrier_min", 1.0))))) / eps) ** 2
            rollout_cbf_prox += (max(0.0, -float(cbf.get("neural_constraint_margin_min", 1.0))) / eps) ** 2
            forces = np.array(info.get("contact_forces", []), dtype=np.float32)
            if forces.size > 0:
                rollout_force_balance += float(np.var(forces / max(env.cfg.contact_force_max, 1e-6)))
                total_force = float(np.sum(np.maximum(forces, 0.0)))
                if total_force > 1e-6 and len(env.robots) == int(forces.shape[0]):
                    measured_share = np.maximum(forces, 0.0) / total_force
                    payload = np.array([max(1e-6, float(robot.spec.payload)) for robot in env.robots], dtype=np.float32)
                    target_share = payload / float(np.sum(payload))
                    rollout_load_share += float(np.mean((measured_share - target_share) ** 2))
            rollout_tilt += float(info.get("object_tilt_rad", 0.0))
            rollout_belief_unc += float(info.get("belief", {}).get("uncertainty_pos", 0.0))
            if neural_cbf_opt is not None:
                for robot_name, robot_metrics in env.cbf_step_metrics.items():
                    neural_input = robot_metrics.get("neural_input")
                    h_curr = float(robot_metrics.get("neural_barrier", 0.0))
                    prev_sample = prev_neural_by_robot.get(robot_name)
                    if prev_sample is not None:
                        prev_input, _h_prev = prev_sample
                        force_safe = float(robot_metrics.get("force_barrier_raw", -1.0)) >= 0.0
                        constraint_safe = float(robot_metrics.get("neural_constraint_margin", -1.0)) >= -1e-4
                        if force_safe and constraint_safe and neural_input is not None:
                            rollout_neural_x_prev.append(prev_input.astype(np.float32))
                            rollout_neural_x_next.append(np.asarray(neural_input, dtype=np.float32))
                    if neural_input is not None:
                        prev_neural_by_robot[robot_name] = (np.asarray(neural_input, dtype=np.float32), h_curr)
                    else:
                        prev_neural_by_robot.pop(robot_name, None)

            rollout_obs.append(obs)
            rollout_global.append(global_state)
            rollout_pos.append(pos)
            rollout_pre.append(act_out.pre_tanh.cpu().numpy())
            rollout_grip.append(act_out.grip_action.cpu().numpy())
            rollout_logp.append(act_out.logprob.cpu().numpy())
            rollout_vals.append(float(value.cpu().item()))
            rollout_rewards.append(reward)
            rollout_dones.append(done)

            steps += 1
            episode_steps += 1
            if done or episode_steps >= args.max_steps:
                episode_count += 1
                if done:
                    episode_successes += 1
                env.reset()
                prev_viol = dict(env.violations)
                prev_neural_by_robot.clear()
                episode_steps = 0

        advantages, returns = _compute_gae(
            np.array(rollout_rewards, dtype=np.float32),
            np.array(rollout_vals, dtype=np.float32),
            np.array(rollout_dones, dtype=np.float32),
            args.gamma,
            args.gae_lambda,
        )
        cbf_rate = rollout_cbf_modified / max(rollout_cbf_calls, 1)
        success_rate = float(episode_successes / max(episode_count, 1))

        obs_arr = np.stack(rollout_obs)  # (T, N, obs_dim)
        global_arr = np.stack(rollout_global)  # (T, G)
        pos_arr = np.stack(rollout_pos)
        pre_arr = np.stack(rollout_pre)
        grip_arr = np.stack(rollout_grip)
        old_logp = np.stack(rollout_logp)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t_all = torch.tensor(advantages, dtype=torch.float32, device=device)

        time_indices = np.arange(obs_arr.shape[0])
        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        global_tensor = torch.tensor(global_arr, dtype=torch.float32, device=device)
        pos_tensor = torch.tensor(pos_arr, dtype=torch.float32, device=device)
        pre_tensor = torch.tensor(pre_arr, dtype=torch.float32, device=device)
        grip_tensor = torch.tensor(grip_arr, dtype=torch.float32, device=device)
        old_logp_tensor = torch.tensor(old_logp, dtype=torch.float32, device=device)

        for _epoch in range(args.epochs):
            rng.shuffle(time_indices)
            for start in range(0, len(time_indices), args.minibatch):
                batch = time_indices[start : start + args.minibatch]
                actor_loss = 0.0
                entropy_loss = 0.0
                for t in batch:
                    obs_t = obs_tensor[t]
                    pos_t = pos_tensor[t]
                    pre_t = pre_tensor[t]
                    grip_t = grip_tensor[t]
                    old_logp_t = old_logp_tensor[t]
                    adv_t = adv_t_all[t]

                    new_logp, entropy = policy.evaluate_actions(obs_t, pos_t, pre_t, grip_t)
                    ratio = torch.exp(new_logp - old_logp_t)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_t
                    actor_loss = actor_loss + (-torch.min(surr1, surr2).mean() - args.entropy * entropy.mean())
                    entropy_loss = entropy_loss + entropy.mean()

                actor_loss = actor_loss / max(1, len(batch))
                policy_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_opt.step()

            value_pred = critic(obs_tensor, global_tensor)
            value_loss = (returns_t - value_pred).pow(2).mean()

            critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()

        if neural_cbf_opt is not None and env.neural_cbf is not None and rollout_neural_x_prev:
            x_prev_arr = np.stack(rollout_neural_x_prev).astype(np.float32)
            x_next_arr = np.stack(rollout_neural_x_next).astype(np.float32)
            idx = np.arange(x_prev_arr.shape[0])
            model = env.neural_cbf.model
            model.train()
            loss_acc = 0.0
            loss_count = 0
            for _ in range(max(1, args.neural_cbf_epochs)):
                rng.shuffle(idx)
                for start in range(0, len(idx), args.minibatch):
                    batch_idx = idx[start : start + args.minibatch]
                    batch_x_prev = torch.tensor(x_prev_arr[batch_idx], dtype=torch.float32, device=env.neural_cbf.device)
                    batch_x_next = torch.tensor(x_next_arr[batch_idx], dtype=torch.float32, device=env.neural_cbf.device)
                    h_prev = model(batch_x_prev)
                    h_next = model(batch_x_next)
                    loss = neural_cbf_temporal_loss(
                        h_prev=h_prev,
                        h_next=h_next,
                        alpha_dt=args.neural_cbf_train_alpha * env.control_dt,
                        margin=args.neural_cbf_margin,
                    )
                    neural_cbf_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    neural_cbf_opt.step()
                    loss_acc += float(loss.item())
                    loss_count += 1
            model.eval()
            neural_loss_value = loss_acc / max(loss_count, 1)
            if args.neural_cbf_model:
                env.neural_cbf.save(args.neural_cbf_model)

        current_metric = float(episode_successes / max(episode_count, 1))
        if args.best_metric == "mean_reward":
            current_metric = float(np.mean(rollout_rewards))
        if current_metric > best_metric:
            best_metric = current_metric
            best_ckpt = {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "policy_opt": policy_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "obs_dim": obs_dim(),
                "n_agents": n_agents,
                "global_dim": global_state_dim(),
                "update": update,
                "best_metric": best_metric,
            }
            _atomic_torch_save(best_ckpt, checkpoint_dir / "mappo_policy_best.pt")
            if args.verify_checkpoints:
                _verify_checkpoint(checkpoint_dir / "mappo_policy_best.pt", update)

        checkpoint = {
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "policy_opt": policy_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "obs_dim": obs_dim(),
            "n_agents": n_agents,
            "global_dim": global_state_dim(),
            "update": update,
            "best_metric": best_metric,
        }
        if args.save_latest:
            _atomic_torch_save(checkpoint, checkpoint_dir / "mappo_policy_latest.pt")
            # Avoid expensive load-back every update; verify latest at periodic boundaries.
            if args.verify_checkpoints and (update % args.save_every == 0 or update == args.updates):
                _verify_checkpoint(checkpoint_dir / "mappo_policy_latest.pt", update)
        if update % args.save_every == 0 or update == args.updates:
            ckpt_path = checkpoint_dir / f"mappo_policy_update_{update:04d}.pt"
            _atomic_torch_save(checkpoint, ckpt_path)
            if args.verify_checkpoints:
                _verify_checkpoint(ckpt_path, update)
            if update == args.updates:
                _atomic_torch_save(checkpoint, checkpoint_path)
                if args.verify_checkpoints:
                    _verify_checkpoint(checkpoint_path, update)

        logger.write(
            {
                "update": update,
                "mean_reward": float(np.mean(rollout_rewards)),
                "value_loss": float(value_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "entropy": float(entropy_loss.item()) if hasattr(entropy_loss, "item") else float(entropy_loss),
                "episodes": float(episode_count),
                "success_rate": success_rate,
                "cbf_calls": float(rollout_cbf_calls),
                "cbf_modified": float(rollout_cbf_modified),
                "cbf_fallback": float(rollout_cbf_fallback),
                "cbf_force_stop": float(rollout_cbf_force_stop),
                "cbf_rate": float(cbf_rate),
                "cbf_intervention_l2": float(rollout_int_l2 / max(steps, 1)),
                "cbf_proximity": float(rollout_cbf_prox / max(steps, 1)),
                "force_balance_var": float(rollout_force_balance / max(steps, 1)),
                "load_share_mse": float(rollout_load_share / max(steps, 1)),
                "object_tilt_mean": float(rollout_tilt / max(steps, 1)),
                "belief_uncertainty_mean": float(rollout_belief_unc / max(steps, 1)),
                "neural_cbf_loss": float(neural_loss_value),
                "update_sec": float(time.time() - update_start_t),
            }
        )

        if update == start_update or update % max(1, args.log_interval) == 0 or update == args.updates:
            print(
                f"Update {update}/{args.updates} | mean reward {np.mean(rollout_rewards):.3f} | "
                f"value loss {value_loss.item():.4f} | actor loss {actor_loss.item():.4f} | "
                f"cbf rate {cbf_rate:.3f} | neural cbf loss {neural_loss_value:.4f} | "
                f"belief unc {rollout_belief_unc / max(steps, 1):.4f} | "
                f"{time.time() - update_start_t:.2f}s/update"
            )

    env.close()


if __name__ == "__main__":
    main()
