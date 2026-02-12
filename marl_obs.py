from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pybullet as p

from vlm_cbf_env import Phase, VlmCbfEnv


PHASES: List[str] = [
    Phase.OBSERVE.value,
    Phase.PLAN.value,
    Phase.APPROACH.value,
    Phase.FINE_APPROACH.value,
    Phase.CONTACT.value,
    Phase.PROBE.value,
    Phase.CORRECT.value,
    Phase.LIFT.value,
    Phase.TRANSPORT.value,
    Phase.PLACE.value,
    Phase.DONE.value,
]


def _phase_one_hot(phase: str) -> np.ndarray:
    vec = np.zeros(len(PHASES), dtype=np.float32)
    if phase in PHASES:
        vec[PHASES.index(phase)] = 1.0
    return vec


def _yaw_from_quat(quat) -> float:
    return p.getEulerFromQuaternion(quat)[2]


def _wrap_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return float(angle)


def build_observation(env: VlmCbfEnv) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        obs: (N, obs_dim)
        positions: (N, 2) xy positions for message passing
    """
    obj_pos, obj_quat = env._get_object_pose(noisy=env.cfg.use_noisy_obs)
    obj_pos = np.array(obj_pos, dtype=np.float32)
    obj_yaw = float(_yaw_from_quat(obj_quat))
    goal = np.array(env.goal_pos, dtype=np.float32)
    phase_vec = _phase_one_hot(env.phase.value)

    obs_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []

    for robot in env.robots:
        pos, quat = env._get_robot_pose(robot, noisy=env.cfg.use_noisy_obs)
        pos = np.array(pos, dtype=np.float32)
        yaw = float(_yaw_from_quat(quat))
        pos_xy = pos[:2]
        pos_list.append(pos_xy)

        waypoint = robot.waypoint if robot.waypoint is not None else pos_xy
        waypoint = np.array(waypoint, dtype=np.float32)
        waypoint_vec = waypoint[:2] - pos_xy

        obj_vec = obj_pos[:2] - pos_xy
        goal_vec = goal[:2] - pos_xy

        payload_norm = float(robot.spec.payload / 150.0)
        type_one_hot = (
            np.array([1.0, 0.0], dtype=np.float32)
            if "heavy" in robot.spec.name
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        grip = 1.0 if robot.grip_active else 0.0
        force = float(env._contact_force(robot)) if hasattr(env, "_contact_force") else 0.0
        force_norm = force / max(env.cfg.contact_force_max, 1e-3)
        if env.object_belief is not None:
            belief_mu = env.object_belief.mean().astype(np.float32)
            belief_cov = env.object_belief.covariance().astype(np.float32)
        else:
            belief_mu = np.array([obj_pos[0], obj_pos[1], obj_yaw, 0.0, 0.0, 0.0], dtype=np.float32)
            belief_cov = np.eye(6, dtype=np.float32) * 1e-3
        belief_rel = np.array(
            [
                belief_mu[0] - pos_xy[0],
                belief_mu[1] - pos_xy[1],
                _wrap_angle(float(belief_mu[2] - yaw)),
            ],
            dtype=np.float32,
        )
        belief_vel = belief_mu[3:6]
        belief_cov_diag = np.diag(belief_cov).astype(np.float32)[:6]

        obs = np.concatenate(
            [
                pos_xy,
                np.array([yaw], dtype=np.float32),
                obj_vec,
                waypoint_vec,
                goal_vec,
                np.array([obj_pos[2], env.current_lift], dtype=np.float32),
                phase_vec,
                type_one_hot,
                np.array([payload_norm, grip, force_norm], dtype=np.float32),
                belief_rel,
                belief_vel,
                belief_cov_diag,
            ],
            axis=0,
        )
        obs_list.append(obs)

    return np.stack(obs_list, axis=0), np.stack(pos_list, axis=0)


def obs_dim() -> int:
    return 2 + 1 + 2 + 2 + 2 + 2 + len(PHASES) + 2 + 3 + 3 + 3 + 6


def build_global_state(env: VlmCbfEnv) -> np.ndarray:
    obj_pos, obj_quat = env._get_object_pose(noisy=env.cfg.use_noisy_obs)
    obj_pos = np.array(obj_pos, dtype=np.float32)
    obj_yaw = float(_yaw_from_quat(obj_quat))
    goal = np.array(env.goal_pos, dtype=np.float32)
    phase_vec = _phase_one_hot(env.phase.value)
    if env.object_belief is not None:
        belief_mu = env.object_belief.mean().astype(np.float32)
        belief_cov = env.object_belief.covariance().astype(np.float32)
    else:
        belief_mu = np.array([obj_pos[0], obj_pos[1], obj_yaw, 0.0, 0.0, 0.0], dtype=np.float32)
        belief_cov = np.eye(6, dtype=np.float32) * 1e-3

    forces = []
    for robot in env.robots:
        force = float(env._contact_force(robot)) if hasattr(env, "_contact_force") else 0.0
        forces.append(force / max(env.cfg.contact_force_max, 1e-3))
    force_arr = np.asarray(forces, dtype=np.float32)
    if force_arr.size == 0:
        force_stats = np.zeros(3, dtype=np.float32)
    else:
        force_stats = np.array(
            [float(np.mean(force_arr)), float(np.std(force_arr)), float(np.max(force_arr))],
            dtype=np.float32,
        )

    global_state = np.concatenate(
        [
            obj_pos,
            np.array([obj_yaw], dtype=np.float32),
            goal,
            phase_vec,
            belief_mu,
            np.diag(belief_cov).astype(np.float32)[:6],
            force_stats,
        ],
        axis=0,
    )
    return global_state


def global_state_dim() -> int:
    return 3 + 1 + 3 + len(PHASES) + 6 + 6 + 3
