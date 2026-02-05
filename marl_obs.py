from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pybullet as p

from vlm_cbf_env import Phase, VlmCbfEnv


PHASES: List[str] = [
    Phase.OBSERVE.value,
    Phase.PLAN.value,
    Phase.APPROACH.value,
    Phase.CONTACT.value,
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


def build_observation(env: VlmCbfEnv) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        obs: (N, obs_dim)
        positions: (N, 2) xy positions for message passing
    """
    obj_pos, obj_quat = env._get_object_pose(noisy=env.cfg.use_noisy_obs)
    obj_pos = np.array(obj_pos, dtype=np.float32)
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
        belief_unc = float(env.object_belief.uncertainty()) if env.object_belief is not None else 0.0

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
                np.array([payload_norm, grip, force_norm, belief_unc], dtype=np.float32),
            ],
            axis=0,
        )
        obs_list.append(obs)

    return np.stack(obs_list, axis=0), np.stack(pos_list, axis=0)


def obs_dim() -> int:
    return 2 + 1 + 2 + 2 + 2 + 2 + len(PHASES) + 2 + 4
