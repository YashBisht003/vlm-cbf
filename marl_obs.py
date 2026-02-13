from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

from vlm_cbf_env import Phase, VlmCbfEnv


PHASE_BINS: List[List[str]] = [
    [Phase.OBSERVE.value, Phase.PLAN.value, Phase.APPROACH.value, Phase.FINE_APPROACH.value],
    [Phase.CONTACT.value, Phase.PROBE.value, Phase.CORRECT.value],
    [Phase.LIFT.value, Phase.TRANSPORT.value],
    [Phase.PLACE.value, Phase.DONE.value],
]


def _phase_one_hot4(phase: str) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    for idx, group in enumerate(PHASE_BINS):
        if phase in group:
            vec[idx] = 1.0
            return vec
    vec[0] = 1.0
    return vec


def _pad_or_trim(values: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= size:
        return arr[:size]
    out = np.zeros(size, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _robot_force_wrench(env: VlmCbfEnv, robot) -> np.ndarray:
    # PyBullet contact API gives reliable normal force; tangential/torque are not directly observed here.
    fz = float(env._contact_force(robot)) if hasattr(env, "_contact_force") else 0.0
    return np.array([0.0, 0.0, fz, 0.0, 0.0, 0.0], dtype=np.float32)


def _robot_kinematic_snapshot(env: VlmCbfEnv, robot) -> Dict[str, np.ndarray | float]:
    base_pos, base_quat = env._get_robot_pose(robot, noisy=env.cfg.use_noisy_obs)
    base_pos = np.asarray(base_pos, dtype=np.float32)
    base_yaw = float(_yaw_from_quat(base_quat))
    lin_vel, ang_vel = p.getBaseVelocity(robot.base_id, physicsClientId=env.client_id)
    lin_vel = np.asarray(lin_vel, dtype=np.float32)
    ang_vel = np.asarray(ang_vel, dtype=np.float32)
    q = []
    dq = []
    for j in robot.joint_indices:
        st = p.getJointState(robot.arm_id, int(j), physicsClientId=env.client_id)
        q.append(float(st[0]))
        dq.append(float(st[1]))
    q7 = _pad_or_trim(np.asarray(q, dtype=np.float32), 7)
    dq7 = _pad_or_trim(np.asarray(dq, dtype=np.float32), 7)
    wrench = _robot_force_wrench(env, robot)
    return {
        "pos": base_pos,
        "yaw": base_yaw,
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "q7": q7,
        "dq7": dq7,
        "wrench": wrench,
    }


def _belief_features(env: VlmCbfEnv, obj_pos: np.ndarray) -> np.ndarray:
    if env.object_belief is not None:
        mu = env.object_belief.mean().astype(np.float32)
        cov = env.object_belief.covariance().astype(np.float32)
        mu7 = _pad_or_trim(mu, 7)
        cov_diag = np.diag(cov).astype(np.float32)
    else:
        if env.object_spec is not None:
            mass = float(env.object_spec.mass)
            com = np.asarray(env.object_spec.com_offset, dtype=np.float32).reshape(3)
            dims = np.asarray(env.object_spec.dims, dtype=np.float32).reshape(3)
            lx, ly, lz = [float(max(1e-3, v)) for v in dims]
            ixx = mass * (ly * ly + lz * lz) / 12.0
            iyy = mass * (lx * lx + lz * lz) / 12.0
            izz = mass * (lx * lx + ly * ly) / 12.0
            mu7 = np.array([mass, com[0], com[1], com[2], ixx, iyy, izz], dtype=np.float32)
            cov_diag = np.array([100.0, 0.05, 0.05, 0.05, 10.0, 10.0, 10.0], dtype=np.float32)
        else:
            mu7 = np.array([100.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float32)
            cov_diag = np.array([100.0, 0.05, 0.05, 0.05, 10.0, 10.0, 10.0], dtype=np.float32)
    belief12 = np.concatenate([mu7, cov_diag[:5]], axis=0)
    return belief12.astype(np.float32)


def _safety_features(env: VlmCbfEnv, robot_name: str) -> np.ndarray:
    metrics = env.cbf_step_metrics.get(robot_name, {})
    sep = float(metrics.get("sep_barrier_norm", 1.0))
    speed = float(metrics.get("speed_barrier_norm", 1.0))
    force = float(metrics.get("force_barrier_norm", 1.0))
    neural = float(np.tanh(float(metrics.get("neural_barrier", 1.0))))
    minimum = float(min(sep, speed, force, neural))
    return np.array([sep, speed, force, neural, minimum], dtype=np.float32)


def _neighbor_block(me: Dict[str, np.ndarray | float], other: Dict[str, np.ndarray | float]) -> np.ndarray:
    me_pos = np.asarray(me["pos"], dtype=np.float32)
    ot_pos = np.asarray(other["pos"], dtype=np.float32)
    me_yaw = float(me["yaw"])
    ot_yaw = float(other["yaw"])
    me_lv = np.asarray(me["lin_vel"], dtype=np.float32)
    ot_lv = np.asarray(other["lin_vel"], dtype=np.float32)
    me_wz = float(np.asarray(me["ang_vel"], dtype=np.float32)[2])
    ot_wz = float(np.asarray(other["ang_vel"], dtype=np.float32)[2])
    ot_wrench = np.asarray(other["wrench"], dtype=np.float32)
    rel_pose6 = np.array(
        [
            ot_pos[0] - me_pos[0],
            ot_pos[1] - me_pos[1],
            ot_pos[2] - me_pos[2],
            0.0,
            0.0,
            _wrap_angle(ot_yaw - me_yaw),
        ],
        dtype=np.float32,
    )
    rel_vel3 = np.array([ot_lv[0] - me_lv[0], ot_lv[1] - me_lv[1], ot_wz - me_wz], dtype=np.float32)
    force3 = np.array([ot_wrench[0], ot_wrench[1], ot_wrench[2]], dtype=np.float32)
    return np.concatenate([rel_pose6, rel_vel3, force3], axis=0).astype(np.float32)


def _ego20(snapshot: Dict[str, np.ndarray | float]) -> np.ndarray:
    pos = np.asarray(snapshot["pos"], dtype=np.float32)
    yaw = float(snapshot["yaw"])
    lv = np.asarray(snapshot["lin_vel"], dtype=np.float32)
    wz = float(np.asarray(snapshot["ang_vel"], dtype=np.float32)[2])
    q7 = np.asarray(snapshot["q7"], dtype=np.float32)
    dq7 = np.asarray(snapshot["dq7"], dtype=np.float32)
    ego = np.concatenate(
        [
            np.array([pos[0], pos[1], yaw], dtype=np.float32),
            np.array([lv[0], lv[1], wz], dtype=np.float32),
            q7,
            dq7,
        ],
        axis=0,
    )
    return ego.astype(np.float32)


def _object_relative12(
    snapshot: Dict[str, np.ndarray | float],
    obj_pos: np.ndarray,
    obj_euler: np.ndarray,
    obj_lin_vel: np.ndarray,
    obj_ang_vel: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(snapshot["pos"], dtype=np.float32)
    yaw = float(snapshot["yaw"])
    lv = np.asarray(snapshot["lin_vel"], dtype=np.float32)
    wz = float(np.asarray(snapshot["ang_vel"], dtype=np.float32)[2])
    rel_pose6 = np.array(
        [
            obj_pos[0] - pos[0],
            obj_pos[1] - pos[1],
            obj_pos[2] - pos[2],
            obj_euler[0],
            obj_euler[1],
            _wrap_angle(float(obj_euler[2] - yaw)),
        ],
        dtype=np.float32,
    )
    rel_vel6 = np.array(
        [
            obj_lin_vel[0] - lv[0],
            obj_lin_vel[1] - lv[1],
            obj_lin_vel[2],
            obj_ang_vel[0],
            obj_ang_vel[1],
            obj_ang_vel[2] - wz,
        ],
        dtype=np.float32,
    )
    return np.concatenate([rel_pose6, rel_vel6], axis=0).astype(np.float32)


def _goal6(snapshot: Dict[str, np.ndarray | float], waypoint3: np.ndarray) -> np.ndarray:
    pos = np.asarray(snapshot["pos"], dtype=np.float32)
    direction = waypoint3 - pos
    n = float(np.linalg.norm(direction))
    if n > 1e-6:
        direction = direction / n
    else:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([waypoint3.astype(np.float32), direction.astype(np.float32)], axis=0)


def _cbf_team_features(env: VlmCbfEnv) -> np.ndarray:
    if not env.cbf_step_metrics:
        return np.ones(5, dtype=np.float32)
    sep = min(float(v.get("sep_barrier_norm", 1.0)) for v in env.cbf_step_metrics.values())
    speed = min(float(v.get("speed_barrier_norm", 1.0)) for v in env.cbf_step_metrics.values())
    force = min(float(v.get("force_barrier_norm", 1.0)) for v in env.cbf_step_metrics.values())
    neural = min(float(np.tanh(float(v.get("neural_barrier", 1.0)))) for v in env.cbf_step_metrics.values())
    minimum = min(sep, speed, force, neural)
    return np.array([sep, speed, force, neural, minimum], dtype=np.float32)


def _phase_one_hot_legacy(phase: str) -> np.ndarray:
    # Kept for compatibility in older analysis scripts.
    vec = np.zeros(11, dtype=np.float32)
    names = [
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
    if phase in names:
        vec[names.index(phase)] = 1.0
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
    obj_euler = np.asarray(p.getEulerFromQuaternion(obj_quat), dtype=np.float32)
    obj_lin_vel, obj_ang_vel = p.getBaseVelocity(env.object_id, physicsClientId=env.client_id)
    obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32)
    obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32)
    goal = np.array(env.goal_pos, dtype=np.float32)
    phase_vec = _phase_one_hot4(env.phase.value)
    belief12 = _belief_features(env, obj_pos)

    obs_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    snapshots: Dict[str, Dict[str, np.ndarray | float]] = {}
    for robot in env.robots:
        snapshots[robot.spec.name] = _robot_kinematic_snapshot(env, robot)

    for robot in env.robots:
        me = snapshots[robot.spec.name]
        pos = np.asarray(me["pos"], dtype=np.float32)
        pos_list.append(pos[:2])

        waypoint_xy = np.asarray(robot.waypoint[:2], dtype=np.float32) if robot.waypoint is not None else pos[:2]
        waypoint3 = np.array([waypoint_xy[0], waypoint_xy[1], obj_pos[2]], dtype=np.float32)
        ego20 = _ego20(me)
        obj_rel12 = _object_relative12(me, obj_pos, obj_euler, obj_lin_vel, obj_ang_vel)
        force6 = np.asarray(me["wrench"], dtype=np.float32)
        goal6 = _goal6(me, waypoint3)

        other_names = [r.spec.name for r in env.robots if r.spec.name != robot.spec.name]
        other_names = sorted(
            other_names,
            key=lambda name: float(np.linalg.norm(np.asarray(snapshots[name]["pos"], dtype=np.float32)[:2] - pos[:2])),
        )
        neighbor_vecs: List[np.ndarray] = []
        for name in other_names[:3]:
            neighbor_vecs.append(_neighbor_block(me, snapshots[name]))
        while len(neighbor_vecs) < 3:
            neighbor_vecs.append(np.zeros(12, dtype=np.float32))
        neighbors36 = np.concatenate(neighbor_vecs, axis=0).astype(np.float32)

        safety5 = _safety_features(env, robot.spec.name)
        obs = np.concatenate([ego20, obj_rel12, force6, goal6, neighbors36, belief12, safety5, phase_vec], axis=0)
        obs_list.append(obs)

    return np.stack(obs_list, axis=0), np.stack(pos_list, axis=0)


def obs_dim() -> int:
    # 20 + 12 + 6 + 6 + 36 + 12 + 5 + 4
    return 101


def build_global_state(env: VlmCbfEnv) -> np.ndarray:
    obj_pos, obj_quat = env._get_object_pose(noisy=env.cfg.use_noisy_obs)
    obj_pos = np.array(obj_pos, dtype=np.float32)
    obj_euler = np.asarray(p.getEulerFromQuaternion(obj_quat), dtype=np.float32)
    obj_lin_vel, obj_ang_vel = p.getBaseVelocity(env.object_id, physicsClientId=env.client_id)
    obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32)
    obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32)
    goal = np.array(env.goal_pos, dtype=np.float32)
    phase4 = _phase_one_hot4(env.phase.value)
    belief12 = _belief_features(env, obj_pos)
    obj12 = np.concatenate([obj_pos, obj_euler, obj_lin_vel, obj_ang_vel], axis=0).astype(np.float32)

    snapshots = {r.spec.name: _robot_kinematic_snapshot(env, r) for r in env.robots}
    robot_blocks: List[np.ndarray] = []
    for robot in env.robots:
        snap = snapshots[robot.spec.name]
        pos = np.asarray(snap["pos"], dtype=np.float32)
        yaw = float(snap["yaw"])
        lv = np.asarray(snap["lin_vel"], dtype=np.float32)
        wz = float(np.asarray(snap["ang_vel"], dtype=np.float32)[2])
        force_norm = float(np.asarray(snap["wrench"], dtype=np.float32)[2] / max(1e-6, env.cfg.contact_force_max))
        payload_norm = float(robot.spec.payload / 150.0)
        grip = 1.0 if robot.grip_active else 0.0
        block = np.array([pos[0], pos[1], yaw, lv[0], lv[1], wz, force_norm, payload_norm, grip], dtype=np.float32)
        robot_blocks.append(block)
    robots36 = np.concatenate(robot_blocks, axis=0).astype(np.float32)
    cbf5 = _cbf_team_features(env)
    global_state = np.concatenate([obj12, goal, phase4, belief12, robots36, cbf5], axis=0).astype(np.float32)
    return global_state


def global_state_dim() -> int:
    # object12 + goal3 + phase4 + belief12 + robots36 + cbf5
    return 72
