from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import json
import math
import statistics
from pathlib import Path
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "pybullet is required. Install it with: pip install -r requirements.txt"
    ) from exc

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional fallback
    linear_sum_assignment = None

from belief_ekf import BeliefEKF
from cbf_qp import solve_cbf_qp
from p2p_udp import UdpPeerBus


class Phase(Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    APPROACH = "approach"
    CONTACT = "contact"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    DONE = "done"


@dataclass
class TaskConfig:
    gui: bool = True
    control_hz: int = 100
    physics_hz: int = 240
    random_seed: Optional[int] = None

    spawn_xy_min: Tuple[float, float] = (-1.0, -1.0)
    spawn_xy_max: Tuple[float, float] = (1.0, 1.0)
    goal_xy: Tuple[float, float] = (2.5, 0.0)
    podium_size: Tuple[float, float, float] = (0.8, 0.8, 0.3)

    object_mass_range: Tuple[float, float] = (50.0, 280.0)
    friction_range: Tuple[float, float] = (0.4, 0.9)
    com_offset_frac: float = 0.4
    standoff: float = 0.6

    speed_limit: float = 0.25
    separation_min: float = 0.5
    contact_force_max: float = 140.0

    contact_dist: float = 0.08
    lift_height: float = 0.25
    lift_speed: float = 0.08

    use_ik: bool = True
    kinematic_base: bool = False
    carry_mode: str = "auto"  # auto, constraint, or kinematic
    constraint_fallback_time: float = 2.0
    constraint_force_scale: float = 1.5
    vacuum_attach_dist: float = 0.18
    vacuum_break_dist: float = 0.30
    vacuum_force_margin: float = 1.05

    yaw_rate_max: float = 1.2
    diff_heading_k: float = 2.5

    sensor_pos_noise: float = 0.0
    sensor_yaw_noise: float = 0.0
    sensor_force_noise: float = 0.0
    use_noisy_obs: bool = False
    use_noisy_control: bool = False
    use_noisy_plan: bool = False

    use_cbf: bool = True
    cbf_alpha: float = 2.0
    cbf_kappa: float = 2.0
    cbf_risk_scale: float = 1.0
    cbf_slack_weight: float = 100.0
    cbf_slack_max: float = 3.0
    use_ekf: bool = True
    ekf_q_pos: float = 1e-3
    ekf_q_vel: float = 1e-2
    ekf_r_meas: float = 5e-3

    use_udp_phase: bool = True
    use_udp_neighbor_state: bool = True
    udp_host: str = "127.0.0.1"
    udp_base_port: int = 39000
    udp_state_ttl_s: float = 0.25
    phase_consensus_window_s: float = 0.10
    phase_approach_dist: float = 0.25
    phase_approach_timeout_s: float = 20.0
    phase_approach_min_ready: int = 2
    phase_contact_force_threshold: float = 5.0
    phase_lift_stability_s: float = 1.0
    phase_lift_vertical_speed_max: float = 0.05

    vlm_json_path: Optional[str] = None
    vlm_confidence_threshold: float = 0.5

    heavy_urdf: Optional[str] = None
    agile_urdf: Optional[str] = None
    heavy_ee_link: Optional[int] = None
    agile_ee_link: Optional[int] = None
    heavy_ee_link_name: Optional[str] = None
    agile_ee_link_name: Optional[str] = None

    base_omni_urdf: Optional[str] = None
    base_diff_urdf: Optional[str] = None
    wheel_radius: float = 0.08
    wheel_base: float = 0.36
    track_width: float = 0.28
    base_mass: float = 50.0
    base_ground_friction: float = 0.05
    wheel_friction_omni: float = 1.1
    wheel_friction_diff: float = 1.3
    wheel_motor_force_omni: float = 1200.0
    wheel_motor_force_diff: float = 900.0
    base_drive_mode: str = "velocity"  # velocity or wheel

    object_size_ratio: Tuple[float, float] = (1.0, 3.0)
    robot_size_mode: str = "base"  # base or full
    podium_auto_scale: bool = True
    podium_margin: float = 0.15
    podium_height_range: Tuple[float, float] = (0.3, 0.6)


@dataclass
class RobotSpec:
    name: str
    urdf: str
    dof: int
    base_type: str  # omni or diff
    payload: float
    color: Tuple[float, float, float, float]
    base_z: float
    ee_link_override: Optional[int] = None
    ee_link_name_override: Optional[str] = None


@dataclass
class RobotInstance:
    spec: RobotSpec
    base_id: int
    arm_id: int
    wheel_joints: List[int]
    joint_indices: List[int]
    end_effector_link: int
    base_constraint_id: Optional[int] = None
    grip_active: bool = False
    constraint_id: Optional[int] = None
    waypoint: Optional[np.ndarray] = None
    waypoint_offset: Optional[np.ndarray] = None


@dataclass
class ObjectSpec:
    shape: str
    dims: Tuple[float, float, float]
    mass: float
    friction: float
    com_offset: Tuple[float, float, float]


@dataclass
class RobotAction:
    base_vel: Tuple[float, float, float]  # vx, vy, yaw_rate (world frame)
    joint_positions: Optional[List[float]] = None
    grip: bool = False


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _yaw_from_quat(quat: Tuple[float, float, float, float]) -> float:
    _, _, yaw = p.getEulerFromQuaternion(quat)
    return yaw


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler((0.0, 0.0, yaw))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class VlmCbfEnv:
    def __init__(self, config: Optional[TaskConfig] = None) -> None:
        self.cfg = config or TaskConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.client_id = self._connect()
        self._setup_time()
        self._reset_state()

    def _connect(self) -> int:
        mode = p.GUI if self.cfg.gui else p.DIRECT
        client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        return client_id

    def _setup_time(self) -> None:
        self.control_dt = 1.0 / float(self.cfg.control_hz)
        self.physics_dt = 1.0 / float(self.cfg.physics_hz)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)

    def _reset_state(self) -> None:
        if hasattr(self, "udp_bus") and self.udp_bus is not None:
            self.udp_bus.close()
        self.robots: List[RobotInstance] = []
        self.object_id: Optional[int] = None
        self.object_spec: Optional[ObjectSpec] = None
        self.phase = Phase.OBSERVE
        self.phase_time = 0.0
        self.sim_time = 0.0
        self.goal_pos = np.array([self.cfg.goal_xy[0], self.cfg.goal_xy[1], 0.0])
        self.carrying = False
        self.current_lift = 0.0
        self.goal_waypoints: Optional[List[np.ndarray]] = None
        self.violations: Dict[str, int] = {"speed": 0, "separation": 0, "force": 0}
        self.cbf_stats: Dict[str, int] = {"calls": 0, "modified": 0, "fallback": 0, "force_stop": 0}
        self.grasp_stats: Dict[str, int] = {
            "attach_attempts": 0,
            "attach_success": 0,
            "detach_events": 0,
            "overload_drop": 0,
            "stretch_drop": 0,
        }
        if self.cfg.carry_mode == "auto":
            self.active_carry_mode = "constraint"
        else:
            self.active_carry_mode = self.cfg.carry_mode
        self._vlm_json_cache: Optional[dict] = None
        self.robot_size_ref: Optional[float] = None
        self.podium_id: Optional[int] = None
        self.podium_height: float = self.cfg.podium_size[2]
        self.last_safe_vel: Dict[int, np.ndarray] = {}
        self.object_belief: Optional[BeliefEKF] = None
        self.object_start_height: float = 0.0
        self._lift_stable_since: Optional[float] = None
        self.udp_bus: Optional[UdpPeerBus] = None
        self.peer_packets: Dict[str, dict] = {}
        self.phase_sync_delays_ms: List[float] = []
        self.phase_sync_last_delay_ms: float = 0.0

    def reset(self) -> Dict:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        self._reset_state()
        self._spawn_scene()
        self._spawn_robots()
        self._spawn_object()
        self._spawn_podium()
        if self.cfg.use_ekf:
            self.object_belief = BeliefEKF(
                dt=self.control_dt,
                q_pos=self.cfg.ekf_q_pos,
                q_vel=self.cfg.ekf_q_vel,
                r_meas=self.cfg.ekf_r_meas,
            )
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
            self.object_belief.update((float(obj_pos[0]), float(obj_pos[1])))
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        self.object_start_height = float(obj_pos[2])
        if self.cfg.use_udp_phase or self.cfg.use_udp_neighbor_state:
            self.udp_bus = UdpPeerBus(
                robot_names=[robot.spec.name for robot in self.robots],
                host=self.cfg.udp_host,
                base_port=self.cfg.udp_base_port,
            )
            self._udp_broadcast_local_state(self._phase_local_ready_map())
            self._udp_poll()
        self.phase = Phase.PLAN
        return self._get_obs()

    def close(self) -> None:
        if self.udp_bus is not None:
            self.udp_bus.close()
            self.udp_bus = None
        p.disconnect(physicsClientId=self.client_id)

    def step(self, actions: Optional[Dict[int, RobotAction]] = None) -> Tuple[Dict, Dict]:
        self._udp_poll()
        if actions is None:
            actions = self._simple_policy()
        self._apply_actions(actions)
        self._step_physics()
        local_ready = self._phase_local_ready_map()
        self._udp_broadcast_local_state(local_ready)
        self._udp_poll()
        self._update_phase()
        obs = self._get_obs()
        phase_sync_mean = (
            float(statistics.fmean(self.phase_sync_delays_ms))
            if self.phase_sync_delays_ms
            else 0.0
        )
        info = {
            "phase": self.phase.value,
            "time": self.sim_time,
            "violations": dict(self.violations),
            "carry_mode": self.active_carry_mode,
            "cbf": dict(self.cbf_stats),
            "grasp": dict(self.grasp_stats),
            "phase_sync": {
                "last_delay_ms": float(self.phase_sync_last_delay_ms),
                "mean_delay_ms": phase_sync_mean,
                "events": len(self.phase_sync_delays_ms),
            },
        }
        return obs, info

    def _spawn_scene(self) -> None:
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

    def _spawn_podium(self) -> None:
        if self.cfg.podium_auto_scale and self.object_spec is not None:
            length, width, _height = self.object_spec.dims
            margin = max(0.0, float(self.cfg.podium_margin))
            pod_x = length * (1.0 + margin)
            pod_y = width * (1.0 + margin)
            if self.robot_size_ref is not None:
                target_height = 0.35 * self.robot_size_ref
            else:
                target_height = self.cfg.podium_size[2]
            pod_h = _clamp(target_height, self.cfg.podium_height_range[0], self.cfg.podium_height_range[1])
            pod_half = np.array([pod_x * 0.5, pod_y * 0.5, pod_h * 0.5])
        else:
            pod_half = np.array(self.cfg.podium_size) * 0.5

        self.podium_height = float(pod_half[2] * 2.0)
        podium_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=pod_half, physicsClientId=self.client_id
        )
        podium_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=pod_half,
            rgbaColor=(0.3, 0.3, 0.3, 1.0),
            physicsClientId=self.client_id,
        )
        self.podium_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=podium_shape,
            baseVisualShapeIndex=podium_vis,
            basePosition=[self.goal_pos[0], self.goal_pos[1], pod_half[2]],
            physicsClientId=self.client_id,
        )

    def _robot_specs(self) -> List[RobotSpec]:
        heavy_color = (0.85, 0.25, 0.25, 1.0)
        agile_color = (0.25, 0.45, 0.85, 1.0)
        heavy_urdf = self.cfg.heavy_urdf or "kuka_iiwa/model.urdf"
        agile_urdf = self.cfg.agile_urdf or "xarm/xarm6_robot.urdf"
        return [
            RobotSpec(
                name="heavy_1",
                urdf=heavy_urdf,
                dof=7,
                base_type="omni",
                payload=150.0,
                color=heavy_color,
                base_z=0.02,
                ee_link_override=self.cfg.heavy_ee_link,
                ee_link_name_override=self.cfg.heavy_ee_link_name,
            ),
            RobotSpec(
                name="heavy_2",
                urdf=heavy_urdf,
                dof=7,
                base_type="omni",
                payload=150.0,
                color=heavy_color,
                base_z=0.02,
                ee_link_override=self.cfg.heavy_ee_link,
                ee_link_name_override=self.cfg.heavy_ee_link_name,
            ),
            RobotSpec(
                name="agile_1",
                urdf=agile_urdf,
                dof=6,
                base_type="diff",
                payload=50.0,
                color=agile_color,
                base_z=0.02,
                ee_link_override=self.cfg.agile_ee_link,
                ee_link_name_override=self.cfg.agile_ee_link_name,
            ),
            RobotSpec(
                name="agile_2",
                urdf=agile_urdf,
                dof=6,
                base_type="diff",
                payload=50.0,
                color=agile_color,
                base_z=0.02,
                ee_link_override=self.cfg.agile_ee_link,
                ee_link_name_override=self.cfg.agile_ee_link_name,
            ),
        ]

    def _base_urdf_path(self, base_type: str) -> str:
        if base_type == "omni":
            if self.cfg.base_omni_urdf:
                return self.cfg.base_omni_urdf
            return str(Path(__file__).parent / "assets" / "base_omni.urdf")
        if self.cfg.base_diff_urdf:
            return self.cfg.base_diff_urdf
        return str(Path(__file__).parent / "assets" / "base_diff.urdf")

    def _wheel_joint_indices(self, base_id: int, base_type: str) -> List[int]:
        if base_type == "omni":
            names = ["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"]
        else:
            names = ["left_wheel_joint", "right_wheel_joint"]
        joint_indices: List[int] = []
        num_joints = p.getNumJoints(base_id, physicsClientId=self.client_id)
        for j in range(num_joints):
            joint_name = p.getJointInfo(base_id, j, physicsClientId=self.client_id)[1]
            if isinstance(joint_name, (bytes, bytearray)):
                joint_name = joint_name.decode("utf-8", errors="ignore")
            if str(joint_name) in names:
                joint_indices.append(j)
        return joint_indices

    def _load_arm(
        self, spec: RobotSpec, base_pos: List[float], yaw: float
    ) -> Tuple[int, List[int], int]:
        arm_id = p.loadURDF(
            spec.urdf,
            basePosition=base_pos,
            baseOrientation=_quat_from_yaw(yaw),
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        num_joints = p.getNumJoints(arm_id, physicsClientId=self.client_id)
        joint_indices = [
            j
            for j in range(num_joints)
            if p.getJointInfo(arm_id, j, physicsClientId=self.client_id)[2]
            == p.JOINT_REVOLUTE
        ]
        if spec.ee_link_override is not None and 0 <= spec.ee_link_override < num_joints:
            end_effector_link = spec.ee_link_override
        else:
            name_candidates = []
            if spec.ee_link_name_override:
                name_candidates.append(spec.ee_link_name_override)
            name_candidates.extend(["tool0", "tool", "flange", "tcp", "ee", "end_effector"])
            end_effector_link = self._find_link_index_by_name(arm_id, name_candidates)
            if end_effector_link is None:
                end_effector_link = num_joints - 1 if num_joints > 0 else -1
        for link_idx in range(-1, num_joints):
            p.changeVisualShape(
                arm_id,
                link_idx,
                rgbaColor=spec.color,
                physicsClientId=self.client_id,
            )
        return arm_id, joint_indices, end_effector_link

    def _spawn_robots(self) -> None:
        specs = self._robot_specs()
        ring_r = 1.2
        for idx, spec in enumerate(specs):
            angle = (2.0 * math.pi * idx) / len(specs)
            x = ring_r * math.cos(angle)
            y = ring_r * math.sin(angle)
            yaw = angle + math.pi
            base_pos = [x, y, 0.0]
            base_id = p.loadURDF(
                self._base_urdf_path(spec.base_type),
                basePosition=base_pos,
                baseOrientation=_quat_from_yaw(yaw),
                useFixedBase=False,
                physicsClientId=self.client_id,
            )
            p.changeDynamics(
                base_id,
                -1,
                mass=self.cfg.base_mass,
                lateralFriction=self.cfg.base_ground_friction,
                rollingFriction=0.0,
                spinningFriction=0.0,
                physicsClientId=self.client_id,
            )
            for link_idx in range(-1, p.getNumJoints(base_id, physicsClientId=self.client_id)):
                p.changeVisualShape(
                    base_id,
                    link_idx,
                    rgbaColor=spec.color,
                    physicsClientId=self.client_id,
                )
            wheel_joints = self._wheel_joint_indices(base_id, spec.base_type)
            wheel_friction = (
                self.cfg.wheel_friction_diff
                if spec.base_type == "diff"
                else self.cfg.wheel_friction_omni
            )
            for joint_idx in wheel_joints:
                p.changeDynamics(
                    base_id,
                    joint_idx,
                    lateralFriction=wheel_friction,
                    rollingFriction=0.001,
                    spinningFriction=0.001,
                    physicsClientId=self.client_id,
                )
                p.setJointMotorControl2(
                    base_id,
                    joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=0.0,
                    physicsClientId=self.client_id,
                )
            base_aabb = p.getAABB(base_id, -1, physicsClientId=self.client_id)
            base_top = base_aabb[1][2]
            arm_pos = [x, y, base_top + spec.base_z]
            arm_id, joint_indices, end_effector_link = self._load_arm(spec, arm_pos, yaw)
            base_constraint_id = p.createConstraint(
                base_id,
                -1,
                arm_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, base_top],
                [0, 0, 0],
                physicsClientId=self.client_id,
            )
            p.changeConstraint(base_constraint_id, maxForce=1e5)
            for base_link in range(-1, p.getNumJoints(base_id, physicsClientId=self.client_id)):
                for arm_link in range(-1, p.getNumJoints(arm_id, physicsClientId=self.client_id)):
                    p.setCollisionFilterPair(
                        base_id,
                        arm_id,
                        base_link,
                        arm_link,
                        enableCollision=0,
                        physicsClientId=self.client_id,
                    )

            size_xy = self._compute_robot_size_xy(base_id, arm_id)
            if size_xy is not None:
                if self.robot_size_ref is None or size_xy > self.robot_size_ref:
                    self.robot_size_ref = size_xy
            self.robots.append(
                RobotInstance(
                    spec=spec,
                    base_id=base_id,
                    arm_id=arm_id,
                    wheel_joints=wheel_joints,
                    joint_indices=joint_indices,
                    end_effector_link=end_effector_link,
                    base_constraint_id=base_constraint_id,
                )
            )
            self.last_safe_vel[base_id] = np.zeros(2, dtype=np.float32)

    def _find_link_index_by_name(self, body_id: int, candidates) -> Optional[int]:
        if not candidates:
            return None
        if isinstance(candidates, str):
            candidates = [candidates]
        normalized = [c.lower() for c in candidates if c]
        if not normalized:
            return None
        num_joints = p.getNumJoints(body_id, physicsClientId=self.client_id)
        for j in range(num_joints):
            link_name = p.getJointInfo(body_id, j, physicsClientId=self.client_id)[12]
            if isinstance(link_name, (bytes, bytearray)):
                link_name = link_name.decode("utf-8", errors="ignore")
            link_lower = str(link_name).lower()
            if link_lower in normalized:
                return j
        for j in range(num_joints):
            link_name = p.getJointInfo(body_id, j, physicsClientId=self.client_id)[12]
            if isinstance(link_name, (bytes, bytearray)):
                link_name = link_name.decode("utf-8", errors="ignore")
            link_lower = str(link_name).lower()
            for key in normalized:
                if key in link_lower:
                    return j
        return None

    def _compute_robot_size_xy(self, base_id: int, arm_id: int) -> Optional[float]:
        try:
            base_min, base_max = p.getAABB(base_id, -1, physicsClientId=self.client_id)
            aabb_min = list(base_min)
            aabb_max = list(base_max)
            if self.cfg.robot_size_mode == "full":
                arm_min, arm_max = p.getAABB(arm_id, -1, physicsClientId=self.client_id)
                aabb_min = [min(aabb_min[i], arm_min[i]) for i in range(3)]
                aabb_max = [max(aabb_max[i], arm_max[i]) for i in range(3)]
                for link_idx in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
                    link_min, link_max = p.getAABB(arm_id, link_idx, physicsClientId=self.client_id)
                    aabb_min = [min(aabb_min[i], link_min[i]) for i in range(3)]
                    aabb_max = [max(aabb_max[i], link_max[i]) for i in range(3)]
        except Exception:
            return None
        size_x = float(aabb_max[0] - aabb_min[0])
        size_y = float(aabb_max[1] - aabb_min[1])
        return max(size_x, size_y, 1e-3)

    def _spawn_object(self) -> None:
        shape = self.rng.choice(["cuboid", "l_shape", "t_shape"])
        dims = self._sample_dims(shape)
        mass = float(self.rng.uniform(*self.cfg.object_mass_range))
        friction = float(self.rng.uniform(*self.cfg.friction_range))
        com_offset = self._sample_com_offset(dims)

        obj_spec = ObjectSpec(
            shape=shape, dims=dims, mass=mass, friction=friction, com_offset=com_offset
        )
        self.object_spec = obj_spec

        spawn_x = float(self.rng.uniform(self.cfg.spawn_xy_min[0], self.cfg.spawn_xy_max[0]))
        spawn_y = float(self.rng.uniform(self.cfg.spawn_xy_min[1], self.cfg.spawn_xy_max[1]))
        spawn_z = dims[2] * 0.5 + 0.02

        self.object_id = self._create_object_body(
            obj_spec, base_pos=(spawn_x, spawn_y, spawn_z)
        )

    def _sample_dims(self, shape: str) -> Tuple[float, float, float]:
        base = self.robot_size_ref
        if base is None:
            if shape == "cuboid":
                return (
                    float(self.rng.uniform(0.4, 0.9)),
                    float(self.rng.uniform(0.3, 0.7)),
                    float(self.rng.uniform(0.2, 0.5)),
                )
            if shape == "l_shape":
                return (
                    float(self.rng.uniform(0.6, 1.0)),
                    float(self.rng.uniform(0.3, 0.6)),
                    float(self.rng.uniform(0.2, 0.4)),
                )
            return (
                float(self.rng.uniform(0.6, 1.0)),
                float(self.rng.uniform(0.3, 0.6)),
                float(self.rng.uniform(0.2, 0.4)),
            )

        ratio = float(self.rng.uniform(*self.cfg.object_size_ratio))
        length = base * ratio * float(self.rng.uniform(0.9, 1.1))
        if shape == "cuboid":
            width = base * ratio * float(self.rng.uniform(0.6, 0.9))
        else:
            width = base * ratio * float(self.rng.uniform(0.4, 0.8))
        height = base * float(self.rng.uniform(0.25, 0.45))
        length = max(length, 0.3)
        width = max(width, 0.2)
        height = max(height, 0.15)
        return (float(length), float(width), float(height))

    def _sample_com_offset(self, dims: Tuple[float, float, float]) -> Tuple[float, float, float]:
        frac = self.cfg.com_offset_frac
        return (
            float(self.rng.uniform(-frac, frac) * dims[0]),
            float(self.rng.uniform(-frac, frac) * dims[1]),
            float(self.rng.uniform(-0.2 * frac, 0.2 * frac) * dims[2]),
        )

    def _create_object_body(self, obj: ObjectSpec, base_pos: Tuple[float, float, float]) -> int:
        if obj.shape == "cuboid":
            half_ext = [d * 0.5 for d in obj.dims]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=self.client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=(0.7, 0.7, 0.2, 1.0), physicsClientId=self.client_id)
            body = p.createMultiBody(
                baseMass=obj.mass,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=base_pos,
                baseInertialFramePosition=obj.com_offset,
                physicsClientId=self.client_id,
            )
        else:
            body = self._create_compound_body(obj, base_pos)

        p.changeDynamics(body, -1, lateralFriction=obj.friction, physicsClientId=self.client_id)
        return body

    def _create_compound_body(self, obj: ObjectSpec, base_pos: Tuple[float, float, float]) -> int:
        length, width, height = obj.dims
        thickness = height * 0.6
        if obj.shape == "l_shape":
            shapes = [
                (p.GEOM_BOX, [length * 0.5, thickness * 0.5, height * 0.5], [0, 0, 0]),
                (p.GEOM_BOX, [thickness * 0.5, width * 0.5, height * 0.5], [-(length * 0.5) + (thickness * 0.5), width * 0.5 - thickness * 0.5, 0]),
            ]
        else:
            shapes = [
                (p.GEOM_BOX, [length * 0.5, thickness * 0.5, height * 0.5], [0, 0, 0]),
                (p.GEOM_BOX, [thickness * 0.5, width * 0.5, height * 0.5], [0, 0, 0]),
            ]

        shape_types = [s[0] for s in shapes]
        half_extents = [s[1] for s in shapes]
        positions = [s[2] for s in shapes]
        col = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            physicsClientId=self.client_id,
        )
        vis = p.createVisualShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            visualFramePositions=positions,
            rgbaColors=[(0.7, 0.5, 0.2, 1.0)] * len(shapes),
            physicsClientId=self.client_id,
        )
        body = p.createMultiBody(
            baseMass=obj.mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=base_pos,
            baseInertialFramePosition=obj.com_offset,
            physicsClientId=self.client_id,
        )
        return body

    def _get_obs(self) -> Dict:
        obj_pos, obj_quat = self._get_object_pose(noisy=self.cfg.use_noisy_obs)
        robots_state = []
        for robot in self.robots:
            pos, quat = self._get_robot_pose(robot, noisy=self.cfg.use_noisy_obs)
            robots_state.append(
                {
                    "name": robot.spec.name,
                    "pos": pos,
                    "yaw": _yaw_from_quat(quat),
                    "grip": robot.grip_active,
                }
            )
        return {
            "phase": self.phase.value,
            "object_pos": obj_pos,
            "object_quat": obj_quat,
            "robots": robots_state,
        }

    def _get_object_pose(self, noisy: bool = False) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pos, quat = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        if not noisy:
            return pos, quat
        pos = self._apply_pos_noise(np.array(pos))
        quat = self._apply_yaw_noise(quat)
        return (float(pos[0]), float(pos[1]), float(pos[2])), quat

    def _get_robot_pose(self, robot: RobotInstance, noisy: bool = False) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pos, quat = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
        if not noisy:
            return pos, quat
        pos = self._apply_pos_noise(np.array(pos))
        quat = self._apply_yaw_noise(quat)
        return (float(pos[0]), float(pos[1]), float(pos[2])), quat

    def _apply_pos_noise(self, pos: np.ndarray) -> np.ndarray:
        if self.cfg.sensor_pos_noise <= 0.0:
            return pos
        noise = self.rng.normal(0.0, self.cfg.sensor_pos_noise, size=3)
        return pos + noise

    def _apply_yaw_noise(self, quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if self.cfg.sensor_yaw_noise <= 0.0:
            return quat
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw += float(self.rng.normal(0.0, self.cfg.sensor_yaw_noise))
        return p.getQuaternionFromEuler((roll, pitch, yaw))

    def _simple_policy(self) -> Dict[int, RobotAction]:
        actions: Dict[int, RobotAction] = {}
        for idx, robot in enumerate(self.robots):
            base_pos, base_quat = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
            base_pos = np.array(base_pos)

            target = None
            grip = False
            if self.phase in (Phase.APPROACH, Phase.PLAN):
                target = robot.waypoint
            elif self.phase == Phase.CONTACT:
                target = self._object_contact_target(base_pos)
                grip = True
            elif self.phase == Phase.LIFT:
                target = base_pos
                grip = True
            elif self.phase == Phase.TRANSPORT:
                if self.goal_waypoints:
                    target = self.goal_waypoints[idx]
                grip = True
            elif self.phase == Phase.PLACE:
                target = base_pos
                grip = False
            else:
                target = base_pos

            base_vel = self._p_base_controller(base_pos, target) if target is not None else (0.0, 0.0, 0.0)
            actions[robot.base_id] = RobotAction(base_vel=base_vel, grip=grip)
        return actions

    def _object_contact_target(self, base_pos: np.ndarray) -> np.ndarray:
        if self.cfg.use_noisy_control:
            obj_pos, _ = self._get_object_pose(noisy=True)
        else:
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        obj_pos = np.array(obj_pos)
        vec = obj_pos - base_pos
        dist = np.linalg.norm(vec[:2]) + 1e-6
        offset = vec[:2] / dist * (self.cfg.contact_dist * 0.5)
        return np.array([obj_pos[0] - offset[0], obj_pos[1] - offset[1], base_pos[2]])

    def _p_base_controller(self, base_pos: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        if target is None:
            return (0.0, 0.0, 0.0)
        delta = target[:2] - base_pos[:2]
        if self.phase == Phase.APPROACH and self.object_id is not None and self.object_spec is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
            obj_xy = np.array(obj_pos[:2], dtype=np.float32)
            to_obj = base_pos[:2] - obj_xy
            obj_dist = float(np.linalg.norm(to_obj) + 1e-6)
            obj_radius = 0.5 * max(float(self.object_spec.dims[0]), float(self.object_spec.dims[1]))
            avoid_radius = obj_radius + 0.45
            if obj_dist < avoid_radius:
                away = to_obj / obj_dist
                avoid_gain = (avoid_radius - obj_dist) / max(avoid_radius, 1e-6)
                # Repulsive and tangential components to avoid deadlocks at object boundary.
                delta = delta + away * (0.8 * avoid_gain)
                tangent = np.array([-away[1], away[0]], dtype=np.float32)
                to_goal_from_obj = target[:2] - obj_xy
                turn_sign = 1.0 if (away[0] * to_goal_from_obj[1] - away[1] * to_goal_from_obj[0]) >= 0.0 else -1.0
                delta = delta + turn_sign * tangent * (0.25 * avoid_gain)
        dist = np.linalg.norm(delta)
        if dist < 1e-3:
            return (0.0, 0.0, 0.0)
        vel_dir = delta / dist
        speed = min(self.cfg.speed_limit, dist * 0.8)
        vx, vy = vel_dir[0] * speed, vel_dir[1] * speed
        return (vx, vy, 0.0)

    def _contact_force(self, robot: RobotInstance) -> float:
        if self.object_id is None:
            return 0.0
        link_idx = robot.end_effector_link if robot.end_effector_link is not None else -1
        total_force = 0.0
        contacts = p.getContactPoints(
            bodyA=robot.arm_id,
            bodyB=self.object_id,
            linkIndexA=int(link_idx),
            physicsClientId=self.client_id,
        )
        for contact in contacts:
            # contact[9] is normal force in PyBullet
            total_force += float(contact[9])
        return total_force

    def _udp_poll(self) -> None:
        if self.udp_bus is None:
            return
        packets = self.udp_bus.poll()
        self.peer_packets = {
            name: {
                "phase": pkt.phase,
                "ready": int(pkt.ready),
                "sim_time": float(pkt.sim_time),
                "pos": [float(pkt.pos[0]), float(pkt.pos[1])],
                "vel": [float(pkt.vel[0]), float(pkt.vel[1])],
            }
            for name, pkt in packets.items()
        }

    def _phase_local_ready_map(self) -> Dict[str, int]:
        ready: Dict[str, int] = {}
        for robot in self.robots:
            ready[robot.spec.name] = 1 if self._phase_local_ready(robot) else 0
        return ready

    def _phase_local_ready(self, robot: RobotInstance) -> bool:
        if self.phase == Phase.APPROACH:
            if robot.waypoint is None:
                return False
            pos, _ = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
            return np.linalg.norm(np.array(pos)[:2] - robot.waypoint[:2]) < self.cfg.phase_approach_dist

        if self.phase == Phase.CONTACT:
            if robot.constraint_id is not None:
                return True
            return self._contact_force(robot) >= self.cfg.phase_contact_force_threshold

        if self.phase == Phase.LIFT:
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
            obj_vel, _ = p.getBaseVelocity(self.object_id, physicsClientId=self.client_id)
            h_thresh = self.object_start_height + max(0.05, self.cfg.lift_height * 0.8)
            stable = (
                float(obj_pos[2]) >= float(h_thresh)
                and abs(float(obj_vel[2])) <= self.cfg.phase_lift_vertical_speed_max
            )
            if stable:
                if self._lift_stable_since is None:
                    self._lift_stable_since = self.sim_time
            else:
                self._lift_stable_since = None
            return bool(
                self._lift_stable_since is not None
                and (self.sim_time - self._lift_stable_since) >= self.cfg.phase_lift_stability_s
            )

        if self.phase == Phase.TRANSPORT:
            return self._object_near_goal()

        return False

    def _phase_ready_count(self) -> int:
        if self.phase not in (Phase.APPROACH, Phase.CONTACT, Phase.LIFT, Phase.TRANSPORT):
            return 0
        count = 0
        for robot in self.robots:
            pkt = self.peer_packets.get(robot.spec.name)
            if pkt is None:
                continue
            if pkt.get("phase") != self.phase.value:
                continue
            if int(pkt.get("ready", 0)) == 1:
                count += 1
        return count

    def _udp_broadcast_local_state(self, local_ready: Dict[str, int]) -> None:
        if self.udp_bus is None:
            return
        for robot in self.robots:
            pos, _ = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
            vel = self.last_safe_vel.get(robot.base_id, np.zeros(2, dtype=np.float32))
            self.udp_bus.send(
                sender=robot.spec.name,
                phase=self.phase.value,
                ready=int(local_ready.get(robot.spec.name, 0)),
                sim_time=float(self.sim_time),
                pos_xy=(float(pos[0]), float(pos[1])),
                vel_xy=(float(vel[0]), float(vel[1])),
            )

    def _phase_consensus_ready(self) -> Tuple[bool, float]:
        """
        Returns:
            (consensus_reached, sync_delay_seconds)
        """
        if self.phase not in (Phase.APPROACH, Phase.CONTACT, Phase.LIFT, Phase.TRANSPORT):
            return False, 0.0
        names = [robot.spec.name for robot in self.robots]
        ready_times: List[float] = []
        for name in names:
            pkt = self.peer_packets.get(name)
            if pkt is None:
                return False, 0.0
            if pkt.get("phase") != self.phase.value:
                return False, 0.0
            if int(pkt.get("ready", 0)) != 1:
                return False, 0.0
            ready_times.append(float(pkt.get("sim_time", -1e9)))
        if not ready_times:
            return False, 0.0
        delay = max(ready_times) - min(ready_times)
        if delay > self.cfg.phase_consensus_window_s:
            return False, delay
        return True, delay

    def _cbf_filter(self, robot: RobotInstance, v_des: np.ndarray) -> np.ndarray:
        self.cbf_stats["calls"] += 1
        pos_i, _ = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
        pos_i = np.array(pos_i[:2], dtype=np.float32)
        neighbor_pos = []
        neighbor_vel = []
        for other in self.robots:
            if other.base_id == robot.base_id:
                continue
            peer = self.peer_packets.get(other.spec.name)
            use_peer = (
                self.cfg.use_udp_neighbor_state
                and peer is not None
                and (self.sim_time - float(peer.get("sim_time", -1e9))) <= self.cfg.udp_state_ttl_s
            )
            if use_peer:
                neighbor_pos.append(np.array(peer["pos"], dtype=np.float32))
                neighbor_vel.append(np.array(peer["vel"], dtype=np.float32))
            else:
                pos_j, _ = p.getBasePositionAndOrientation(other.base_id, physicsClientId=self.client_id)
                neighbor_pos.append(np.array(pos_j[:2], dtype=np.float32))
                neighbor_vel.append(self.last_safe_vel.get(other.base_id, np.zeros(2, dtype=np.float32)))

        eta = 1.0
        if self.cfg.use_ekf and self.object_belief is not None:
            eta = 1.0 + self.cfg.cbf_kappa * self.object_belief.uncertainty()
        v_max = self.cfg.speed_limit / max(eta, 1e-3)
        d_min = self.cfg.separation_min * self.cfg.cbf_risk_scale
        alpha_eff = self.cfg.cbf_alpha / max(eta, 1e-3)

        result = solve_cbf_qp(
            v_des=v_des,
            pos_i=pos_i,
            neighbor_pos=neighbor_pos,
            neighbor_vel=neighbor_vel,
            v_max=v_max,
            d_min=d_min,
            alpha=alpha_eff,
            slack_weight=self.cfg.cbf_slack_weight,
            slack_max=self.cfg.cbf_slack_max,
        )
        v_safe = result.v_safe

        if not result.success:
            # fallback: monitored stop
            v_safe = np.zeros_like(v_safe)
            self.cbf_stats["fallback"] += 1

        contact_force = self._contact_force(robot)
        if contact_force > self.cfg.contact_force_max:
            self.violations["force"] += 1
            v_safe = np.zeros_like(v_safe)
            self.cbf_stats["force_stop"] += 1

        if np.linalg.norm(v_safe - v_des) > 1e-4:
            self.cbf_stats["modified"] += 1

        return v_safe

    def _apply_actions(self, actions: Dict[int, RobotAction]) -> None:
        for robot in self.robots:
            action = actions.get(robot.base_id, RobotAction((0.0, 0.0, 0.0)))
            vx, vy, yaw_rate = action.base_vel
            v_des = np.array([vx, vy], dtype=np.float32)
            if self.cfg.use_cbf:
                v_safe = self._cbf_filter(robot, v_des)
                vx, vy = float(v_safe[0]), float(v_safe[1])

            speed = math.hypot(vx, vy)
            if speed > self.cfg.speed_limit:
                scale = self.cfg.speed_limit / max(speed, 1e-6)
                vx *= scale
                vy *= scale
                self.violations["speed"] += 1

            self._apply_base_motion(robot, vx, vy, yaw_rate)
            self.last_safe_vel[robot.base_id] = np.array([vx, vy], dtype=np.float32)

            if self.cfg.use_ik and robot.end_effector_link >= 0:
                if self.phase in (Phase.CONTACT, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
                    self._apply_ik(robot)
                else:
                    self._hold_arm_home(robot)

            robot.grip_active = action.grip
            if robot.grip_active:
                self._maybe_attach(robot)
            else:
                self._maybe_detach(robot)

        self._apply_separation_safety()

    def _apply_base_motion(self, robot: RobotInstance, vx: float, vy: float, yaw_rate: float) -> None:
        pos, quat = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
        yaw = _yaw_from_quat(quat)
        if robot.spec.base_type == "diff":
            speed = math.hypot(vx, vy)
            if speed > 1e-6:
                desired_heading = math.atan2(vy, vx)
                heading_error = _wrap_angle(desired_heading - yaw)
                yaw_rate = _clamp(
                    heading_error * self.cfg.diff_heading_k,
                    -self.cfg.yaw_rate_max,
                    self.cfg.yaw_rate_max,
                )
            else:
                yaw_rate = 0.0
            forward_speed = math.cos(yaw) * vx + math.sin(yaw) * vy
            forward_speed = _clamp(forward_speed, -self.cfg.speed_limit, self.cfg.speed_limit)
            vx = math.cos(yaw) * forward_speed
            vy = math.sin(yaw) * forward_speed
        else:
            yaw_rate = _clamp(yaw_rate, -self.cfg.yaw_rate_max, self.cfg.yaw_rate_max)

        if self.cfg.kinematic_base:
            new_pos = [pos[0] + vx * self.control_dt, pos[1] + vy * self.control_dt, pos[2]]
            new_yaw = yaw + yaw_rate * self.control_dt
            p.resetBasePositionAndOrientation(
                robot.base_id, new_pos, _quat_from_yaw(new_yaw), physicsClientId=self.client_id
            )
            return

        use_wheel_drive = self.cfg.base_drive_mode == "wheel" and bool(robot.wheel_joints)
        if not use_wheel_drive:
            # Keep wheel angular velocity consistent with commanded chassis velocity
            # so contact friction does not immediately cancel the commanded motion.
            self._drive_wheels(robot, vx, vy, yaw_rate, yaw)
            p.resetBaseVelocity(
                robot.base_id,
                linearVelocity=[vx, vy, 0.0],
                angularVelocity=[0.0, 0.0, yaw_rate],
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                robot.arm_id,
                linearVelocity=[vx, vy, 0.0],
                angularVelocity=[0.0, 0.0, yaw_rate],
                physicsClientId=self.client_id,
            )
            return
        self._drive_wheels(robot, vx, vy, yaw_rate, yaw)

    def _drive_wheels(self, robot: RobotInstance, vx: float, vy: float, yaw_rate: float, yaw: float) -> None:
        if not robot.wheel_joints:
            return
        wheel_r = self.cfg.wheel_radius
        half_w = self.cfg.track_width * 0.5
        half_l = self.cfg.wheel_base * 0.5

        vx_body = math.cos(yaw) * vx + math.sin(yaw) * vy
        vy_body = -math.sin(yaw) * vx + math.cos(yaw) * vy

        if robot.spec.base_type == "diff":
            omega_left = (vx_body - yaw_rate * half_w) / wheel_r
            omega_right = (vx_body + yaw_rate * half_w) / wheel_r
            speeds = [omega_left, omega_right]
            motor_force = float(self.cfg.wheel_motor_force_diff)
        else:
            omega_fl = (vx_body - vy_body - (half_l + half_w) * yaw_rate) / wheel_r
            omega_fr = (vx_body + vy_body + (half_l + half_w) * yaw_rate) / wheel_r
            omega_rl = (vx_body + vy_body - (half_l + half_w) * yaw_rate) / wheel_r
            omega_rr = (vx_body - vy_body + (half_l + half_w) * yaw_rate) / wheel_r
            speeds = [omega_fl, omega_fr, omega_rl, omega_rr]
            motor_force = float(self.cfg.wheel_motor_force_omni)

        for joint_idx, target_vel in zip(robot.wheel_joints, speeds):
            p.setJointMotorControl2(
                robot.base_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=float(target_vel),
                force=motor_force,
                physicsClientId=self.client_id,
            )

    def _apply_ik(self, robot: RobotInstance) -> None:
        if self.object_id is None or self.object_spec is None:
            return
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        lift_offset = 0.0
        if self.phase in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
            lift_offset = self.current_lift
        target_x = obj_pos[0]
        target_y = obj_pos[1]
        if robot.waypoint_offset is not None:
            target_x = obj_pos[0] + robot.waypoint_offset[0]
            target_y = obj_pos[1] + robot.waypoint_offset[1]
        target = [
            target_x,
            target_y,
            obj_pos[2] + self.object_spec.dims[2] * 0.5 + lift_offset,
        ]
        try:
            joints = p.calculateInverseKinematics(
                robot.arm_id,
                robot.end_effector_link,
                target,
                physicsClientId=self.client_id,
            )
        except Exception:
            return
        for idx, joint_idx in enumerate(robot.joint_indices):
            if idx >= len(joints):
                break
            p.setJointMotorControl2(
                robot.arm_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joints[idx],
                force=500.0,
                physicsClientId=self.client_id,
            )

    def _hold_arm_home(self, robot: RobotInstance) -> None:
        if not robot.joint_indices:
            return
        if robot.spec.dof >= 7:
            home = [0.0, -0.7, 0.0, 1.3, 0.0, -0.8, 0.0]
        else:
            home = [0.0, -0.8, 0.0, 1.5, 0.0, 0.6]
        for idx, joint_idx in enumerate(robot.joint_indices):
            target = home[idx] if idx < len(home) else 0.0
            p.setJointMotorControl2(
                robot.arm_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=600.0,
                physicsClientId=self.client_id,
            )

    def _maybe_attach(self, robot: RobotInstance) -> None:
        if robot.constraint_id is not None or self.object_id is None:
            return
        if self.active_carry_mode != "constraint":
            return
        self.grasp_stats["attach_attempts"] += 1
        attach_dist = max(self.cfg.contact_dist, self.cfg.vacuum_attach_dist)
        closest = p.getClosestPoints(
            bodyA=robot.arm_id,
            bodyB=self.object_id,
            distance=attach_dist,
            linkIndexA=robot.end_effector_link,
            physicsClientId=self.client_id,
        )
        parent_link = robot.end_effector_link
        if not closest:
            closest = p.getClosestPoints(
                bodyA=robot.arm_id,
                bodyB=self.object_id,
                distance=attach_dist * 2.0,
                physicsClientId=self.client_id,
            )
            if not closest:
                return
            closest = sorted(closest, key=lambda c: float(c[8]))  # contact distance
            parent_link = int(closest[0][3])
        robot.constraint_id = p.createConstraint(
            robot.arm_id,
            parent_link,
            self.object_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.client_id,
        )
        max_force = self._constraint_force_capacity(robot)
        p.changeConstraint(robot.constraint_id, maxForce=max_force)
        self.grasp_stats["attach_success"] += 1

    def _maybe_detach(self, robot: RobotInstance, reason: str = "release") -> None:
        if robot.constraint_id is None:
            return
        p.removeConstraint(robot.constraint_id, physicsClientId=self.client_id)
        robot.constraint_id = None
        self.grasp_stats["detach_events"] += 1
        if reason == "stretch":
            self.grasp_stats["stretch_drop"] += 1
        elif reason == "overload":
            self.grasp_stats["overload_drop"] += 1

    def _constraint_force_capacity(self, robot: RobotInstance) -> float:
        return max(300.0, robot.spec.payload * 9.81 * self.cfg.constraint_force_scale)

    def _end_effector_pos(self, robot: RobotInstance) -> Tuple[float, float, float]:
        if robot.end_effector_link < 0:
            pos, _ = p.getBasePositionAndOrientation(robot.arm_id, physicsClientId=self.client_id)
            return pos
        link_state = p.getLinkState(robot.arm_id, robot.end_effector_link, physicsClientId=self.client_id)
        return link_state[0]

    def _apply_separation_safety(self) -> None:
        positions = []
        for robot in self.robots:
            pos, quat = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
            positions.append((robot, np.array(pos), quat))

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                ri, pi, qi = positions[i]
                rj, pj, qj = positions[j]
                delta = pi[:2] - pj[:2]
                dist = np.linalg.norm(delta)
                if dist < self.cfg.separation_min:
                    self.violations["separation"] += 1
                    continue

    def _step_physics(self) -> None:
        steps = int(self.control_dt / self.physics_dt)
        for _ in range(max(1, steps)):
            p.stepSimulation(physicsClientId=self.client_id)
        self.sim_time += self.control_dt
        self.phase_time += self.control_dt
        if self.cfg.use_ekf and self.object_belief is not None:
            self.object_belief.predict()
            meas_pos, _ = self._get_object_pose(noisy=True)
            self.object_belief.update((float(meas_pos[0]), float(meas_pos[1])))
        self._update_lift_target()
        self._maybe_fallback_carry_mode()
        self._carry_object_if_needed()
        self._check_contact_forces()
        self._enforce_vacuum_limits()

    def _check_contact_forces(self) -> None:
        if self.object_id is None:
            return
        for robot in self.robots:
            normal_force = self._contact_force(robot)
            if self.cfg.sensor_force_noise > 0.0:
                normal_force += float(self.rng.normal(0.0, self.cfg.sensor_force_noise))
                normal_force = max(0.0, normal_force)
            if normal_force > self.cfg.contact_force_max:
                self.violations["force"] += 1
                return

    def _maybe_fallback_carry_mode(self) -> None:
        if self.cfg.carry_mode != "auto":
            return
        if self.active_carry_mode != "constraint":
            return
        if self.phase not in (Phase.CONTACT, Phase.LIFT, Phase.TRANSPORT):
            return
        if self._all_constraints_attached():
            return
        if self.phase_time >= self.cfg.constraint_fallback_time:
            for robot in self.robots:
                self._maybe_detach(robot, reason="release")
            self.active_carry_mode = "kinematic"

    def _enforce_vacuum_limits(self) -> None:
        if self.object_id is None:
            return
        if self.active_carry_mode != "constraint":
            return
        attached = [robot for robot in self.robots if robot.constraint_id is not None]
        if not attached:
            return

        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        obj_pos = np.array(obj_pos, dtype=np.float32)
        break_dist = max(self.cfg.contact_dist * 1.5, self.cfg.vacuum_break_dist)
        for robot in list(attached):
            ee_pos = np.array(self._end_effector_pos(robot), dtype=np.float32)
            if float(np.linalg.norm(ee_pos - obj_pos)) > break_dist:
                self._maybe_detach(robot, reason="stretch")

        attached = [robot for robot in self.robots if robot.constraint_id is not None]
        if not attached:
            return
        if self.phase not in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
            return
        if self.object_spec is None:
            return

        required_force = self.object_spec.mass * 9.81 * self.cfg.vacuum_force_margin
        available_force = sum(self._constraint_force_capacity(robot) for robot in attached)
        if available_force + 1e-6 >= required_force:
            return
        for robot in list(attached):
            self._maybe_detach(robot, reason="overload")

    def _update_lift_target(self) -> None:
        if self.phase == Phase.LIFT:
            self.current_lift = min(
                self.cfg.lift_height, self.current_lift + self.cfg.lift_speed * self.control_dt
            )
        elif self.phase == Phase.PLACE:
            self.current_lift = max(
                0.0, self.current_lift - self.cfg.lift_speed * self.control_dt
            )

    def _carry_object_if_needed(self) -> None:
        if self.object_id is None:
            return
        if self.phase not in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
            return
        if not self._all_gripping():
            return
        if self.active_carry_mode != "kinematic":
            return

        ee_positions = np.array([self._end_effector_pos(r) for r in self.robots])
        centroid = ee_positions.mean(axis=0)
        target_pos = [centroid[0], centroid[1], centroid[2] + self.current_lift]
        p.resetBasePositionAndOrientation(
            self.object_id,
            target_pos,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

    def _all_constraints_attached(self) -> bool:
        for robot in self.robots:
            if robot.grip_active and robot.constraint_id is None:
                return False
        return True

    def _update_phase(self) -> None:
        if self.cfg.use_udp_phase and self.udp_bus is not None:
            self._update_phase_distributed()
            return
        self._update_phase_centralized()

    def _update_phase_centralized(self) -> None:
        if self.phase == Phase.PLAN:
            self._plan_formation()
            self.phase = Phase.APPROACH
            self.phase_time = 0.0
            return

        if self.phase == Phase.APPROACH:
            local_ready = self._phase_local_ready_map()
            if all(v == 1 for v in local_ready.values()):
                self.phase = Phase.CONTACT
                self.phase_time = 0.0
                return
            if (
                self.phase_time >= self.cfg.phase_approach_timeout_s
                and sum(local_ready.values()) >= self.cfg.phase_approach_min_ready
            ):
                self.phase = Phase.CONTACT
                self.phase_time = 0.0
                return

        if self.phase == Phase.CONTACT and self._all_gripping():
            self.phase = Phase.LIFT
            self.phase_time = 0.0
            return

        if self.phase == Phase.LIFT and self.current_lift >= self.cfg.lift_height - 1e-3:
            self.phase = Phase.TRANSPORT
            self.phase_time = 0.0
            self._plan_goal_formation()
            return

        if self.phase == Phase.TRANSPORT and self._object_near_goal():
            self.phase = Phase.PLACE
            self.phase_time = 0.0
            return

        if self.phase == Phase.PLACE and self.current_lift <= 1e-3:
            self.phase = Phase.DONE
            self.phase_time = 0.0

    def _update_phase_distributed(self) -> None:
        if self.phase == Phase.PLAN:
            self._plan_formation()
            self.phase = Phase.APPROACH
            self.phase_time = 0.0
            return
        if self.phase == Phase.PLACE:
            if self.current_lift <= 1e-3:
                self.phase = Phase.DONE
                self.phase_time = 0.0
            return
        if self.phase == Phase.DONE:
            return

        ready, delay = self._phase_consensus_ready()
        if not ready:
            # Timeout fallback for approach deadlocks: proceed if quorum reached.
            if (
                self.phase == Phase.APPROACH
                and self.phase_time >= self.cfg.phase_approach_timeout_s
                and self._phase_ready_count() >= self.cfg.phase_approach_min_ready
            ):
                self.phase_sync_last_delay_ms = float(self.cfg.phase_consensus_window_s * 1000.0)
                self.phase_sync_delays_ms.append(self.phase_sync_last_delay_ms)
                self.phase = Phase.CONTACT
                self.phase_time = 0.0
            return

        self.phase_sync_last_delay_ms = float(delay * 1000.0)
        self.phase_sync_delays_ms.append(self.phase_sync_last_delay_ms)

        if self.phase == Phase.APPROACH:
            self.phase = Phase.CONTACT
            self.phase_time = 0.0
            return

        if self.phase == Phase.CONTACT:
            self.phase = Phase.LIFT
            self.phase_time = 0.0
            return

        if self.phase == Phase.LIFT:
            self.phase = Phase.TRANSPORT
            self.phase_time = 0.0
            self._plan_goal_formation()
            return

        if self.phase == Phase.TRANSPORT:
            self.phase = Phase.PLACE
            self.phase_time = 0.0
            return

    def _plan_formation(self) -> None:
        if self.object_id is None or self.object_spec is None:
            return
        if self.cfg.use_noisy_plan:
            obj_pos, obj_quat = self._get_object_pose(noisy=True)
        else:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        dims = self.object_spec.dims
        vlm_out = self._vlm_json_formation()
        if vlm_out is None:
            waypoints, load_labels, conf = self._vlm_stub_formation(dims)
        else:
            waypoints, load_labels, conf = vlm_out
        if conf < self.cfg.vlm_confidence_threshold:
            waypoints, load_labels = self._geometric_fallback(dims)
        world_waypoints = self._to_world_waypoints(obj_pos, obj_quat, waypoints)
        assignment = self._assign_robots(world_waypoints, load_labels)
        obj_center = np.array(obj_pos)
        for robot, waypoint in zip(self.robots, assignment):
            robot.waypoint = np.array(waypoint)
            robot.waypoint_offset = np.array(waypoint) - obj_center

    def _plan_goal_formation(self) -> None:
        if self.object_spec is None:
            return
        dims = self.object_spec.dims
        center = np.array(
            [self.goal_pos[0], self.goal_pos[1], self.podium_height + dims[2] * 0.5]
        )
        offsets = [robot.waypoint_offset for robot in self.robots if robot.waypoint_offset is not None]
        if len(offsets) == len(self.robots):
            self.goal_waypoints = [center + offset for offset in offsets]
        else:
            base_waypoints, _labels = self._geometric_fallback(dims)
            self.goal_waypoints = self._to_world_waypoints(center, (0.0, 0.0, 0.0, 1.0), base_waypoints)

    def _load_vlm_json(self) -> Optional[dict]:
        if not self.cfg.vlm_json_path:
            return None
        path = Path(self.cfg.vlm_json_path)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if self._vlm_json_cache and self._vlm_json_cache.get("_mtime") == mtime:
            return self._vlm_json_cache.get("data")
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        self._vlm_json_cache = {"_mtime": mtime, "data": data}
        return data

    def _normalize_load_label(self, value) -> str:
        if isinstance(value, str):
            text = value.strip().lower()
            if "high" in text or text.startswith("h"):
                return "high"
            if "low" in text or text.startswith("l"):
                return "low"
        if isinstance(value, (int, float)):
            return "high" if float(value) >= 0.5 else "low"
        return "low"

    def _parse_positions(self, positions, labels=None) -> Tuple[List[Tuple[float, float]], List[str]]:
        points: List[Tuple[float, float]] = []
        load_labels: List[str] = []
        labels = labels or []
        for idx, entry in enumerate(positions):
            x = y = None
            if isinstance(entry, dict):
                if "pos" in entry and len(entry["pos"]) >= 2:
                    x, y = entry["pos"][0], entry["pos"][1]
                elif "position" in entry and len(entry["position"]) >= 2:
                    x, y = entry["position"][0], entry["position"][1]
                elif "x" in entry and "y" in entry:
                    x, y = entry["x"], entry["y"]
                label_val = entry.get("load", entry.get("label", None))
                label = self._normalize_load_label(label_val)
            else:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    x, y = entry[0], entry[1]
                label = self._normalize_load_label(labels[idx] if idx < len(labels) else None)
            if x is None or y is None:
                continue
            points.append((float(x), float(y)))
            load_labels.append(label)
        return points, load_labels

    def _vlm_json_formation(self) -> Optional[Tuple[List[Tuple[float, float]], List[str], float]]:
        data = self._load_vlm_json()
        if data is None:
            return None
        if isinstance(data, dict) and "formation" in data:
            data = data["formation"]

        confidence = 1.0
        if isinstance(data, dict) and "confidence" in data:
            confidence = float(data.get("confidence", 1.0))

        points: List[Tuple[float, float]] = []
        labels: List[str] = []

        if isinstance(data, dict) and "waypoints" in data:
            points, labels = self._parse_positions(data.get("waypoints", []))
        elif isinstance(data, dict) and "positions" in data:
            points, labels = self._parse_positions(
                data.get("positions", []), labels=data.get("load_labels", [])
            )
        elif isinstance(data, list):
            points, labels = self._parse_positions(data)

        if len(points) < 4:
            return None
        points = points[:4]
        labels = (labels + ["low"] * 4)[:4]
        return points, labels, confidence

    def _vlm_stub_formation(self, dims: Tuple[float, float, float]) -> Tuple[List[Tuple[float, float]], List[str], float]:
        length, width, _ = dims
        points = [
            (length * 0.5, 0.0),
            (-length * 0.5, 0.0),
            (0.0, width * 0.5),
            (0.0, -width * 0.5),
        ]
        load_labels = ["high", "high", "low", "low"]
        confidence = float(self.rng.uniform(0.4, 0.9))
        return points, load_labels, confidence

    def _geometric_fallback(self, dims: Tuple[float, float, float]) -> Tuple[List[Tuple[float, float]], List[str]]:
        length, width, _ = dims
        points = [
            (length / 3.0, 0.0),
            (-length / 3.0, 0.0),
            (0.0, width / 3.0),
            (0.0, -width / 3.0),
        ]
        load_labels = ["high", "high", "low", "low"]
        return points, load_labels

    def _to_world_waypoints(
        self,
        obj_pos: Tuple[float, float, float],
        obj_quat: Tuple[float, float, float, float],
        points: List[Tuple[float, float]],
    ) -> List[np.ndarray]:
        rot = np.array(p.getMatrixFromQuaternion(obj_quat), dtype=float).reshape(3, 3)
        obj_center = np.array(obj_pos)
        world_points = []
        for pt in points:
            local = np.array([pt[0], pt[1], 0.0])
            world = obj_center + rot @ local
            normal = world - obj_center
            norm = np.linalg.norm(normal[:2])
            if norm < 1e-6:
                normal = rot @ np.array([1.0, 0.0, 0.0])
            else:
                normal = normal / max(norm, 1e-6)
            world_points.append(world + normal * self.cfg.standoff)
        return world_points

    def _assign_robots(self, waypoints: List[np.ndarray], labels: List[str]) -> List[np.ndarray]:
        heavy_indices = [i for i, r in enumerate(self.robots) if r.spec.payload >= 100.0]
        agile_indices = [i for i, r in enumerate(self.robots) if r.spec.payload < 100.0]
        high_pts = [waypoints[i] for i, lab in enumerate(labels) if lab == "high"]
        low_pts = [waypoints[i] for i, lab in enumerate(labels) if lab != "high"]

        assignment = [None] * len(self.robots)
        if high_pts:
            heavy_assignment = self._hungarian_assign(heavy_indices, high_pts)
            for idx, wp in heavy_assignment.items():
                assignment[idx] = wp
        if low_pts:
            agile_assignment = self._hungarian_assign(agile_indices, low_pts)
            for idx, wp in agile_assignment.items():
                assignment[idx] = wp
        for i, wp in enumerate(assignment):
            if wp is None:
                assignment[i] = waypoints[i % len(waypoints)]
        return assignment

    def _hungarian_assign(self, robot_indices: List[int], waypoints: List[np.ndarray]) -> Dict[int, np.ndarray]:
        if not robot_indices or not waypoints:
            return {}
        positions = []
        for idx in robot_indices:
            pos, _ = p.getBasePositionAndOrientation(self.robots[idx].base_id, physicsClientId=self.client_id)
            positions.append(np.array(pos))
        cost = np.zeros((len(robot_indices), len(waypoints)))
        for i, pos in enumerate(positions):
            for j, wp in enumerate(waypoints):
                cost[i, j] = np.linalg.norm(pos[:2] - wp[:2])

        if linear_sum_assignment is None:
            assign = {}
            used = set()
            for i in range(cost.shape[0]):
                j = int(np.argmin(cost[i]))
                while j in used and len(used) < cost.shape[1]:
                    cost[i, j] = np.inf
                    j = int(np.argmin(cost[i]))
                used.add(j)
                assign[robot_indices[i]] = waypoints[j]
            return assign

        row_ind, col_ind = linear_sum_assignment(cost)
        return {robot_indices[i]: waypoints[j] for i, j in zip(row_ind, col_ind)}

    def _all_at_waypoints(self) -> bool:
        for robot in self.robots:
            if robot.waypoint is None:
                return False
            pos, _ = p.getBasePositionAndOrientation(robot.base_id, physicsClientId=self.client_id)
            if np.linalg.norm(np.array(pos)[:2] - robot.waypoint[:2]) > 0.1:
                return False
        return True

    def _all_gripping(self) -> bool:
        return all(robot.grip_active for robot in self.robots)

    def _object_near_goal(self) -> bool:
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        dist = np.linalg.norm(np.array(obj_pos)[:2] - self.goal_pos[:2])
        return dist < 0.2
