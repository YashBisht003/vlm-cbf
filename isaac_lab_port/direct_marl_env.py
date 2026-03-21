from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = _Box

    class _Gym:
        spaces = _Spaces()

    gym = _Gym()

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
except Exception:
    class _UsdFileCfg:
        def __init__(
            self,
            usd_path: str = "",
            activate_contact_sensors: bool = False,
            rigid_props: object | None = None,
            articulation_props: object | None = None,
        ) -> None:
            self.usd_path = str(usd_path)
            self.activate_contact_sensors = bool(activate_contact_sensors)
            self.rigid_props = rigid_props
            self.articulation_props = articulation_props

    @dataclass
    class _RigidBodyPropertiesCfg:
        disable_gravity: bool = False
        max_depenetration_velocity: float = 5.0

    @dataclass
    class _ArticulationRootPropertiesCfg:
        enabled_self_collisions: bool = False
        solver_position_iteration_count: int = 8
        solver_velocity_iteration_count: int = 0

    class _SimUtils:
        UsdFileCfg = _UsdFileCfg
        RigidBodyPropertiesCfg = _RigidBodyPropertiesCfg
        ArticulationRootPropertiesCfg = _ArticulationRootPropertiesCfg

    sim_utils = _SimUtils()

    @dataclass
    class SimulationCfg:
        dt: float = 0.02
        device: str = "cpu"

    @dataclass
    class InteractiveSceneCfg:
        num_envs: int = 1
        env_spacing: float = 1.0
        replicate_physics: bool = True

    @dataclass
    class ContactSensorCfg:
        prim_path: str = ""
        track_pose: bool = False
        debug_vis: bool = False
        update_period: float = 0.0
        filter_prim_paths_expr: List[str] = field(default_factory=list)

    @dataclass
    class ArticulationCfg:
        @dataclass
        class InitialStateCfg:
            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            joint_pos: Dict[str, float] = field(default_factory=dict)

        prim_path: str = ""
        spawn: object = None
        init_state: object = None
        actuators: Dict[str, object] = field(default_factory=dict)

        def replace(self, **kwargs):
            out = copy.deepcopy(self)
            for k, v in kwargs.items():
                setattr(out, k, v)
            return out

    @dataclass
    class RigidObjectCfg:
        prim_path: str = ""
        spawn: object = None

        def replace(self, **kwargs):
            out = copy.deepcopy(self)
            for k, v in kwargs.items():
                setattr(out, k, v)
            return out

    @dataclass
    class ImplicitActuatorCfg:
        joint_names_expr: Sequence[str] = field(default_factory=list)
        effort_limit: float = 0.0
        velocity_limit: float = 0.0
        stiffness: float = 0.0
        damping: float = 0.0

    class _EntityData:
        def __init__(self) -> None:
            self.default_root_state = torch.zeros((1, 13), dtype=torch.float32)
            self.default_joint_pos = torch.zeros((1, 13), dtype=torch.float32)
            self.default_joint_vel = torch.zeros((1, 13), dtype=torch.float32)
            self.joint_names = [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]

    class Articulation:
        def __init__(self, _cfg: ArticulationCfg) -> None:
            self.data = _EntityData()
            self.joint_names = list(self.data.joint_names)

        def write_root_pose_to_sim(self, _pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
            _ = env_ids

        def write_root_velocity_to_sim(self, _vel: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
            _ = env_ids

        def write_joint_state_to_sim(
            self, _joint_pos: torch.Tensor, _joint_vel: torch.Tensor, env_ids: torch.Tensor | None = None
        ) -> None:
            _ = env_ids

        def set_joint_velocity_target(self, _target: torch.Tensor, joint_ids=None) -> None:
            _ = joint_ids

        def set_joint_position_target(self, _target: torch.Tensor, joint_ids=None) -> None:
            _ = joint_ids

        def write_data_to_sim(self) -> None:
            return

        def update(self, _dt: float) -> None:
            return

    class RigidObject:
        def __init__(self, _cfg: RigidObjectCfg) -> None:
            self.data = _EntityData()

        def write_root_pose_to_sim(self, _pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
            _ = env_ids

        def write_root_velocity_to_sim(self, _vel: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
            _ = env_ids

        def update(self, _dt: float) -> None:
            return

    class DirectMARLEnvCfg:
        pass

    def configclass(cls):
        return cls

    class DirectMARLEnv:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Isaac Lab is not available. This module is a scaffold and requires Isaac Lab 2.3.2 + Isaac Sim 5.1."
            )

try:
    from .torch_phase_manager import PHASE_TO_ID, Phase, PhaseConfig, TorchBatchedNoVlmPhaseManager, TorchBatchedPhaseInputs
    from .vacuum_attachment import AutoAttachmentBackend
except ImportError:
    from torch_phase_manager import PHASE_TO_ID, Phase, PhaseConfig, TorchBatchedNoVlmPhaseManager, TorchBatchedPhaseInputs
    from vacuum_attachment import AutoAttachmentBackend

try:
    from belief_ekf import BeliefEKF
    from cbf_qp import solve_cbf_qp
    from neural_cbf import NeuralCbfRuntime
    from residual_corrector import FEATURE_NAMES as RESIDUAL_FEATURE_NAMES
    from residual_corrector import ResidualCorrectorRuntime
except ImportError:
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from belief_ekf import BeliefEKF
    from cbf_qp import solve_cbf_qp
    from neural_cbf import NeuralCbfRuntime
    from residual_corrector import FEATURE_NAMES as RESIDUAL_FEATURE_NAMES
    from residual_corrector import ResidualCorrectorRuntime


BASE_ACT_DIM = 3
ARM_ACT_DIM = 3
GRIP_ACT_DIM = 1
BASE_CMD_DIM = 3
AGENT_ACT_DIM = BASE_ACT_DIM + ARM_ACT_DIM + GRIP_ACT_DIM
AGENT_OBS_DIM = 65
# Centralized critic state = concat(4 * per-agent obs) + global block.
# With current features: 4*65 + 72 = 332.
CENTRAL_STATE_DIM = AGENT_OBS_DIM * 4 + 72
ENV_REGEX_NS = "{ENV_REGEX_NS}"


def _agent_names(num_robots: int) -> tuple[str, ...]:
    return tuple(f"robot_{i}" for i in range(int(num_robots)))


def _robot_prim_paths(num_robots: int) -> tuple[str, ...]:
    return tuple(f"{ENV_REGEX_NS}/Robot_{i}" for i in range(int(num_robots)))


def _clone_cfg_with_prim_path(cfg_obj, prim_path: str):
    if hasattr(cfg_obj, "replace"):
        return cfg_obj.replace(prim_path=prim_path)
    out = copy.deepcopy(cfg_obj)
    setattr(out, "prim_path", prim_path)
    return out


def _enable_contact_sensors(cfg_obj):
    out = copy.deepcopy(cfg_obj)
    spawn = getattr(out, "spawn", None)
    if spawn is not None and hasattr(spawn, "activate_contact_sensors"):
        setattr(spawn, "activate_contact_sensors", True)
    return out


def _load_first_attr(candidates: Sequence[tuple[str, str]]):
    for mod_name, attr_name in candidates:
        try:
            module = importlib.import_module(mod_name)
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
        except Exception:
            continue
    return None


def _resolve_implicit_actuator_cfg_cls():
    cls = _load_first_attr((("isaaclab.actuators", "ImplicitActuatorCfg"),))
    if cls is None:
        cls = globals().get("ImplicitActuatorCfg")
    return cls


def _default_ridgeback_franka_cfg(prim_path: str) -> ArticulationCfg:
    base = _load_first_attr(
        (
            ("isaaclab_assets.robots.ridgeback_franka", "RIDGEBACK_FRANKA_PANDA_CFG"),
            ("isaaclab_assets.robots", "RIDGEBACK_FRANKA_PANDA_CFG"),
            ("isaaclab_assets", "RIDGEBACK_FRANKA_PANDA_CFG"),
        )
    )
    if base is not None:
        return _clone_cfg_with_prim_path(_enable_contact_sensors(base), prim_path)

    usd_kwargs = {
        "usd_path": "Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
        "activate_contact_sensors": True,
    }
    if hasattr(sim_utils, "RigidBodyPropertiesCfg"):
        usd_kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        )
    if hasattr(sim_utils, "ArticulationRootPropertiesCfg"):
        usd_kwargs["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        )
    spawn_cfg = sim_utils.UsdFileCfg(**usd_kwargs)

    init_state = None
    if hasattr(ArticulationCfg, "InitialStateCfg"):
        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.3,
                "panda_joint3": 0.0,
                "panda_joint4": -2.2,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.8,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        )

    actuator_cfg_cls = _resolve_implicit_actuator_cfg_cls()
    actuators = {}
    if actuator_cfg_cls is not None:
        actuators = {
            "base_velocity": actuator_cfg_cls(
                joint_names_expr=[".*wheel.*"],
                effort_limit=400.0,
                velocity_limit=40.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "arm_position": actuator_cfg_cls(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit=120.0,
                velocity_limit=2.5,
                stiffness=320.0,
                damping=50.0,
            ),
            "gripper_position": actuator_cfg_cls(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=40.0,
                velocity_limit=0.5,
                stiffness=180.0,
                damping=15.0,
            ),
        }

    kwargs = {
        "prim_path": prim_path,
        "spawn": spawn_cfg,
    }
    if init_state is not None:
        kwargs["init_state"] = init_state
    if actuators:
        kwargs["actuators"] = actuators
    return ArticulationCfg(**kwargs)


def _default_payload_cfg(prim_path: str) -> RigidObjectCfg:
    base = _load_first_attr(
        (
            ("isaaclab_assets", "CUBOID_CFG"),
            ("isaaclab_assets.objects", "CUBOID_CFG"),
        )
    )
    if base is None:
        cuboid_cfg_cls = getattr(sim_utils, "CuboidCfg", None)
        rigid_props_cls = getattr(sim_utils, "RigidBodyPropertiesCfg", None)
        mass_props_cls = getattr(sim_utils, "MassPropertiesCfg", None)
        material_cls = getattr(sim_utils, "RigidBodyMaterialCfg", None)
        preview_cls = getattr(sim_utils, "PreviewSurfaceCfg", None)
        if cuboid_cfg_cls is not None:
            spawn_kwargs = {
                "size": (0.8, 0.6, 0.4),
            }
            if rigid_props_cls is not None:
                spawn_kwargs["rigid_props"] = rigid_props_cls(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                )
            if mass_props_cls is not None:
                spawn_kwargs["mass_props"] = mass_props_cls(mass=140.0)
            if material_cls is not None:
                spawn_kwargs["physics_material"] = material_cls()
            if preview_cls is not None:
                spawn_kwargs["visual_material"] = preview_cls(diffuse_color=(0.25, 0.55, 0.75))
            return RigidObjectCfg(
                prim_path=prim_path,
                spawn=cuboid_cfg_cls(**spawn_kwargs),
            )
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path="Props/Blocks/block.usd"),
        )
    return _clone_cfg_with_prim_path(base, prim_path)


def _build_agent_spaces(agent_names: Sequence[str]) -> tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    observation_spaces = {name: AGENT_OBS_DIM for name in agent_names}
    action_spaces = {name: AGENT_ACT_DIM for name in agent_names}
    shared_observation_spaces = {name: CENTRAL_STATE_DIM for name in agent_names}
    return observation_spaces, action_spaces, shared_observation_spaces


@configclass
@dataclass
class NoVlmSceneCfg(InteractiveSceneCfg):
    num_envs: int = 128
    env_spacing: float = 8.0
    replicate_physics: bool = True
    robot_0: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_0"))
    robot_1: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_1"))
    robot_2: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_2"))
    robot_3: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_3"))
    payload: RigidObjectCfg = field(default_factory=lambda: _default_payload_cfg(f"{ENV_REGEX_NS}/Payload"))
    contact_robot_0: ContactSensorCfg = field(
        default_factory=lambda: ContactSensorCfg(
            prim_path=f"{ENV_REGEX_NS}/Robot_0/panda_hand",
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Payload"],
        )
    )
    contact_robot_1: ContactSensorCfg = field(
        default_factory=lambda: ContactSensorCfg(
            prim_path=f"{ENV_REGEX_NS}/Robot_1/panda_hand",
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Payload"],
        )
    )
    contact_robot_2: ContactSensorCfg = field(
        default_factory=lambda: ContactSensorCfg(
            prim_path=f"{ENV_REGEX_NS}/Robot_2/panda_hand",
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Payload"],
        )
    )
    contact_robot_3: ContactSensorCfg = field(
        default_factory=lambda: ContactSensorCfg(
            prim_path=f"{ENV_REGEX_NS}/Robot_3/panda_hand",
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Payload"],
        )
    )


@configclass
@dataclass
class NoVlmCoopTransportEnvCfg(DirectMARLEnvCfg):
    """Direct MARL environment config contract for SKRL MAPPO."""

    num_robots: int = 4
    device: str = "cuda:0"
    decimation: int = 2
    episode_length_s: float = 30.0
    wheel_radius_m: float = 0.0762
    mecanum_yaw_coupling_m: float = 0.35
    arm_delta_limit_rad: float = 0.35
    approach_radius_m: float = 1.2
    fine_approach_radius_m: float = 0.55
    object_mass_nominal_kg: float = 140.0
    payload_dims_m: tuple[float, float, float] = (0.8, 0.6, 0.4)
    belief_meas_force_frac: float = 0.15
    belief_meas_total_frac: float = 0.10
    separation_min_m: float = 0.5
    speed_limit_mps: float = 0.25
    yaw_rate_limit_radps: float = 1.2
    contact_force_threshold_n: float = 5.0
    contact_force_balance_std_n: float = 20.0
    contact_force_overload_n: float = 220.0
    cbf_alpha: float = 2.0
    cbf_slack_weight: float = 100.0
    cbf_slack_max: float = 3.0
    cbf_slack_threshold: float = 0.25
    cbf_recovery_gain: float = 0.5
    cbf_enable_after_s: float = 0.0
    cbf_warmup_s: float = 2.0
    curriculum_phase: str = "full"
    curriculum_use_stage_cbf_schedule: bool = True
    curriculum_face_margin_m: float = 0.15
    arm_action_joint_names: tuple[str, ...] = ("panda_joint2", "panda_joint4", "panda_joint6")
    cbf_dmin_com_gain: float = 1.0
    cbf_alpha_mass_gain: float = 2.0
    cbf_speed_mass_gain: float = 1.0
    belief_force_ema_alpha: float = 0.25
    belief_min_contact_robots: int = 2
    probe_hold_s: float = 0.25
    probe_belief_com_threshold_m: float = 0.03
    probe_belief_mass_rel_threshold: float = 0.05
    regrip_hold_s: float = 0.5
    grip_attach_threshold: float = 0.25
    grip_release_threshold: float = -0.25
    grip_consensus_release_votes: int = 2
    grip_contact_force_threshold_n: float = 10.0
    regrip_balance_std_n: float = 15.0
    debug_belief_steps: int = 0
    debug_belief_log_every: int = 1
    debug_belief_env: int = 0
    debug_cbf_steps: int = 0
    debug_cbf_log_every: int = 1
    debug_cbf_env: int = 0
    use_cbf: bool = True
    use_belief_ekf: bool = True
    use_neural_cbf: bool = False
    neural_cbf_hidden: int = 64
    neural_cbf_model_path: str = ""
    residual_model_path: str = ""
    residual_gain: float = 0.12
    residual_max_offset_m: float = 0.12
    lift_settle_steps: int = 2
    lift_arm_stiffness: float = 800.0
    lift_arm_damping: float = 80.0
    lift_preshape_joint2_rad: float = -0.5
    lift_preshape_joint4_rad: float = -1.2
    lift_preshape_joint6_rad: float = 1.15
    lift_payload_mass_levels_kg: tuple[float, ...] = (6.0, 8.0, 10.0)
    lift_payload_mass_level: int = 0
    lift_contact_only: bool = True
    lift_contact_preload_m: float = 0.02
    lift_contact_vertical_preload_m: float = 0.02
    lift_height_m: float = 0.18
    place_height_m: float = 0.09
    goal_tolerance_m: float = 0.30
    sim: SimulationCfg = field(default_factory=lambda: SimulationCfg(dt=0.02, device="cuda:0"))
    scene: NoVlmSceneCfg = field(default_factory=NoVlmSceneCfg)

    # Do not use deprecated num_observations/num_actions.
    possible_agents: tuple[str, ...] = field(default_factory=tuple)
    robot_prim_paths: tuple[str, ...] = field(default_factory=tuple)
    observation_spaces: Dict[str, int] = field(default_factory=dict)
    action_spaces: Dict[str, int] = field(default_factory=dict)
    shared_observation_spaces: Dict[str, int] = field(default_factory=dict)

    # Positive value means env will use explicit _get_states() for critic state.
    state_space: int = CENTRAL_STATE_DIM

    def __post_init__(self) -> None:
        if self.num_robots != 4:
            raise ValueError("Current scaffold supports exactly 4 robots.")
        if self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be > 0.")
        phase = str(self.curriculum_phase).strip().lower()
        if phase not in {"full", "approach", "contact", "probe", "lift"}:
            raise ValueError("curriculum_phase must be one of: full, approach, contact, probe, lift.")
        self.curriculum_phase = phase
        if self.curriculum_face_margin_m < 0.0:
            raise ValueError("curriculum_face_margin_m must be >= 0.")
        if len(self.arm_action_joint_names) != ARM_ACT_DIM:
            raise ValueError(f"arm_action_joint_names must contain exactly {ARM_ACT_DIM} joint names.")
        if int(self.lift_settle_steps) < 0:
            raise ValueError("lift_settle_steps must be >= 0.")
        if float(self.lift_arm_stiffness) < 0.0:
            raise ValueError("lift_arm_stiffness must be >= 0.")
        if float(self.lift_arm_damping) < 0.0:
            raise ValueError("lift_arm_damping must be >= 0.")
        if len(self.lift_payload_mass_levels_kg) == 0:
            raise ValueError("lift_payload_mass_levels_kg must contain at least one mass value.")
        if any(float(mass_kg) <= 0.0 for mass_kg in self.lift_payload_mass_levels_kg):
            raise ValueError("lift_payload_mass_levels_kg values must be > 0.")
        if int(self.lift_payload_mass_level) < 0 or int(self.lift_payload_mass_level) >= len(self.lift_payload_mass_levels_kg):
            raise ValueError("lift_payload_mass_level must index into lift_payload_mass_levels_kg.")
        if float(self.lift_contact_preload_m) < 0.0:
            raise ValueError("lift_contact_preload_m must be >= 0.")
        if float(self.lift_contact_vertical_preload_m) < 0.0:
            raise ValueError("lift_contact_vertical_preload_m must be >= 0.")
        self.sim.device = str(self.device)
        if not self.possible_agents:
            self.possible_agents = _agent_names(self.num_robots)
        if not self.robot_prim_paths:
            self.robot_prim_paths = _robot_prim_paths(self.num_robots)
        if not self.observation_spaces or not self.action_spaces or not self.shared_observation_spaces:
            obs, act, shared = _build_agent_spaces(self.possible_agents)
            self.observation_spaces = obs
            self.action_spaces = act
            self.shared_observation_spaces = shared


class NoVlmCoopTransportDirectEnv(DirectMARLEnv):
    """Skeleton DirectMARLEnv with explicit MAPPO critic state and torch phase runtime."""

    cfg: NoVlmCoopTransportEnvCfg

    def __init__(self, cfg: NoVlmCoopTransportEnvCfg, **kwargs):
        self.cfg = cfg
        self.possible_agents = list(cfg.possible_agents)
        self._num_envs = int(cfg.scene.num_envs)
        self._device = torch.device(cfg.device)
        self._curriculum_phase = str(cfg.curriculum_phase).strip().lower()
        self._belief_ekf: List[BeliefEKF] = []
        self._neural_cbf_runtime: Optional[NeuralCbfRuntime] = None
        self._residual_runtime: Optional[ResidualCorrectorRuntime] = None
        self._agent_index = {name: idx for idx, name in enumerate(self.possible_agents)}
        self._last_actions = {
            agent: torch.zeros((self._num_envs, AGENT_ACT_DIM), dtype=torch.float32, device=self._device)
            for agent in self.possible_agents
        }
        self._phase_mgr = TorchBatchedNoVlmPhaseManager(
            num_envs=self._num_envs,
            cfg=PhaseConfig(include_correct_phase=False),
            device=self._device,
        )
        self._attachment_backend = AutoAttachmentBackend()
        self._attachment_runtime_available = bool(self._attachment_backend.is_available())
        self._attachments: dict[tuple[int, str], str] = {}
        self._pending_lift_attachments: list[tuple[int, str]] = []
        self._pending_lift_attachment_wait_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._attachment_count_buf = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        self._attachment_mask = torch.zeros(
            (self._num_envs, len(self.possible_agents)), dtype=torch.bool, device=self._device
        )
        self._robot_entities: dict[str, Articulation] = {}
        self._payload_entity: RigidObject | None = None
        self._contact_sensors: dict[str, object] = {}
        self._goal_xy = torch.tensor([2.5, 0.0], dtype=torch.float32, device=self._device).view(1, 2).repeat(
            self._num_envs, 1
        )
        self._command_cache = {
            agent: torch.zeros((self._num_envs, BASE_CMD_DIM), dtype=torch.float32, device=self._device)
            for agent in self.possible_agents
        }
        self._base_joint_ids: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_joint_ids: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_action_target_indices: dict[str, torch.Tensor | None] = {
            agent: None for agent in self.possible_agents
        }
        self._arm_hold_targets: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_default_targets: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._base_contact_force_n = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._all_attached_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._contact_force_cache: Dict[str, torch.Tensor] = {}
        self._payload_reset_z = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._payload_mass_kg = torch.full(
            (self._num_envs,),
            float(self._active_payload_mass_kg()),
            dtype=torch.float32,
            device=self._device,
        )
        self._lift_settle_baseline_captured = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._belief_mu = torch.zeros((self._num_envs, 7), dtype=torch.float32, device=self._device)
        self._belief_cov_diag = torch.zeros((self._num_envs, 7), dtype=torch.float32, device=self._device)
        self._belief_uncertainty = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device)
        self._belief_force_ema = torch.zeros(
            (self._num_envs, len(self.possible_agents)), dtype=torch.float32, device=self._device
        )
        self._belief_dirty = True
        (
            self._active_use_cbf,
            self._active_cbf_enable_after_s,
            self._active_cbf_warmup_s,
        ) = self._resolve_active_cbf_schedule()
        self._cbf_slack = torch.zeros((self._num_envs, len(self.possible_agents)), dtype=torch.float32, device=self._device)
        self._cbf_applied = torch.zeros((self._num_envs, len(self.possible_agents)), dtype=torch.bool, device=self._device)
        self._neural_barrier = torch.zeros((self._num_envs, len(self.possible_agents)), dtype=torch.float32, device=self._device)
        self._residual_goal_offset = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device)
        self._debug_last_belief_step = -1
        self._debug_last_cbf_step = -1
        self._init_belief_runtime()
        self._init_model_runtime()
        super().__init__(cfg=cfg, **kwargs)
        self._refresh_entity_buffers()
        self._ensure_joint_groups_configured()

    def _init_belief_runtime(self) -> None:
        self._belief_ekf = []
        dt = float(self.cfg.sim.dt) * float(max(1, int(self.cfg.decimation)))
        dims = np.asarray(self.cfg.payload_dims_m, dtype=np.float32)
        for env_id in range(self._num_envs):
            ekf = BeliefEKF(
                dt=dt,
                meas_force_frac=float(self.cfg.belief_meas_force_frac),
                meas_total_frac=float(self.cfg.belief_meas_total_frac),
            )
            ekf.initialize(
                mass_kg=float(self._payload_mass_kg[env_id].item()),
                com_offset_xyz=(0.0, 0.0, 0.0),
                dims_xyz=dims,
            )
            self._belief_ekf.append(ekf)
        self._sync_belief_tensors()

    def _init_model_runtime(self) -> None:
        if bool(self.cfg.use_neural_cbf):
            self._neural_cbf_runtime = NeuralCbfRuntime(
                mu_dim=7,
                hidden=int(self.cfg.neural_cbf_hidden),
                model_path=str(self.cfg.neural_cbf_model_path or ""),
                device=str(self._device),
            )
        residual_path = str(self.cfg.residual_model_path or "").strip()
        if residual_path:
            self._residual_runtime = ResidualCorrectorRuntime(residual_path)

    def _sync_belief_tensors(self) -> None:
        if not self._belief_ekf:
            return
        mu = np.stack([ekf.mean() for ekf in self._belief_ekf], axis=0).astype(np.float32)
        cov_diag = np.stack([np.diag(ekf.covariance()) for ekf in self._belief_ekf], axis=0).astype(np.float32)
        unc = np.asarray([ekf.risk_components() for ekf in self._belief_ekf], dtype=np.float32)
        self._belief_mu = torch.as_tensor(mu, dtype=torch.float32, device=self._device)
        self._belief_cov_diag = torch.as_tensor(cov_diag, dtype=torch.float32, device=self._device)
        self._belief_uncertainty = torch.as_tensor(unc, dtype=torch.float32, device=self._device)

    def _setup_scene(self) -> None:
        # Mandatory for replicated environments.
        self.scene.clone_environments(copy_from_source=False)

        # Recommended on CPU to avoid cross-env collision artifacts.
        if str(self._device).startswith("cpu") and hasattr(self.scene, "filter_collisions"):
            self.scene.filter_collisions()

        self._robot_entities = {}
        for idx, agent_name in enumerate(self.possible_agents):
            key = f"robot_{idx}"
            robot = None
            try:
                if hasattr(self.scene, "__getitem__"):
                    robot = self.scene[key]
            except Exception:
                robot = None
            if robot is None and hasattr(self.scene, "articulations"):
                robot = self.scene.articulations.get(key)
            if robot is None:
                raise RuntimeError(
                    f"Scene entity '{key}' not found. Declare it as ArticulationCfg on env cfg."
                )
            self._robot_entities[agent_name] = robot

        payload = None
        try:
            if hasattr(self.scene, "__getitem__"):
                payload = self.scene["payload"]
        except Exception:
            payload = None
        if payload is None and hasattr(self.scene, "rigid_objects"):
            payload = self.scene.rigid_objects.get("payload")
        self._payload_entity = payload
        self._contact_sensors = {}
        if hasattr(self.scene, "sensors"):
            for idx, agent_name in enumerate(self.possible_agents):
                sensor = self.scene.sensors.get(f"contact_robot_{idx}")
                if sensor is not None:
                    self._contact_sensors[agent_name] = sensor
        self._configure_joint_groups()

    def _joint_names(self, entity: Articulation) -> list[str]:
        if hasattr(entity, "joint_names"):
            names = getattr(entity, "joint_names")
            if names is not None:
                return [str(name) for name in names]
        if hasattr(entity, "data") and hasattr(entity.data, "joint_names"):
            names = getattr(entity.data, "joint_names")
            if names is not None:
                return [str(name) for name in names]
        return []

    def _joint_count(self, entity: Articulation) -> int:
        if hasattr(entity, "data") and hasattr(entity.data, "default_joint_pos"):
            joint_pos = entity.data.default_joint_pos
            if isinstance(joint_pos, torch.Tensor) and joint_pos.ndim == 2:
                return int(joint_pos.shape[1])
        names = self._joint_names(entity)
        return len(names)

    def _select_joint_ids(
        self,
        entity: Articulation,
        include_tokens: Sequence[str],
        exclude_tokens: Sequence[str] = (),
    ) -> torch.Tensor | None:
        names = self._joint_names(entity)
        if not names:
            return None
        include = [token.lower() for token in include_tokens]
        exclude = [token.lower() for token in exclude_tokens]
        ids: list[int] = []
        for idx, name in enumerate(names):
            lname = name.lower()
            if include and not any(token in lname for token in include):
                continue
            if exclude and any(token in lname for token in exclude):
                continue
            ids.append(idx)
        if not ids:
            return None
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    def _default_joint_targets(self, entity: Articulation, joint_ids: torch.Tensor | None) -> torch.Tensor | None:
        if joint_ids is None or joint_ids.numel() == 0:
            return None
        default_joint_pos = None
        if hasattr(entity, "data") and hasattr(entity.data, "default_joint_pos"):
            joint_pos = entity.data.default_joint_pos
            if isinstance(joint_pos, torch.Tensor) and joint_pos.ndim == 2:
                default_joint_pos = joint_pos.to(device=self._device, dtype=torch.float32)
        if default_joint_pos is None:
            return torch.zeros((self._num_envs, int(joint_ids.numel())), dtype=torch.float32, device=self._device)

        idx = joint_ids.detach().to(dtype=torch.long, device=default_joint_pos.device)
        idx_max = int(torch.max(idx).item())
        if default_joint_pos.shape[1] <= idx_max:
            return torch.zeros((self._num_envs, int(joint_ids.numel())), dtype=torch.float32, device=self._device)

        base_row = default_joint_pos[0, idx]
        return base_row.unsqueeze(0).repeat(self._num_envs, 1).contiguous()

    def _configure_joint_groups(self) -> None:
        for agent_name, robot in self._robot_entities.items():
            base_ids = self._select_joint_ids(robot, include_tokens=("wheel",), exclude_tokens=("finger",))
            if base_ids is None:
                joint_count = self._joint_count(robot)
                if joint_count >= 4:
                    base_ids = torch.arange(4, dtype=torch.long, device=self._device)

            arm_ids = self._select_joint_ids(
                robot,
                include_tokens=("panda_joint", "arm", "shoulder", "elbow", "wrist"),
                exclude_tokens=("wheel", "finger"),
            )
            if arm_ids is None:
                joint_count = self._joint_count(robot)
                if joint_count > 0:
                    reserved = set()
                    if base_ids is not None:
                        reserved.update(base_ids.detach().to(device="cpu", dtype=torch.long).tolist())
                    remaining = [idx for idx in range(joint_count) if idx not in reserved]
                    if remaining:
                        arm_ids = torch.tensor(remaining, dtype=torch.long, device=self._device)

            self._base_joint_ids[agent_name] = base_ids
            self._arm_joint_ids[agent_name] = arm_ids
            action_target_indices = None
            if arm_ids is not None and arm_ids.numel() > 0:
                joint_names = self._joint_names(robot)
                arm_id_list = arm_ids.detach().to(device="cpu", dtype=torch.long).tolist()
                selected_indices: list[int] = []
                for desired_name in self.cfg.arm_action_joint_names:
                    desired = str(desired_name).strip().lower()
                    match_idx = None
                    for offset_idx, joint_id in enumerate(arm_id_list):
                        if joint_id < len(joint_names) and joint_names[joint_id].strip().lower() == desired:
                            match_idx = offset_idx
                            break
                    if match_idx is not None:
                        selected_indices.append(int(match_idx))
                if len(selected_indices) == ARM_ACT_DIM:
                    action_target_indices = torch.tensor(selected_indices, dtype=torch.long, device=self._device)
            if action_target_indices is None and arm_ids is not None and arm_ids.numel() > 0:
                fallback_count = min(int(arm_ids.numel()), ARM_ACT_DIM)
                action_target_indices = torch.arange(fallback_count, dtype=torch.long, device=self._device)
            self._arm_action_target_indices[agent_name] = action_target_indices
            default_targets = self._default_joint_targets(robot, arm_ids)
            self._arm_default_targets[agent_name] = default_targets
            self._arm_hold_targets[agent_name] = default_targets

    def _ensure_joint_groups_configured(self) -> None:
        if not self._robot_entities:
            return
        needs_config = False
        for agent_name in self.possible_agents:
            if self._base_joint_ids.get(agent_name) is None:
                needs_config = True
                break
            if self._arm_joint_ids.get(agent_name) is None:
                needs_config = True
                break
            if self._arm_action_target_indices.get(agent_name) is None:
                needs_config = True
                break
            if self._arm_hold_targets.get(agent_name) is None:
                needs_config = True
                break
        if needs_config:
            self._configure_joint_groups()

    def _set_joint_velocity_target(self, robot: Articulation, targets: torch.Tensor, joint_ids: torch.Tensor) -> None:
        if not hasattr(robot, "set_joint_velocity_target"):
            return
        try:
            robot.set_joint_velocity_target(targets, joint_ids=joint_ids)
        except Exception:
            robot.set_joint_velocity_target(
                targets,
                joint_ids=joint_ids.detach().to(device="cpu", dtype=torch.long).tolist(),
            )

    def _set_joint_position_target(self, robot: Articulation, targets: torch.Tensor, joint_ids: torch.Tensor) -> None:
        if not hasattr(robot, "set_joint_position_target"):
            return
        try:
            robot.set_joint_position_target(targets, joint_ids=joint_ids)
        except Exception:
            robot.set_joint_position_target(
                targets,
                joint_ids=joint_ids.detach().to(device="cpu", dtype=torch.long).tolist(),
            )

    def _refresh_entity_buffers(self) -> None:
        # In Isaac Lab, data buffers are valid after at least one sim update.
        # Refreshing here prevents stale/zero reads when observation code runs early.
        sim_dt = float(getattr(self, "physics_dt", float(self.cfg.sim.dt)))
        for robot in self._robot_entities.values():
            if hasattr(robot, "update"):
                robot.update(sim_dt)
        if self._payload_entity is not None and hasattr(self._payload_entity, "update"):
            self._payload_entity.update(sim_dt)

    def _lift_uses_contact_only(self) -> bool:
        return self._curriculum_phase == "lift" and bool(getattr(self.cfg, "lift_contact_only", True))

    def _selected_lift_payload_mass_kg(self) -> float:
        levels = tuple(float(mass_kg) for mass_kg in self.cfg.lift_payload_mass_levels_kg)
        level_idx = int(np.clip(int(self.cfg.lift_payload_mass_level), 0, len(levels) - 1))
        return float(levels[level_idx])

    def _active_payload_mass_kg(self) -> float:
        if self._curriculum_phase == "lift":
            return self._selected_lift_payload_mass_kg()
        return float(self.cfg.object_mass_nominal_kg)

    def _set_payload_mass(self, env_ids: torch.Tensor, mass_kg: float) -> None:
        if self._payload_entity is None or env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        target_mass = float(max(1.0e-6, mass_kg))
        masses = self._payload_entity.root_physx_view.get_masses().clone()
        masses[env_ids_cpu] = target_mass
        self._payload_entity.root_physx_view.set_masses(masses, env_ids_cpu)

        default_mass = getattr(self._payload_entity.data, "default_mass", None)
        default_inertia = getattr(self._payload_entity.data, "default_inertia", None)
        if isinstance(default_mass, torch.Tensor) and isinstance(default_inertia, torch.Tensor):
            inertias = self._payload_entity.root_physx_view.get_inertias().clone()
            ratios = masses[env_ids_cpu] / torch.clamp(default_mass[env_ids_cpu], min=1.0e-6)
            inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratios
            self._payload_entity.root_physx_view.set_inertias(inertias, env_ids_cpu)
        self._payload_mass_kg[env_ids] = target_mass

    def _safe_contact_force_norm(self, entity) -> torch.Tensor:
        if entity is not None:
            for agent_name, robot in self._robot_entities.items():
                if robot is entity:
                    sensor = self._contact_sensors.get(agent_name)
                    if sensor is not None and hasattr(sensor, "data"):
                        data = sensor.data
                        for attr in ("net_forces_w", "net_forces_w_history", "force_matrix_w"):
                            if not hasattr(data, attr):
                                continue
                            tensor = getattr(data, attr)
                            if not isinstance(tensor, torch.Tensor):
                                continue
                            t = tensor.to(device=self._device, dtype=torch.float32)
                            if t.ndim == 4 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                                vec = t[: self._num_envs, ...]
                                norms = torch.linalg.vector_norm(vec, dim=-1)
                                return torch.amax(norms.reshape(self._num_envs, -1), dim=1)
                            if t.ndim == 3 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                                vec = t[: self._num_envs, :, :3]
                                norms = torch.linalg.vector_norm(vec, dim=-1)
                                return torch.max(norms, dim=1).values
        if entity is None or not hasattr(entity, "data"):
            return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        data = entity.data
        for attr in (
            "net_contact_forces_w",
            "net_contact_forces",
            "body_net_contact_forces_w",
            "body_net_forces_w",
            "contact_forces",
        ):
            if not hasattr(data, attr):
                continue
            tensor = getattr(data, attr)
            if not isinstance(tensor, torch.Tensor):
                continue
            t = tensor.to(device=self._device, dtype=torch.float32)
            if t.ndim == 3 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                vec = t[: self._num_envs, :, :3]
                norms = torch.linalg.vector_norm(vec, dim=-1)
                return torch.max(norms, dim=1).values
            if t.ndim == 2 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                return torch.linalg.vector_norm(t[: self._num_envs, :3], dim=-1)
            if t.ndim == 1 and t.shape[0] >= self._num_envs:
                return t[: self._num_envs]
        return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

    def _safe_contact_force_z(self, agent_name: str) -> torch.Tensor:
        """Return a per-env vertical support-force proxy for the robot hand.

        The belief EKF predicts per-robot vertical support loads. Feeding the
        3D contact-force magnitude overstates mass during approach and early
        contact because the dominant contact component is often horizontal.
        """
        sensor = self._contact_sensors.get(agent_name)
        if sensor is not None and hasattr(sensor, "data"):
            data = sensor.data
            for attr in ("net_forces_w", "net_forces_w_history"):
                if not hasattr(data, attr):
                    continue
                tensor = getattr(data, attr)
                if not isinstance(tensor, torch.Tensor):
                    continue
                t = tensor.to(device=self._device, dtype=torch.float32)
                if t.ndim == 4 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                    fz = t[: self._num_envs, -1, :, 2]
                    return torch.amax(torch.clamp(fz, min=0.0), dim=1)
                if t.ndim == 3 and t.shape[0] >= self._num_envs and t.shape[-1] >= 3:
                    fz = t[: self._num_envs, :, 2]
                    return torch.amax(torch.clamp(fz, min=0.0), dim=1)
        robot = self._robot_entities.get(agent_name)
        raw = self._safe_contact_force_norm(robot)
        max_force = self._payload_mass_kg * (9.81 / float(max(1, self.cfg.num_robots)))
        return torch.minimum(raw, max_force * 2.0)

    def _robot_payload_distances(self) -> Dict[str, torch.Tensor]:
        payload_xy = self._payload_features()[:, 0:2]
        out: Dict[str, torch.Tensor] = {}
        for name, robot in self._robot_entities.items():
            base_xy = self._robot_base_features(robot)[:, 0:2]
            out[name] = torch.linalg.vector_norm(base_xy - payload_xy, dim=1)
        return out

    def _attachment_counts(self) -> torch.Tensor:
        return self._attachment_count_buf

    def _env_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.scene, "env_origins"):
            return self.scene.env_origins[env_ids].to(device=self._device, dtype=torch.float32)
        return torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self._device)

    def _reset_articulation(self, entity: Articulation, env_ids: torch.Tensor) -> None:
        if not hasattr(entity, "data"):
            return
        data = entity.data
        if not hasattr(data, "default_root_state"):
            return
        default_root = data.default_root_state.to(device=self._device, dtype=torch.float32)
        if default_root.shape[0] == 1 and env_ids.numel() > 1:
            root_state = default_root.repeat(env_ids.numel(), 1)
        else:
            root_state = default_root[env_ids].clone()
        root_state[:, :3] += self._env_origins(env_ids)
        if hasattr(entity, "write_root_pose_to_sim"):
            entity.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if hasattr(entity, "write_root_velocity_to_sim"):
            entity.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        if (
            hasattr(data, "default_joint_pos")
            and hasattr(data, "default_joint_vel")
            and hasattr(entity, "write_joint_state_to_sim")
        ):
            default_joint_pos = data.default_joint_pos.to(device=self._device, dtype=torch.float32)
            default_joint_vel = data.default_joint_vel.to(device=self._device, dtype=torch.float32)
            if default_joint_pos.shape[0] == 1 and env_ids.numel() > 1:
                joint_pos = default_joint_pos.repeat(env_ids.numel(), 1)
                joint_vel = default_joint_vel.repeat(env_ids.numel(), 1)
            else:
                joint_pos = default_joint_pos[env_ids].clone()
                joint_vel = default_joint_vel[env_ids].clone()
            entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if hasattr(entity, "write_data_to_sim"):
            entity.write_data_to_sim()

    def _reset_rigid_object(self, entity: RigidObject, env_ids: torch.Tensor) -> None:
        if not hasattr(entity, "data") or not hasattr(entity.data, "default_root_state"):
            return
        default_root = entity.data.default_root_state.to(device=self._device, dtype=torch.float32)
        if default_root.shape[0] == 1 and env_ids.numel() > 1:
            root_state = default_root.repeat(env_ids.numel(), 1)
        else:
            root_state = default_root[env_ids].clone()
        root_state[:, :3] += self._env_origins(env_ids)
        if hasattr(entity, "write_root_pose_to_sim"):
            entity.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if hasattr(entity, "write_root_velocity_to_sim"):
            entity.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

    def _clear_env_attachments(self, env_ids: torch.Tensor) -> None:
        env_set = set(env_ids.detach().to(device="cpu", dtype=torch.long).tolist())
        to_remove = [key for key in self._attachments.keys() if int(key[0]) in env_set]
        for key in to_remove:
            path = self._attachments.pop(key)
            self._attachment_backend.detach(path)
        if self._pending_lift_attachments:
            self._pending_lift_attachments = [
                item for item in self._pending_lift_attachments if int(item[0]) not in env_set
            ]
        self._pending_lift_attachment_wait_steps[env_ids] = 0
        self._attachment_count_buf[env_ids] = 0
        self._attachment_mask[env_ids] = False

    def _capture_payload_reset_height(self, env_ids: torch.Tensor) -> None:
        if self._payload_entity is None or env_ids.numel() == 0:
            return
        payload_root = self._safe_root_state(self._payload_entity)
        if payload_root.ndim != 2 or payload_root.shape[0] <= int(torch.max(env_ids).item()):
            return
        self._payload_reset_z[env_ids] = payload_root[env_ids, 2].to(device=self._device, dtype=torch.float32)
        settle_steps = max(0, int(getattr(self.cfg, "lift_settle_steps", 0)))
        if self._curriculum_phase == "lift" and settle_steps > 0:
            self._lift_settle_baseline_captured[env_ids] = False
        else:
            self._lift_settle_baseline_captured[env_ids] = True

    def _maybe_capture_lift_settled_baseline(self) -> None:
        if self._curriculum_phase != "lift" or self._payload_entity is None:
            return
        settle_steps = max(0, int(getattr(self.cfg, "lift_settle_steps", 0)))
        if settle_steps <= 0:
            self._lift_settle_baseline_captured[:] = True
            return
        if not hasattr(self, "episode_length_buf"):
            return
        steps = self.episode_length_buf.to(device=self._device, dtype=torch.long)
        ready_mask = (~self._lift_settle_baseline_captured) & (steps >= settle_steps)
        if not torch.any(ready_mask):
            return
        payload_root = self._safe_root_state(self._payload_entity)
        self._payload_reset_z[ready_mask] = payload_root[ready_mask, 2].to(device=self._device, dtype=torch.float32)
        self._lift_settle_baseline_captured[ready_mask] = True

    def _lift_progress_state(self, payload_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_capture_lift_settled_baseline()
        lift_delta_z = torch.clamp(payload_z - self._payload_reset_z, min=0.0)
        if self._curriculum_phase == "lift":
            ready_mask = self._lift_settle_baseline_captured.clone()
        else:
            ready_mask = torch.ones_like(lift_delta_z, dtype=torch.bool, device=self._device)
        return lift_delta_z, ready_mask

    def _reset_belief_idx(self, env_ids: torch.Tensor) -> None:
        dims = np.asarray(self.cfg.payload_dims_m, dtype=np.float32)
        for env_id in env_ids.detach().to(device="cpu", dtype=torch.long).tolist():
            ekf = BeliefEKF(
                dt=float(self.cfg.sim.dt) * float(max(1, int(self.cfg.decimation))),
                meas_force_frac=float(self.cfg.belief_meas_force_frac),
                meas_total_frac=float(self.cfg.belief_meas_total_frac),
            )
            ekf.initialize(
                mass_kg=float(self._payload_mass_kg[int(env_id)].item()),
                com_offset_xyz=(0.0, 0.0, 0.0),
                dims_xyz=dims,
            )
            self._belief_ekf[int(env_id)] = ekf
        self._sync_belief_tensors()

    def _refresh_attachment_runtime_available(self) -> bool:
        self._attachment_runtime_available = bool(self._attachment_backend.is_available())
        return self._attachment_runtime_available

    def _resolve_active_cbf_schedule(self) -> tuple[bool, float, float]:
        raw_enable_after = max(0.0, float(self.cfg.cbf_enable_after_s))
        raw_warmup = max(0.0, float(self.cfg.cbf_warmup_s))
        if not bool(self.cfg.use_cbf):
            return False, raw_enable_after, raw_warmup
        if not bool(getattr(self.cfg, "curriculum_use_stage_cbf_schedule", True)):
            return True, raw_enable_after, raw_warmup
        if self._curriculum_phase == "lift":
            return False, 0.0, 0.0
        if self._curriculum_phase == "probe":
            return True, 0.5, 1.5
        if self._curriculum_phase == "contact":
            return True, 0.25, 1.0
        return True, 0.0, 1.0

    def _yaw_to_quaternion(self, yaw: torch.Tensor) -> torch.Tensor:
        half = 0.5 * yaw.to(device=self._device, dtype=torch.float32)
        quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=self._device)
        quat[:, 0] = torch.cos(half)
        quat[:, 3] = torch.sin(half)
        return quat

    def _write_root_pose_xy_yaw(
        self,
        entity,
        env_ids: torch.Tensor,
        xy_world: torch.Tensor,
        yaw_world: torch.Tensor,
    ) -> None:
        if entity is None or env_ids.numel() == 0:
            return
        root_state = self._safe_root_state(entity)[env_ids].clone()
        root_state[:, 0:2] = xy_world.to(device=self._device, dtype=torch.float32)
        root_state[:, 3:7] = self._yaw_to_quaternion(yaw_world.reshape(-1))
        root_state[:, 7:13] = 0.0
        if hasattr(entity, "write_root_pose_to_sim"):
            entity.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if hasattr(entity, "write_root_velocity_to_sim"):
            entity.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        if hasattr(entity, "write_data_to_sim"):
            entity.write_data_to_sim()

    def _write_root_pose_xyz(self, entity, env_ids: torch.Tensor, xyz_world: torch.Tensor) -> None:
        if entity is None or env_ids.numel() == 0:
            return
        root_state = self._safe_root_state(entity)[env_ids].clone()
        root_state[:, 0:3] = xyz_world.to(device=self._device, dtype=torch.float32)
        root_state[:, 7:13] = 0.0
        if hasattr(entity, "write_root_pose_to_sim"):
            entity.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if hasattr(entity, "write_root_velocity_to_sim"):
            entity.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
        if hasattr(entity, "write_data_to_sim"):
            entity.write_data_to_sim()

    def _hand_body_state(self, robot: Articulation) -> torch.Tensor | None:
        if robot is None or not hasattr(robot, "data") or not hasattr(robot.data, "body_state_w"):
            return None
        if not hasattr(robot, "find_bodies"):
            return None
        try:
            body_ids, _ = robot.find_bodies("panda_hand")
        except Exception:
            return None
        if not body_ids:
            return None
        body_state = robot.data.body_state_w
        if not isinstance(body_state, torch.Tensor) or body_state.ndim != 3:
            return None
        hand_idx = int(body_ids[0])
        if hand_idx >= int(body_state.shape[1]):
            return None
        return body_state[:, hand_idx, :].to(device=self._device, dtype=torch.float32)

    def _set_lift_arm_posture(self, env_ids: torch.Tensor) -> None:
        lift_joint_targets = {
            "panda_joint1": 0.0,
            "panda_joint2": float(self.cfg.lift_preshape_joint2_rad),
            "panda_joint3": 0.0,
            "panda_joint4": float(self.cfg.lift_preshape_joint4_rad),
            "panda_joint5": 0.0,
            "panda_joint6": float(self.cfg.lift_preshape_joint6_rad),
            "panda_joint7": 0.8,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
        for agent_name, robot in self._robot_entities.items():
            if robot is None or not hasattr(robot, "data") or not hasattr(robot.data, "default_joint_pos"):
                continue
            joint_names = self._joint_names(robot)
            default_joint_pos = robot.data.default_joint_pos.to(device=self._device, dtype=torch.float32)
            if default_joint_pos.shape[0] == 1 and env_ids.numel() > 1:
                joint_pos = default_joint_pos.repeat(env_ids.numel(), 1)
            else:
                joint_pos = default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            for joint_name, target in lift_joint_targets.items():
                try:
                    joint_idx = joint_names.index(joint_name)
                except ValueError:
                    continue
                joint_pos[:, joint_idx] = float(target)
            if hasattr(robot, "write_joint_state_to_sim"):
                robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            if hasattr(robot, "write_data_to_sim"):
                robot.write_data_to_sim()
            arm_ids = self._arm_joint_ids.get(agent_name)
            if arm_ids is not None and arm_ids.numel() > 0:
                arm_targets = joint_pos[:, arm_ids].clone()
                self._arm_default_targets[agent_name] = arm_targets
                self._arm_hold_targets[agent_name] = arm_targets

    def _set_lift_arm_gains(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        lift_stiffness = float(max(0.0, float(self.cfg.lift_arm_stiffness)))
        lift_damping = float(max(0.0, float(self.cfg.lift_arm_damping)))
        for agent_name, robot in self._robot_entities.items():
            arm_ids = self._arm_joint_ids.get(agent_name)
            if robot is None or arm_ids is None or arm_ids.numel() == 0:
                continue
            num_envs = int(env_ids.numel())
            num_joints = int(arm_ids.numel())
            stiffness = torch.full((num_envs, num_joints), lift_stiffness, dtype=torch.float32, device=self._device)
            damping = torch.full((num_envs, num_joints), lift_damping, dtype=torch.float32, device=self._device)
            if hasattr(robot, "write_joint_stiffness_to_sim"):
                robot.write_joint_stiffness_to_sim(stiffness, joint_ids=arm_ids, env_ids=env_ids)
            if hasattr(robot, "write_joint_damping_to_sim"):
                robot.write_joint_damping_to_sim(damping, joint_ids=arm_ids, env_ids=env_ids)
            if hasattr(robot, "data") and hasattr(robot.data, "default_joint_stiffness"):
                default_joint_stiffness = robot.data.default_joint_stiffness
                if isinstance(default_joint_stiffness, torch.Tensor) and default_joint_stiffness.ndim == 2:
                    joint_idx = arm_ids.detach().to(device=self._device, dtype=torch.long)
                    default_joint_stiffness[env_ids[:, None], joint_idx.view(1, -1)] = lift_stiffness
            if hasattr(robot, "data") and hasattr(robot.data, "default_joint_damping"):
                default_joint_damping = robot.data.default_joint_damping
                if isinstance(default_joint_damping, torch.Tensor) and default_joint_damping.ndim == 2:
                    joint_idx = arm_ids.detach().to(device=self._device, dtype=torch.long)
                    default_joint_damping[env_ids[:, None], joint_idx.view(1, -1)] = lift_damping
            actuators = getattr(robot, "actuators", None)
            if isinstance(actuators, dict):
                arm_actuator = actuators.get("arm_position")
                if arm_actuator is not None:
                    actuator_stiffness = getattr(arm_actuator, "stiffness", None)
                    if isinstance(actuator_stiffness, torch.Tensor) and actuator_stiffness.ndim == 2:
                        actuator_stiffness[env_ids] = stiffness.to(
                            device=actuator_stiffness.device,
                            dtype=actuator_stiffness.dtype,
                        )
                    actuator_damping = getattr(arm_actuator, "damping", None)
                    if isinstance(actuator_damping, torch.Tensor) and actuator_damping.ndim == 2:
                        actuator_damping[env_ids] = damping.to(
                            device=actuator_damping.device,
                            dtype=actuator_damping.dtype,
                        )

    def _align_payload_height_to_hands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or self._payload_entity is None:
            return
        hand_zs: list[torch.Tensor] = []
        for agent_name in self.possible_agents:
            hand_state = self._hand_body_state(self._robot_entities.get(agent_name))
            if hand_state is not None:
                hand_zs.append(hand_state[env_ids, 2])
        if not hand_zs:
            return
        target_payload_xyz = self._safe_root_state(self._payload_entity)[env_ids, 0:3].clone()
        mean_hand_z = torch.stack(hand_zs, dim=1).mean(dim=1)
        target_payload_xyz[:, 2] = mean_hand_z
        if self._lift_uses_contact_only():
            target_payload_xyz[:, 2] -= float(max(0.0, getattr(self.cfg, "lift_contact_vertical_preload_m", 0.0)))
        self._write_root_pose_xyz(self._payload_entity, env_ids, target_payload_xyz)

    def _align_lift_hands_to_payload_faces(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or self._payload_entity is None:
            return
        payload_xy = self._safe_root_state(self._payload_entity)[env_ids, 0:2]
        half_x = 0.5 * float(self.cfg.payload_dims_m[0])
        half_y = 0.5 * float(self.cfg.payload_dims_m[1])
        desired_offsets = torch.tensor(
            [
                [half_x, 0.0],
                [-half_x, 0.0],
                [0.0, half_y],
                [0.0, -half_y],
            ],
            dtype=torch.float32,
            device=self._device,
        )
        if self._lift_uses_contact_only():
            preload = float(max(0.0, getattr(self.cfg, "lift_contact_preload_m", 0.0)))
            if preload > 0.0:
                desired_offsets = desired_offsets.clone()
                for row_idx in range(int(desired_offsets.shape[0])):
                    if abs(float(desired_offsets[row_idx, 0].item())) > 0.0:
                        desired_offsets[row_idx, 0] = torch.sign(desired_offsets[row_idx, 0]) * max(
                            abs(float(desired_offsets[row_idx, 0].item())) - preload,
                            0.0,
                        )
                    if abs(float(desired_offsets[row_idx, 1].item())) > 0.0:
                        desired_offsets[row_idx, 1] = torch.sign(desired_offsets[row_idx, 1]) * max(
                            abs(float(desired_offsets[row_idx, 1].item())) - preload,
                            0.0,
                        )
        for agent_idx, agent_name in enumerate(self.possible_agents):
            robot = self._robot_entities.get(agent_name)
            hand_state = self._hand_body_state(robot)
            if robot is None or hand_state is None:
                continue
            root_state = self._safe_root_state(robot)[env_ids].clone()
            desired_hand_xy = payload_xy + desired_offsets[agent_idx].view(1, 2)
            delta_xy = desired_hand_xy - hand_state[env_ids, 0:2]
            root_state[:, 0:2] += delta_xy
            root_state[:, 7:13] = 0.0
            if hasattr(robot, "write_root_pose_to_sim"):
                robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
            if hasattr(robot, "write_data_to_sim"):
                robot.write_data_to_sim()

    def _layout_curriculum_robots(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or self._payload_entity is None:
            return
        payload_xy = self._safe_root_state(self._payload_entity)[env_ids, 0:2]
        half_x = 0.5 * float(self.cfg.payload_dims_m[0])
        half_y = 0.5 * float(self.cfg.payload_dims_m[1])
        margin = float(max(0.0, self.cfg.curriculum_face_margin_m))
        offsets = torch.tensor(
            [
                [half_x + margin, 0.0],
                [-(half_x + margin), 0.0],
                [0.0, half_y + margin],
                [0.0, -(half_y + margin)],
            ],
            dtype=torch.float32,
            device=self._device,
        )
        yaws = torch.tensor([np.pi, 0.0, -0.5 * np.pi, 0.5 * np.pi], dtype=torch.float32, device=self._device)
        for agent_idx, agent_name in enumerate(self.possible_agents):
            robot = self._robot_entities.get(agent_name)
            if robot is None:
                continue
            xy_world = payload_xy + offsets[agent_idx].view(1, 2)
            yaw_world = yaws[agent_idx].repeat(env_ids.numel())
            self._write_root_pose_xy_yaw(robot, env_ids, xy_world, yaw_world)
        if hasattr(self.scene, "write_data_to_sim"):
            self.scene.write_data_to_sim()

    def _apply_curriculum_reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        phase_name = self._curriculum_phase
        if phase_name in {"full", "approach"}:
            return

        if phase_name == "lift":
            self._set_lift_arm_posture(env_ids)
            self._set_lift_arm_gains(env_ids)
            self._layout_curriculum_robots(env_ids)
            self._refresh_entity_buffers()
            self._align_payload_height_to_hands(env_ids)
            self._refresh_entity_buffers()
            self._align_lift_hands_to_payload_faces(env_ids)
            self._refresh_entity_buffers()
            if not self._lift_uses_contact_only():
                if not self._refresh_attachment_runtime_available():
                    raise RuntimeError("Lift curriculum requires the PhysX attachment runtime to be available.")
                self._pending_lift_attachment_wait_steps[env_ids] = max(1, 2 * int(self.cfg.decimation))
                env_list = env_ids.detach().to(device="cpu", dtype=torch.long).tolist()
                for env_id in env_list:
                    for agent_name in self.possible_agents:
                        key = (int(env_id), str(agent_name))
                        if key not in self._pending_lift_attachments:
                            self._pending_lift_attachments.append(key)

        start_phase = {
            "contact": Phase.CONTACT,
            "probe": Phase.PROBE,
            "lift": Phase.LIFT,
        }.get(phase_name)
        if start_phase is None:
            return
        self._phase_mgr.phase_ids[env_ids] = int(PHASE_TO_ID[start_phase])
        self._phase_mgr.phase_started_at_s[env_ids] = 0.0
        self._phase_mgr.contact_all_attached_since_s[env_ids] = -1.0

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        self._all_attached_mask[env_ids] = False
        self._base_contact_force_n[env_ids] = 0.0
        self._attachment_count_buf[env_ids] = 0
        self._attachment_mask[env_ids] = False
        self._belief_force_ema[env_ids] = 0.0
        for robot in self._robot_entities.values():
            self._reset_articulation(robot, env_ids)
        if self._payload_entity is not None:
            self._reset_rigid_object(self._payload_entity, env_ids)
            self._set_payload_mass(env_ids, self._active_payload_mass_kg())
        self._clear_env_attachments(env_ids)
        self._reset_belief_idx(env_ids)
        self._cbf_slack[env_ids] = 0.0
        self._cbf_applied[env_ids] = False
        self._neural_barrier[env_ids] = 0.0
        self._residual_goal_offset[env_ids] = 0.0
        self._phase_mgr.reset(now_s=0.0, env_ids=env_ids.detach().to("cpu").tolist())
        self._apply_curriculum_reset(env_ids)
        self._capture_payload_reset_height(env_ids)
        self._belief_dirty = True

    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        for name in self.possible_agents:
            if name not in actions:
                raise KeyError(f"Missing action for agent: {name}")
            act = actions[name].to(device=self._device, dtype=torch.float32)
            if act.ndim != 2 or act.shape[1] != AGENT_ACT_DIM:
                raise ValueError(f"Action tensor for {name} must be [num_envs, {AGENT_ACT_DIM}]")
            self._last_actions[name] = torch.clamp(act, -1.0, 1.0)
        self._belief_dirty = True

    def _cbf_extra_constraint(
        self,
        env_id: int,
        agent_name: str,
        force_n: float,
        v_ref: np.ndarray,
        obj_vel_xy: np.ndarray,
        normal_xy: np.ndarray,
    ) -> Optional[tuple[np.ndarray, float, float]]:
        if self._neural_cbf_runtime is None:
            return None
        belief_mu = self._belief_mu[env_id].detach().to(device="cpu", dtype=torch.float32).numpy()
        belief_cov = np.diag(
            self._belief_cov_diag[env_id].detach().to(device="cpu", dtype=torch.float32).numpy()
        ).astype(np.float32)
        linear = self._neural_cbf_runtime.linearized_velocity_constraint(
            force_n=force_n,
            belief_mu=belief_mu,
            belief_cov=belief_cov,
            v_ref=v_ref,
            obj_vel_xy=obj_vel_xy,
            normal_xy=normal_xy,
            dt=float(self.cfg.sim.dt) * float(max(1, int(self.cfg.decimation))),
            force_vel_gain=4.0,
        )
        agent_idx = self.possible_agents.index(agent_name)
        self._neural_barrier[env_id, agent_idx] = float(linear.h_value)
        coeff = linear.grad_v.astype(np.float64)
        if float(np.linalg.norm(coeff)) <= 1.0e-6:
            return None
        rhs = float(np.dot(coeff, v_ref.astype(np.float64)) - linear.h_value)
        return coeff, rhs, float(linear.h_value)

    def _cbf_activation_scale(self, env_id: int) -> float:
        step_dt = float(self.cfg.sim.dt) * float(max(1, int(self.cfg.decimation)))
        elapsed_s = float(self.episode_length_buf[env_id].item()) * step_dt
        enable_after = max(0.0, float(self._active_cbf_enable_after_s))
        warmup = max(0.0, float(self._active_cbf_warmup_s))
        if elapsed_s <= enable_after:
            return 0.0
        if warmup <= 1.0e-6:
            return 1.0
        return float(np.clip((elapsed_s - enable_after) / warmup, 0.0, 1.0))

    def _maybe_debug_belief(self, force_arr: np.ndarray) -> None:
        max_steps = max(0, int(self.cfg.debug_belief_steps))
        if max_steps <= 0 or self._num_envs <= 0:
            return
        env_id = int(np.clip(int(self.cfg.debug_belief_env), 0, self._num_envs - 1))
        step_idx = int(self.episode_length_buf[env_id].item())
        if step_idx >= max_steps:
            return
        log_every = max(1, int(self.cfg.debug_belief_log_every))
        if step_idx == self._debug_last_belief_step or (step_idx % log_every) != 0:
            return
        self._debug_last_belief_step = step_idx
        ekf = self._belief_ekf[env_id]
        mu = ekf.mean().astype(np.float64)
        diag = np.diag(ekf.covariance()).astype(np.float64)
        com_u, mass_rel = ekf.risk_components()
        diag_info = ekf.diagnostics()
        forces = np.asarray(force_arr[env_id], dtype=np.float64).reshape(-1)
        print(
            (
                "[ekf] env={} step={} mass={:.3f} com=({:.4f},{:.4f}) "
                "nis={:.3f} com_u={:.5f} mass_rel={:.5f} forces={}"
            ).format(
                env_id,
                step_idx,
                float(mu[0]),
                float(mu[1]),
                float(mu[2]),
                float(diag_info.get("nis", 0.0)),
                float(com_u),
                float(mass_rel),
                np.array2string(forces, precision=3, suppress_small=False),
            ),
            flush=True,
        )
        print(
            (
                "[ekf] env={} step={} cov_diag_mass_com=({:.5f},{:.5f},{:.5f}) "
                "inertia_obs={}"
            ).format(
                env_id,
                step_idx,
                float(diag[0]),
                float(diag[1]),
                float(diag[2]),
                np.array2string(
                    np.asarray(diag_info.get("inertia_observability", np.zeros(3)), dtype=np.float64),
                    precision=5,
                    suppress_small=False,
                ),
            ),
            flush=True,
        )

    def _maybe_debug_cbf(
        self,
        env_id: int,
        agent_name: str,
        activation: float,
        v_des: np.ndarray,
        v_safe: np.ndarray,
        d_min_eff: float,
        alpha_eff: float,
        v_max_eff: float,
        result: CbfResult,
    ) -> None:
        max_steps = max(0, int(self.cfg.debug_cbf_steps))
        if max_steps <= 0 or self._num_envs <= 0:
            return
        debug_env = int(np.clip(int(self.cfg.debug_cbf_env), 0, self._num_envs - 1))
        if env_id != debug_env or agent_name != self.possible_agents[0]:
            return
        step_idx = int(self.episode_length_buf[env_id].item())
        if step_idx >= max_steps:
            return
        log_every = max(1, int(self.cfg.debug_cbf_log_every))
        if step_idx == self._debug_last_cbf_step or (step_idx % log_every) != 0:
            return
        self._debug_last_cbf_step = step_idx
        print(
            (
                "[cbf] env={} step={} activation={:.3f} d_min={:.4f} alpha={:.4f} "
                "v_max={:.4f} v_des={} v_safe={} slack={:.5f} applied={} recovery={}"
            ).format(
                env_id,
                step_idx,
                float(activation),
                float(d_min_eff),
                float(alpha_eff),
                float(v_max_eff),
                np.array2string(np.asarray(v_des, dtype=np.float64), precision=4, suppress_small=False),
                np.array2string(np.asarray(v_safe, dtype=np.float64), precision=4, suppress_small=False),
                float(result.slack),
                bool(np.linalg.norm(np.asarray(v_safe) - np.asarray(v_des)) > 1.0e-4),
                bool(result.recovery_active),
            ),
            flush=True,
        )

    def _apply_cbf_filter(self, commands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not bool(self._active_use_cbf):
            return commands
        filtered = {name: cmd.clone() for name, cmd in commands.items()}
        robot_state = {
            name: self._robot_base_features(self._robot_entities[name]).detach().to(device="cpu", dtype=torch.float32).numpy()
            for name in self.possible_agents
        }
        payload = self._payload_features().detach().to(device="cpu", dtype=torch.float32).numpy()
        belief_unc = self._belief_uncertainty.detach().to(device="cpu", dtype=torch.float32).numpy()
        for env_id in range(self._num_envs):
            activation = self._cbf_activation_scale(env_id)
            obj_vel_xy = payload[env_id, 3:5].astype(np.float64)
            for agent_idx, name in enumerate(self.possible_agents):
                if activation <= 0.0:
                    self._cbf_slack[env_id, agent_idx] = 0.0
                    self._cbf_applied[env_id, agent_idx] = False
                    continue
                me = robot_state[name][env_id]
                pos_i = me[0:2].astype(np.float64)
                neighbor_pos = []
                neighbor_vel = []
                for other_name in self.possible_agents:
                    if other_name == name:
                        continue
                    ot = robot_state[other_name][env_id]
                    neighbor_pos.append(ot[0:2].astype(np.float64))
                    neighbor_vel.append(ot[3:5].astype(np.float64))
                v_des = filtered[name][env_id, 0:2].detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
                normal = payload[env_id, 0:2] - me[0:2]
                nrm = float(np.linalg.norm(normal))
                if nrm <= 1.0e-6:
                    normal = np.array([1.0, 0.0], dtype=np.float64)
                else:
                    normal = (normal / nrm).astype(np.float64)
                extra_linear = []
                force_t = self._contact_force_cache.get(name)
                force_n = float(force_t[env_id].item()) if force_t is not None else 0.0
                extra = self._cbf_extra_constraint(env_id, name, force_n, v_des, obj_vel_xy, normal)
                if extra is not None:
                    coeff, rhs, _ = extra
                    extra_linear.append((coeff, rhs))
                com_u = float(belief_unc[env_id, 0]) if belief_unc.size else 0.0
                mass_rel = float(belief_unc[env_id, 1]) if belief_unc.size else 0.0
                d_min_target = float(self.cfg.separation_min_m) + float(self.cfg.cbf_dmin_com_gain) * com_u
                d_min_eff = max(1.0e-4, activation * d_min_target)
                alpha_target = float(self.cfg.cbf_alpha) / max(
                    1.0, 1.0 + float(self.cfg.cbf_alpha_mass_gain) * mass_rel
                )
                alpha_eff = max(1.0e-4, activation * alpha_target)
                v_max_target = float(self.cfg.speed_limit_mps) / max(
                    1.0, 1.0 + float(self.cfg.cbf_speed_mass_gain) * mass_rel
                )
                v_max_eff = (1.0 - activation) * float(self.cfg.speed_limit_mps) + activation * v_max_target
                result = solve_cbf_qp(
                    v_des=v_des,
                    pos_i=pos_i,
                    neighbor_pos=neighbor_pos,
                    neighbor_vel=neighbor_vel,
                    v_max=float(v_max_eff),
                    d_min=float(d_min_eff),
                    alpha=float(alpha_eff),
                    slack_weight=float(self.cfg.cbf_slack_weight),
                    slack_max=float(self.cfg.cbf_slack_max),
                    slack_threshold=float(self.cfg.cbf_slack_threshold),
                    recovery_gain=float(self.cfg.cbf_recovery_gain),
                    extra_linear=extra_linear or None,
                )
                v_safe = np.asarray(result.v_safe, dtype=np.float64)
                if result.slack_exceeded:
                    v_safe = np.zeros(2, dtype=np.float64)
                self._maybe_debug_cbf(
                    env_id=env_id,
                    agent_name=name,
                    activation=activation,
                    v_des=v_des,
                    v_safe=v_safe,
                    d_min_eff=d_min_eff,
                    alpha_eff=alpha_eff,
                    v_max_eff=v_max_eff,
                    result=result,
                )
                filtered[name][env_id, 0:2] = torch.tensor(v_safe, dtype=torch.float32, device=self._device)
                self._cbf_slack[env_id, agent_idx] = float(result.slack)
                self._cbf_applied[env_id, agent_idx] = bool(
                    np.linalg.norm(v_safe - v_des) > 1.0e-4
                )
        return filtered

    def _apply_action(self) -> None:
        # Base action mapping:
        # [vx_norm, vy_norm, yaw_norm] -> [m/s, m/s, rad/s], then to wheel rad/s.
        # Arm action mapping:
        # [d_joint1_norm, d_joint2_norm, d_joint3_norm] -> additive joint targets.
        speed_limit = float(self.cfg.speed_limit_mps)
        yaw_limit = float(self.cfg.yaw_rate_limit_radps)
        wheel_radius = float(max(self.cfg.wheel_radius_m, 1.0e-4))
        yaw_coupling = float(max(self.cfg.mecanum_yaw_coupling_m, 0.0))
        arm_delta_limit = float(max(self.cfg.arm_delta_limit_rad, 0.0))
        command_dict: Dict[str, torch.Tensor] = {}
        lift_hold_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        if self._curriculum_phase == "lift":
            lift_hold_mask = ~self._lift_settle_baseline_captured
        for name in self.possible_agents:
            act = self._last_actions[name]
            base_act = act[:, :BASE_ACT_DIM]
            cmd = torch.empty((self._num_envs, BASE_CMD_DIM), dtype=torch.float32, device=self._device)
            cmd[:, 0] = base_act[:, 0] * speed_limit
            cmd[:, 1] = base_act[:, 1] * speed_limit
            cmd[:, 2] = base_act[:, 2] * yaw_limit
            if torch.any(lift_hold_mask):
                cmd[lift_hold_mask] = 0.0
            command_dict[name] = cmd

        self._refresh_entity_buffers()
        self._ensure_joint_groups_configured()
        if self._pending_lift_attachments:
            pending = list(self._pending_lift_attachments)
            delayed_env_ids: set[int] = set()
            ready: list[tuple[int, str]] = []
            still_pending: list[tuple[int, str]] = []
            for env_id, agent_name in pending:
                env_idx = int(env_id)
                if int(self._pending_lift_attachment_wait_steps[env_idx].item()) > 0:
                    still_pending.append((env_idx, agent_name))
                    delayed_env_ids.add(env_idx)
                else:
                    ready.append((env_idx, agent_name))
            for env_id in delayed_env_ids:
                self._pending_lift_attachment_wait_steps[env_id] = torch.clamp(
                    self._pending_lift_attachment_wait_steps[env_id] - 1,
                    min=0,
                )
            self._pending_lift_attachments = still_pending
            try:
                if ready:
                    self._refresh_entity_buffers()
                for env_id, agent_name in ready:
                    self._attach_vacuum(
                        int(env_id),
                        agent_name,
                        self._robot_hand_prim_path(int(env_id), agent_name),
                        self._payload_prim_path(int(env_id)),
                    )
            except Exception as exc:
                self._pending_lift_attachments = pending
                raise RuntimeError(f"Failed to attach deferred lift curriculum welds: {exc}") from exc
        self._phase_inputs()
        self._update_belief_state()
        self._update_residual_goal_offset()
        command_dict = self._apply_cbf_filter(command_dict)

        for name in self.possible_agents:
            act = self._last_actions[name]
            arm_act = act[:, BASE_ACT_DIM : BASE_ACT_DIM + ARM_ACT_DIM]
            if torch.any(lift_hold_mask):
                arm_act = arm_act.clone()
                arm_act[lift_hold_mask] = 0.0
            cmd = command_dict[name]
            self._command_cache[name] = cmd
            robot = self._robot_entities.get(name)
            if robot is None:
                continue

            base_ids = self._base_joint_ids.get(name)
            if base_ids is not None and base_ids.numel() > 0:
                if int(base_ids.numel()) == 4:
                    vx = cmd[:, 0]
                    vy = cmd[:, 1]
                    wz = cmd[:, 2]
                    wz_term = wz * yaw_coupling
                    # Convert wheel tangential linear speed [m/s] into wheel joint speed [rad/s].
                    wheel_targets = torch.stack(
                        (
                            vx - vy - wz_term,
                            vx + vy + wz_term,
                            vx + vy - wz_term,
                            vx - vy + wz_term,
                        ),
                        dim=1,
                    ) / wheel_radius
                else:
                    wheel_targets = cmd[:, 0:1].repeat(1, int(base_ids.numel())) / wheel_radius
                self._set_joint_velocity_target(robot, wheel_targets, base_ids)

            arm_ids = self._arm_joint_ids.get(name)
            arm_hold = self._arm_hold_targets.get(name)
            arm_default = self._arm_default_targets.get(name)
            if arm_ids is not None and arm_ids.numel() > 0:
                if arm_hold is None or arm_hold.shape[1] != int(arm_ids.numel()):
                    arm_hold = torch.zeros((self._num_envs, int(arm_ids.numel())), dtype=torch.float32, device=self._device)
                if arm_default is None or arm_default.shape[1] != int(arm_ids.numel()):
                    arm_default = arm_hold
                arm_target = arm_default.clone()
                action_target_indices = self._arm_action_target_indices.get(name)
                if action_target_indices is None or action_target_indices.numel() == 0:
                    action_target_indices = torch.arange(
                        min(int(arm_target.shape[1]), int(arm_act.shape[1])),
                        dtype=torch.long,
                        device=self._device,
                    )
                n_ctrl = min(int(action_target_indices.numel()), int(arm_act.shape[1]))
                if n_ctrl > 0:
                    arm_target[:, action_target_indices[:n_ctrl]] += arm_act[:, :n_ctrl] * arm_delta_limit
                self._arm_hold_targets[name] = arm_target
                self._set_joint_position_target(robot, arm_target, arm_ids)

            if hasattr(robot, "write_data_to_sim"):
                robot.write_data_to_sim()

        self._process_grip_actions()

    def _phase_inputs(self) -> TorchBatchedPhaseInputs:
        self._refresh_attachment_runtime_available()
        distance_by_agent = self._robot_payload_distances()
        contact_by_agent = {
            name: self._safe_contact_force_norm(self._robot_entities[name])
            for name in self.possible_agents
        }
        self._contact_force_cache = contact_by_agent
        distances = torch.stack([distance_by_agent[name] for name in self.possible_agents], dim=1)
        contact_forces = torch.stack([contact_by_agent[name] for name in self.possible_agents], dim=1)

        contact_mask = contact_forces >= float(self.cfg.contact_force_threshold_n)
        contact_count = torch.sum(contact_mask.to(dtype=torch.int32), dim=1)
        attachment_count = self._attachment_counts()
        all_by_contact = contact_count >= int(self.cfg.num_robots)
        all_by_attachment = attachment_count >= int(self.cfg.num_robots)
        if self._lift_uses_contact_only():
            all_attached = all_by_contact
        elif self._attachment_runtime_available:
            all_attached = all_by_attachment
        else:
            all_attached = all_by_contact | all_by_attachment
        self._all_attached_mask = all_attached
        self._base_contact_force_n = torch.mean(contact_forces, dim=1)

        payload = self._payload_features()
        payload_xy = payload[:, 0:2]
        payload_z = self._safe_root_state(self._payload_entity)[:, 2]
        lift_delta_z, lift_ready_mask = self._lift_progress_state(payload_z)
        goal_dist = torch.linalg.vector_norm(payload_xy - self._goal_xy, dim=1)

        approach_ready = torch.all(distances <= float(self.cfg.approach_radius_m), dim=1)
        fine_approach_ready = torch.all(distances <= float(self.cfg.fine_approach_radius_m), dim=1)
        balance_ok = torch.std(contact_forces, dim=1, unbiased=False) <= float(self.cfg.contact_force_balance_std_n)
        now_s = self._episode_time_seconds()
        probe_elapsed_s = now_s - self._phase_mgr.phase_started_at_s
        probe_phase_id = int(PHASE_TO_ID[Phase.PROBE])
        in_probe = self._phase_mgr.phase_ids == probe_phase_id
        belief_ok = (
            (self._belief_uncertainty[:, 0] <= float(self.cfg.probe_belief_com_threshold_m))
            & (self._belief_uncertainty[:, 1] <= float(self.cfg.probe_belief_mass_rel_threshold))
        )
        probe_done = (
            in_probe
            & all_attached
            & balance_ok
            & belief_ok
            & (probe_elapsed_s >= float(max(0.0, self.cfg.probe_hold_s)))
        )
        correct_done = torch.zeros_like(probe_done, dtype=torch.bool)
        regrip_phase_id = int(PHASE_TO_ID[Phase.REGRIP])
        in_regrip = self._phase_mgr.phase_ids == regrip_phase_id
        regrip_elapsed_s = now_s - self._phase_mgr.phase_started_at_s
        regrip_done = in_regrip & all_attached & (
            torch.std(contact_forces, dim=1, unbiased=False) <= float(self.cfg.regrip_balance_std_n)
        ) & (regrip_elapsed_s >= float(max(0.0, self.cfg.regrip_hold_s)))
        lift_done = lift_ready_mask & (lift_delta_z >= float(self.cfg.lift_height_m))
        transport_done = lift_done & (goal_dist <= float(self.cfg.goal_tolerance_m))
        place_done = transport_done & (payload_z <= float(self.cfg.place_height_m))

        return TorchBatchedPhaseInputs(
            approach_ready=approach_ready.to(dtype=torch.bool),
            fine_approach_ready=fine_approach_ready.to(dtype=torch.bool),
            all_attached=all_attached.to(dtype=torch.bool),
            probe_done=probe_done.to(dtype=torch.bool),
            correct_done=correct_done.to(dtype=torch.bool),
            regrip_done=regrip_done.to(dtype=torch.bool),
            lift_done=lift_done.to(dtype=torch.bool),
            transport_done=transport_done.to(dtype=torch.bool),
            place_done=place_done.to(dtype=torch.bool),
        )

    def _episode_time_seconds(self) -> torch.Tensor:
        if hasattr(self, "episode_length_buf"):
            steps = self.episode_length_buf.to(device=self._device, dtype=torch.float32)
        else:
            steps = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        control_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
        return steps * control_dt

    def _phase_one_hot(self) -> torch.Tensor:
        out = torch.zeros((self._num_envs, len(PHASE_TO_ID)), dtype=torch.float32, device=self._device)
        idx = self._phase_mgr.phase_ids.to(dtype=torch.long)
        out.scatter_(1, idx.view(-1, 1), 1.0)
        return out

    def _safe_root_state(self, entity) -> torch.Tensor:
        if entity is not None and hasattr(entity, "data"):
            data = entity.data
            for attr in ("root_state_w", "default_root_state"):
                if hasattr(data, attr):
                    root = getattr(data, attr)
                    if isinstance(root, torch.Tensor):
                        if root.shape[0] >= self._num_envs:
                            return root[: self._num_envs].to(device=self._device, dtype=torch.float32)
                        if root.shape[0] == 1:
                            return root.to(device=self._device, dtype=torch.float32).repeat(self._num_envs, 1)
        return torch.zeros((self._num_envs, 13), dtype=torch.float32, device=self._device)

    def _robot_base_features(self, robot: Articulation) -> torch.Tensor:
        root = self._safe_root_state(robot)
        pos_xy = root[:, 0:2]
        quat = root[:, 3:7]
        lin_xy = root[:, 7:9]
        ang_z = root[:, 12:13]
        # Approximate yaw from quaternion [w, x, y, z] ordering is model dependent; robust placeholder.
        q_w = quat[:, 0:1]
        q_z = quat[:, 3:4]
        yaw = 2.0 * torch.atan2(q_z, torch.clamp(q_w, min=1e-6))
        return torch.cat([pos_xy, yaw, lin_xy, ang_z], dim=1)  # [N, 6]

    def _payload_features(self) -> torch.Tensor:
        root = self._safe_root_state(self._payload_entity)
        pos_xy = root[:, 0:2]
        quat = root[:, 3:7]
        lin_xy = root[:, 7:9]
        ang_z = root[:, 12:13]
        q_w = quat[:, 0:1]
        q_z = quat[:, 3:4]
        yaw = 2.0 * torch.atan2(q_z, torch.clamp(q_w, min=1e-6))
        return torch.cat([pos_xy, yaw, lin_xy, ang_z], dim=1)  # [N, 6]

    def _attachment_mask_for_agent(self, agent_name: str) -> torch.Tensor:
        return self._attachment_mask[:, int(self._agent_index[agent_name])]

    def _attachment_count_fraction(self) -> torch.Tensor:
        return torch.clamp(
            self._attachment_count_buf.to(dtype=torch.float32) / float(max(1, self.cfg.num_robots)),
            0.0,
            1.0,
        )

    def _robot_hand_prim_path(self, env_id: int, agent_name: str) -> str:
        agent_idx = int(str(agent_name).split("_")[-1])
        return f"/World/envs/env_{int(env_id)}/Robot_{agent_idx}/panda_hand"

    def _payload_prim_path(self, env_id: int) -> str:
        return f"/World/envs/env_{int(env_id)}/Payload"

    def _process_grip_actions(self) -> None:
        if not self._refresh_attachment_runtime_available():
            return

        phase_ids = self._phase_mgr.phase_ids
        contact_phase_id = int(PHASE_TO_ID[Phase.CONTACT])
        probe_phase_id = int(PHASE_TO_ID[Phase.PROBE])
        regrip_phase_id = int(PHASE_TO_ID[Phase.REGRIP])
        lift_phase_id = int(PHASE_TO_ID[Phase.LIFT])
        attach_allowed = (phase_ids >= contact_phase_id) & (phase_ids < lift_phase_id)
        collective_release_allowed = (phase_ids == probe_phase_id) | (phase_ids == regrip_phase_id)

        grip_cmd = torch.stack(
            [self._last_actions[name][:, BASE_ACT_DIM + ARM_ACT_DIM] for name in self.possible_agents],
            dim=1,
        )

        # In probe/regrip, let individual robots drop a bad seal without forcing a full team release.
        for agent_idx, agent_name in enumerate(self.possible_agents):
            per_robot_release = grip_cmd[:, agent_idx] <= float(self.cfg.grip_release_threshold)
            release_envs = collective_release_allowed & per_robot_release
            for env_id in (
                torch.nonzero(release_envs, as_tuple=False).view(-1).detach().to(device="cpu", dtype=torch.long).tolist()
            ):
                self._detach_vacuum(int(env_id), agent_name)

        # In regrip, escalate to a full-team reset only if enough robots agree the team geometry is bad.
        in_regrip = phase_ids == regrip_phase_id
        release_votes = (grip_cmd <= float(self.cfg.grip_release_threshold)).sum(dim=1)
        vote_threshold = int(np.clip(int(self.cfg.grip_consensus_release_votes), 1, max(1, self.cfg.num_robots)))
        consensus_release = in_regrip & (release_votes >= vote_threshold)
        for env_id in (
            torch.nonzero(consensus_release, as_tuple=False).view(-1).detach().to(device="cpu", dtype=torch.long).tolist()
        ):
            for agent_name in self.possible_agents:
                self._detach_vacuum(int(env_id), agent_name)

        attach_force_threshold = float(
            max(float(self.cfg.contact_force_threshold_n), float(self.cfg.grip_contact_force_threshold_n))
        )
        for agent_idx, agent_name in enumerate(self.possible_agents):
            attached_mask = self._attachment_mask_for_agent(agent_name)
            grip_request = grip_cmd[:, agent_idx] >= float(self.cfg.grip_attach_threshold)
            force_n = self._contact_force_cache.get(agent_name)
            if force_n is None:
                force_n = self._safe_contact_force_norm(self._robot_entities[agent_name])
            contact_ready = force_n >= attach_force_threshold
            should_attach = attach_allowed & grip_request & contact_ready & (~attached_mask)
            for env_id in (
                torch.nonzero(should_attach, as_tuple=False).view(-1).detach().to(device="cpu", dtype=torch.long).tolist()
            ):
                self._attach_vacuum(
                    int(env_id),
                    agent_name,
                    self._robot_hand_prim_path(int(env_id), agent_name),
                    self._payload_prim_path(int(env_id)),
                )

    def _detach_overloaded_attachments(self) -> None:
        if not self._refresh_attachment_runtime_available():
            return
        overload_limit = float(self.cfg.contact_force_overload_n)
        allow_detach_mask = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        if self._curriculum_phase == "lift":
            allow_detach_mask = self._lift_settle_baseline_captured.clone()
        for agent_name in self.possible_agents:
            attached_mask = self._attachment_mask_for_agent(agent_name) & allow_detach_mask
            if not torch.any(attached_mask):
                continue
            force_n = self._contact_force_cache.get(agent_name)
            if force_n is None:
                force_n = self._safe_contact_force_norm(self._robot_entities[agent_name])
            overload_mask = attached_mask & (force_n > overload_limit)
            for env_id in (
                torch.nonzero(overload_mask, as_tuple=False).view(-1).detach().to(device="cpu", dtype=torch.long).tolist()
            ):
                self._detach_vacuum(int(env_id), agent_name)

    def _update_belief_state(self) -> None:
        if not bool(self.cfg.use_belief_ekf) or not self._belief_ekf:
            return
        if not bool(self._belief_dirty):
            return
        payload_xy = self._payload_features()[:, 0:2].detach().to(device="cpu", dtype=torch.float32).numpy()
        robot_xy = []
        contact_forces = []
        for name in self.possible_agents:
            robot_xy.append(
                self._robot_base_features(self._robot_entities[name])[:, 0:2]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            force_t = self._safe_contact_force_z(name)
            contact_forces.append(force_t.detach().to(device="cpu", dtype=torch.float32).numpy())
        robot_xy_arr = np.stack(robot_xy, axis=1).astype(np.float32)
        force_arr = np.stack(contact_forces, axis=1).astype(np.float32)
        force_tensor = torch.as_tensor(force_arr, dtype=torch.float32, device=self._device)
        alpha = float(np.clip(float(self.cfg.belief_force_ema_alpha), 0.0, 1.0))
        if alpha <= 0.0:
            self._belief_force_ema = force_tensor
        else:
            self._belief_force_ema = (1.0 - alpha) * self._belief_force_ema + alpha * force_tensor
        smoothed_force_arr = self._belief_force_ema.detach().to(device="cpu", dtype=torch.float32).numpy()
        contact_phase_id = int(PHASE_TO_ID[Phase.CONTACT])
        in_contact_mask = (
            self._phase_mgr.phase_ids.detach().to(device="cpu", dtype=torch.int64).numpy() >= contact_phase_id
        )
        attached_mask = (
            self._attachment_count_buf.detach().to(device="cpu", dtype=torch.int32).numpy() > 0
        )
        contact_count_mask = (
            np.count_nonzero(
                smoothed_force_arr >= float(self.cfg.contact_force_threshold_n),
                axis=1,
            )
            >= int(max(1, self.cfg.belief_min_contact_robots))
        )
        stable_contact_mask = in_contact_mask & (attached_mask | contact_count_mask)
        for env_id, ekf in enumerate(self._belief_ekf):
            ekf.predict()
            if bool(stable_contact_mask[env_id]):
                ekf.update(
                    forces_z=smoothed_force_arr[env_id],
                    robot_positions_xy=robot_xy_arr[env_id],
                    object_center_xy=payload_xy[env_id],
                )
        self._sync_belief_tensors()
        self._maybe_debug_belief(smoothed_force_arr)
        self._belief_dirty = False

    def _residual_features(self, env_id: int, agent_idx: int, radial_xy: np.ndarray) -> np.ndarray:
        radial = np.asarray(radial_xy, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(radial))
        if norm > 1.0e-6:
            radial = radial / norm
        measured_share = 0.0
        target_share = 1.0 / float(max(1, len(self.possible_agents)))
        if self.possible_agents:
            force_t = self._contact_force_cache.get(self.possible_agents[agent_idx])
            if force_t is not None:
                total = float(
                    sum(
                        torch.clamp(self._contact_force_cache[name][env_id], min=0.0).item()
                        for name in self.possible_agents
                    )
                )
                if total > 1.0e-6:
                    measured_share = float(max(0.0, force_t[env_id].item()) / total)
        delta_share = measured_share - target_share
        posterior_max = float(np.clip(1.0 - self._belief_uncertainty[env_id, 0].item(), 0.0, 1.0))
        selected_conf = posterior_max
        payload_mass_kg = float(self._payload_mass_kg[env_id].item())
        payload_norm = float(np.clip(payload_mass_kg / 280.0, 0.0, 1.5))
        mass_norm = float(np.clip(self._belief_mu[env_id, 0].item() / 280.0, 0.0, 2.0))
        dims = np.asarray(self.cfg.payload_dims_m, dtype=np.float32)
        dim_l = float(np.clip(dims[0] / 2.0, 0.0, 2.0))
        dim_w = float(np.clip(dims[1] / 2.0, 0.0, 2.0))
        dim_h = float(np.clip(dims[2], 0.0, 2.0))
        return np.array(
            [
                measured_share,
                target_share,
                delta_share,
                posterior_max,
                selected_conf,
                payload_norm,
                1.0 if payload_mass_kg >= 150.0 else 0.0,
                mass_norm,
                dim_l,
                dim_w,
                dim_h,
                float(self._belief_uncertainty[env_id, 0].item()),
                float(radial[0]),
                float(radial[1]),
            ],
            dtype=np.float32,
        )

    def _update_residual_goal_offset(self) -> None:
        offsets = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device)
        payload_xy = self._payload_features()[:, 0:2].detach().to(device="cpu", dtype=torch.float32).numpy()
        goal_xy = self._goal_xy.detach().to(device="cpu", dtype=torch.float32).numpy()
        if self._residual_runtime is None:
            self._residual_goal_offset = offsets
            return
        if len(RESIDUAL_FEATURE_NAMES) != 14:
            self._residual_goal_offset = offsets
            return
        for env_id in range(self._num_envs):
            radial = goal_xy[env_id] - payload_xy[env_id]
            feat = self._residual_features(env_id, agent_idx=0, radial_xy=radial)
            pred = self._residual_runtime.predict(feat)
            offsets[env_id, 0] = float(np.clip(pred.dx, -self.cfg.residual_max_offset_m, self.cfg.residual_max_offset_m))
            offsets[env_id, 1] = float(np.clip(pred.dy, -self.cfg.residual_max_offset_m, self.cfg.residual_max_offset_m))
        self._residual_goal_offset = offsets

    def _agent_obs(self, agent_name: str, agent_idx: int) -> torch.Tensor:
        robot = self._robot_entities[agent_name]
        ego = self._robot_base_features(robot)  # [N, 6]
        payload = self._payload_features()  # [N, 6]
        rel_payload = payload - ego  # [N, 6]

        adjusted_goal_xy = self._goal_xy + self._residual_goal_offset
        goal = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        goal[:, 0:2] = adjusted_goal_xy
        goal[:, 2] = self._belief_uncertainty[:, 0]
        goal[:, 3:5] = adjusted_goal_xy - ego[:, 0:2]
        goal[:, 5] = self._belief_uncertainty[:, 1]

        neighbor_blocks = []
        for j, name_j in enumerate(self.possible_agents):
            if j == agent_idx:
                continue
            other = self._robot_base_features(self._robot_entities[name_j])  # [N,6]
            rel = torch.cat([other[:, 0:2] - ego[:, 0:2], other[:, 2:3] - ego[:, 2:3], other[:, 3:6] - ego[:, 3:6]], dim=1)
            cmd = self._command_cache[name_j]  # [N,BASE_CMD_DIM]
            neighbor_blocks.append(torch.cat([rel, cmd], dim=1))  # [N,9]
        neighbors = torch.cat(neighbor_blocks, dim=1) if neighbor_blocks else torch.zeros((self._num_envs, 27), dtype=torch.float32, device=self._device)

        phase = self._phase_one_hot()  # [N,10]
        act = self._last_actions[agent_name]  # [N,7]
        force_n = self._contact_force_cache.get(agent_name)
        if force_n is None:
            force_n = self._safe_contact_force_norm(robot)
        grip_state = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        grip_state[:, 0] = torch.clamp(force_n / max(1.0, float(self.cfg.contact_force_overload_n)), 0.0, 1.5)
        grip_state[:, 1] = self._attachment_mask_for_agent(agent_name).to(dtype=torch.float32)
        grip_state[:, 2] = self._attachment_count_fraction()

        # Assemble to exactly 65 dimensions with stable ordering.
        parts = [ego, rel_payload, goal, neighbors, phase, act, grip_state]
        obs = torch.cat(parts, dim=1).to(dtype=torch.float32)
        if obs.shape[1] != AGENT_OBS_DIM:
            raise RuntimeError(f"Agent observation dim mismatch: expected {AGENT_OBS_DIM}, got {obs.shape[1]}")
        return obs

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        self._refresh_entity_buffers()
        obs: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.possible_agents):
            obs[name] = self._agent_obs(name, i)
        return obs

    def _get_states(self) -> torch.Tensor:
        self._refresh_entity_buffers()
        self._phase_inputs()
        self._update_belief_state()
        self._update_residual_goal_offset()
        # Explicit centralized critic input (do not rely on implicit concatenation fallback).
        per_agent_obs = [self._agent_obs(name, i) for i, name in enumerate(self.possible_agents)]
        obs_cat = torch.cat(per_agent_obs, dim=1)  # [N, 260] for 4x65

        payload = self._payload_features()  # [N,6]
        phase = self._phase_one_hot()  # [N,10]
        time_s = self._episode_time_seconds().view(-1, 1)
        # Construct a 72D global block with stable ordering; fill remainder with zeros for now.
        # 72D global block so total state dim stays 316 (= 244 + 72).
        global_block = torch.zeros((self._num_envs, 72), dtype=torch.float32, device=self._device)
        global_block[:, 0:6] = payload
        global_block[:, 6:16] = phase
        global_block[:, 16:17] = time_s
        global_block[:, 17:19] = self._goal_xy + self._residual_goal_offset
        global_block[:, 19:26] = self._belief_mu
        global_block[:, 26:33] = self._belief_cov_diag
        global_block[:, 33:35] = self._belief_uncertainty
        global_block[:, 35:39] = self._cbf_slack[:, :4]
        global_block[:, 39:43] = self._cbf_applied[:, :4].to(dtype=torch.float32)
        global_block[:, 43:47] = self._neural_barrier[:, :4]
        global_block[:, 47:49] = self._residual_goal_offset
        global_block[:, 49:53] = torch.stack(
            [self._attachment_mask_for_agent(name).to(dtype=torch.float32) for name in self.possible_agents],
            dim=1,
        )
        global_block[:, 53:54] = self._attachment_count_fraction().view(-1, 1)

        state = torch.cat([obs_cat, global_block], dim=1).to(dtype=torch.float32)
        target_dim = int(self.cfg.state_space)
        if state.shape[1] != target_dim:
            raise RuntimeError(f"Central state dim mismatch: expected {target_dim}, got {state.shape[1]}")
        return state

    def state(self) -> torch.Tensor:
        # SKRL IsaacLabMultiAgentWrapper expects a global tensor state.
        return self._get_states()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # Optional compatibility helper for callers that expect per-agent state dicts.
        state = self._get_states()
        return {name: state for name in self.possible_agents}

    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        self._refresh_entity_buffers()
        phase_inputs = self._phase_inputs()
        self._update_belief_state()
        self._update_residual_goal_offset()
        self._phase_mgr.update(phase_inputs, now_s=self._episode_time_seconds())
        self._detach_overloaded_attachments()
        payload_xy = self._payload_features()[:, 0:2]
        payload_z = self._safe_root_state(self._payload_entity)[:, 2]
        adjusted_goal_xy = self._goal_xy + self._residual_goal_offset
        goal_dist = torch.linalg.vector_norm(payload_xy - adjusted_goal_xy, dim=1)
        goal_bonus = 0.02 * (1.0 - torch.tanh(goal_dist))
        lift_phase_id = int(PHASE_TO_ID[Phase.LIFT])
        in_lift_mask = self._phase_mgr.phase_ids == lift_phase_id
        lift_delta_z, lift_ready_mask = self._lift_progress_state(payload_z)
        lift_progress = torch.clamp(lift_delta_z / float(max(1.0e-6, self.cfg.lift_height_m)), min=0.0, max=1.0)
        lift_shaping = (
            0.05
            * in_lift_mask.to(dtype=torch.float32)
            * lift_ready_mask.to(dtype=torch.float32)
            * lift_progress
        )

        current_all_attached = phase_inputs.all_attached
        if self._attachment_runtime_available and (not self._lift_uses_contact_only()):
            current_all_attached = self._attachment_counts() >= int(self.cfg.num_robots)
        team_attach_bonus = 0.01 * current_all_attached.to(dtype=torch.float32)
        contact_force_stack = torch.stack([self._contact_force_cache[name] for name in self.possible_agents], dim=1)
        in_probe_mask = self._phase_mgr.phase_ids == int(PHASE_TO_ID[Phase.PROBE])
        balance_bonus = 0.003 * in_probe_mask.to(dtype=torch.float32) * (
            1.0
            - torch.tanh(
                torch.std(contact_force_stack, dim=1, unbiased=False)
                / float(max(1.0e-6, self.cfg.contact_force_balance_std_n))
            )
        )
        rewards: Dict[str, torch.Tensor] = {}
        overload_limit = float(self.cfg.contact_force_overload_n)
        for name in self.possible_agents:
            agent_idx = self.possible_agents.index(name)
            act = self._last_actions[name]
            control_pen = 0.001 * torch.sum(act**2, dim=1)
            force_n = self._contact_force_cache.get(name)
            if force_n is None:
                force_n = self._safe_contact_force_norm(self._robot_entities[name])
            contact_bonus = 0.004 * (force_n >= float(self.cfg.contact_force_threshold_n)).to(dtype=torch.float32)
            attach_bonus = 0.006 * self._attachment_mask_for_agent(name).to(dtype=torch.float32)
            if self._lift_uses_contact_only():
                attach_bonus = torch.zeros_like(attach_bonus)
            overload_pen = 0.002 * torch.relu(force_n - overload_limit)
            cbf_pen = 0.002 * self._cbf_slack[:, agent_idx]
            intervention_pen = 0.001 * self._cbf_applied[:, agent_idx].to(dtype=torch.float32)
            belief_bonus = 0.002 * (1.0 - torch.tanh(self._belief_uncertainty[:, 0]))
            rewards[name] = (
                -0.01
                - control_pen
                - overload_pen
                - cbf_pen
                - intervention_pen
                + contact_bonus
                + attach_bonus
                + team_attach_bonus
                + balance_bonus
                + goal_bonus
                + lift_shaping
                + belief_bonus
            ).to(dtype=torch.float32)
        return rewards

    def _get_dones(self) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        done_phase_id = int(PHASE_TO_ID[Phase.DONE])
        terminated = self._phase_mgr.phase_ids == done_phase_id
        if hasattr(self, "episode_length_buf") and hasattr(self, "max_episode_length"):
            truncated = self.episode_length_buf >= int(max(1, int(self.max_episode_length) - 1))
            truncated = truncated.to(device=self._device, dtype=torch.bool)
        else:
            truncated = self._episode_time_seconds() >= float(self.cfg.episode_length_s)
        terminated_dict = {name: terminated for name in self.possible_agents}
        truncated_dict = {name: truncated for name in self.possible_agents}
        return terminated_dict, truncated_dict

    def _attachment_prim_path(self, env_id: int, agent_name: str) -> str:
        return f"/World/envs/env_{int(env_id)}/Attachments/{agent_name}"

    def _attach_vacuum(self, env_id: int, agent_name: str, parent_rigid_body_path: str, child_rigid_body_path: str) -> None:
        key = (int(env_id), str(agent_name))
        if key in self._attachments:
            return
        parent_pos_w = None
        parent_quat_w = None
        child_pos_w = None
        child_quat_w = None
        robot = self._robot_entities.get(agent_name)
        hand_state = self._hand_body_state(robot) if robot is not None else None
        if hand_state is not None and int(env_id) < int(hand_state.shape[0]):
            hand_pose = hand_state[int(env_id)]
            parent_pos_w = (float(hand_pose[0]), float(hand_pose[1]), float(hand_pose[2]))
            parent_quat_w = (float(hand_pose[3]), float(hand_pose[4]), float(hand_pose[5]), float(hand_pose[6]))
        if self._payload_entity is not None:
            payload_root = self._safe_root_state(self._payload_entity)
            if int(env_id) < int(payload_root.shape[0]):
                payload_pose = payload_root[int(env_id)]
                child_pos_w = (float(payload_pose[0]), float(payload_pose[1]), float(payload_pose[2]))
                child_quat_w = (
                    float(payload_pose[3]),
                    float(payload_pose[4]),
                    float(payload_pose[5]),
                    float(payload_pose[6]),
                )
        attachment_path = self._attachment_prim_path(env_id, agent_name)
        created_path = self._attachment_backend.attach(
            attachment_prim_path=attachment_path,
            parent_rigid_body_path=parent_rigid_body_path,
            child_rigid_body_path=child_rigid_body_path,
            parent_pos_w=parent_pos_w,
            parent_quat_w=parent_quat_w,
            child_pos_w=child_pos_w,
            child_quat_w=child_quat_w,
        )
        self._attachments[key] = created_path
        if 0 <= int(env_id) < self._num_envs:
            self._attachment_count_buf[int(env_id)] += 1
            self._attachment_mask[int(env_id), int(self._agent_index[agent_name])] = True

    def _detach_vacuum(self, env_id: int, agent_name: str) -> None:
        key = (int(env_id), str(agent_name))
        attachment_path = self._attachments.pop(key, None)
        if attachment_path is None:
            return
        self._attachment_backend.detach(attachment_path)
        if 0 <= int(env_id) < self._num_envs:
            self._attachment_count_buf[int(env_id)] = torch.clamp(
                self._attachment_count_buf[int(env_id)] - 1,
                min=0,
            )
            self._attachment_mask[int(env_id), int(self._agent_index[agent_name])] = False
