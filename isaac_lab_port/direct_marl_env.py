from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Dict, Sequence

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


BASE_ACT_DIM = 3
ARM_ACT_DIM = 3
BASE_CMD_DIM = 3
AGENT_ACT_DIM = BASE_ACT_DIM + ARM_ACT_DIM
AGENT_OBS_DIM = 61
# Centralized critic state = concat(4 * per-agent obs) + global block.
# With current features: 4*61 + 72 = 316.
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
            ("isaaclab_assets.robots.clearpath", "RIDGEBACK_FRANKA_CFG"),
            ("isaaclab_assets.robots.clearpath.ridgeback_franka", "RIDGEBACK_FRANKA_CFG"),
            ("isaaclab_assets", "RIDGEBACK_FRANKA_CFG"),
        )
    )
    if base is not None:
        return _clone_cfg_with_prim_path(base, prim_path)

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
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path="Props/Blocks/block.usd"),
        )
    return _clone_cfg_with_prim_path(base, prim_path)


def _build_agent_spaces(agent_names: Sequence[str]) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    observation_spaces = {
        name: gym.spaces.Box(low=-1.0e6, high=1.0e6, shape=(AGENT_OBS_DIM,), dtype=float)
        for name in agent_names
    }
    action_spaces = {
        name: gym.spaces.Box(low=-1.0, high=1.0, shape=(AGENT_ACT_DIM,), dtype=float)
        for name in agent_names
    }
    shared_observation_spaces = {
        name: gym.spaces.Box(low=-1.0e6, high=1.0e6, shape=(CENTRAL_STATE_DIM,), dtype=float)
        for name in agent_names
    }
    return observation_spaces, action_spaces, shared_observation_spaces


@configclass
@dataclass
class NoVlmSceneCfg(InteractiveSceneCfg):
    num_envs: int = 128
    env_spacing: float = 8.0
    replicate_physics: bool = True


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
    contact_force_threshold_n: float = 5.0
    contact_force_balance_std_n: float = 20.0
    contact_force_overload_n: float = 220.0
    lift_height_m: float = 0.18
    place_height_m: float = 0.09
    goal_tolerance_m: float = 0.30
    sim: SimulationCfg = field(default_factory=lambda: SimulationCfg(dt=0.02, device="cuda:0"))
    scene: NoVlmSceneCfg = field(default_factory=NoVlmSceneCfg)
    robot_0: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_0"))
    robot_1: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_1"))
    robot_2: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_2"))
    robot_3: ArticulationCfg = field(default_factory=lambda: _default_ridgeback_franka_cfg(f"{ENV_REGEX_NS}/Robot_3"))
    payload: RigidObjectCfg = field(default_factory=lambda: _default_payload_cfg(f"{ENV_REGEX_NS}/Payload"))

    # Do not use deprecated num_observations/num_actions.
    possible_agents: tuple[str, ...] = field(default_factory=tuple)
    robot_prim_paths: tuple[str, ...] = field(default_factory=tuple)
    observation_spaces: Dict[str, object] = field(default_factory=dict)
    action_spaces: Dict[str, object] = field(default_factory=dict)
    shared_observation_spaces: Dict[str, object] = field(default_factory=dict)

    # Positive value means env will use explicit _get_states() for critic state.
    state_space: int = CENTRAL_STATE_DIM

    def __post_init__(self) -> None:
        if self.num_robots != 4:
            raise ValueError("Current scaffold supports exactly 4 robots.")
        if self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be > 0.")
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
        self._last_actions = {
            agent: torch.zeros((self._num_envs, AGENT_ACT_DIM), dtype=torch.float32, device=self._device)
            for agent in self.possible_agents
        }
        self._phase_mgr = TorchBatchedNoVlmPhaseManager(
            num_envs=self._num_envs,
            cfg=PhaseConfig(),
            device=self._device,
        )
        self._attachment_backend = AutoAttachmentBackend()
        self._attachments: dict[tuple[int, str], str] = {}
        self._attachment_count_buf = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        self._robot_entities: dict[str, Articulation] = {}
        self._payload_entity: RigidObject | None = None
        self._goal_xy = torch.tensor([2.5, 0.0], dtype=torch.float32, device=self._device).view(1, 2).repeat(
            self._num_envs, 1
        )
        self._command_cache = {
            agent: torch.zeros((self._num_envs, BASE_CMD_DIM), dtype=torch.float32, device=self._device)
            for agent in self.possible_agents
        }
        self._base_joint_ids: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_joint_ids: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_hold_targets: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._arm_default_targets: dict[str, torch.Tensor | None] = {agent: None for agent in self.possible_agents}
        self._base_contact_force_n = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._all_attached_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._contact_force_cache: Dict[str, torch.Tensor] = {}
        super().__init__(cfg=cfg, **kwargs)

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
            default_targets = self._default_joint_targets(robot, arm_ids)
            self._arm_default_targets[agent_name] = default_targets
            self._arm_hold_targets[agent_name] = default_targets

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

    def _safe_contact_force_norm(self, entity) -> torch.Tensor:
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
        self._attachment_count_buf[env_ids] = 0

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        self._all_attached_mask[env_ids] = False
        self._base_contact_force_n[env_ids] = 0.0
        self._attachment_count_buf[env_ids] = 0
        for robot in self._robot_entities.values():
            self._reset_articulation(robot, env_ids)
        if self._payload_entity is not None:
            self._reset_rigid_object(self._payload_entity, env_ids)
        self._clear_env_attachments(env_ids)
        self._phase_mgr.reset(now_s=0.0, env_ids=env_ids.detach().to("cpu").tolist())

    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]) -> None:
        for name in self.possible_agents:
            if name not in actions:
                raise KeyError(f"Missing action for agent: {name}")
            act = actions[name].to(device=self._device, dtype=torch.float32)
            if act.ndim != 2 or act.shape[1] != AGENT_ACT_DIM:
                raise ValueError(f"Action tensor for {name} must be [num_envs, {AGENT_ACT_DIM}]")
            self._last_actions[name] = torch.clamp(act, -1.0, 1.0)

    def _apply_action(self) -> None:
        # Base action mapping:
        # [vx_norm, vy_norm, yaw_norm] -> [m/s, m/s, rad/s], then to wheel rad/s.
        # Arm action mapping:
        # [d_joint1_norm, d_joint2_norm, d_joint3_norm] -> additive joint targets.
        speed_limit = 0.25
        yaw_limit = 1.2
        wheel_radius = float(max(self.cfg.wheel_radius_m, 1.0e-4))
        yaw_coupling = float(max(self.cfg.mecanum_yaw_coupling_m, 0.0))
        arm_delta_limit = float(max(self.cfg.arm_delta_limit_rad, 0.0))
        for name in self.possible_agents:
            act = self._last_actions[name]
            base_act = act[:, :BASE_ACT_DIM]
            arm_act = act[:, BASE_ACT_DIM : BASE_ACT_DIM + ARM_ACT_DIM]

            cmd = torch.empty((self._num_envs, BASE_CMD_DIM), dtype=torch.float32, device=self._device)
            cmd[:, 0] = base_act[:, 0] * speed_limit
            cmd[:, 1] = base_act[:, 1] * speed_limit
            cmd[:, 2] = base_act[:, 2] * yaw_limit
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
                n_ctrl = min(int(arm_target.shape[1]), int(arm_act.shape[1]))
                if n_ctrl > 0:
                    arm_target[:, :n_ctrl] += arm_act[:, :n_ctrl] * arm_delta_limit
                self._arm_hold_targets[name] = arm_target
                self._set_joint_position_target(robot, arm_target, arm_ids)

            if hasattr(robot, "write_data_to_sim"):
                robot.write_data_to_sim()

    def _phase_inputs(self) -> TorchBatchedPhaseInputs:
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
        all_attached = all_by_contact | all_by_attachment
        self._all_attached_mask = all_attached
        self._base_contact_force_n = torch.mean(contact_forces, dim=1)

        payload = self._payload_features()
        payload_xy = payload[:, 0:2]
        payload_z = self._safe_root_state(self._payload_entity)[:, 2]
        goal_dist = torch.linalg.vector_norm(payload_xy - self._goal_xy, dim=1)

        approach_ready = torch.all(distances <= float(self.cfg.approach_radius_m), dim=1)
        fine_approach_ready = torch.all(distances <= float(self.cfg.fine_approach_radius_m), dim=1)
        balance_ok = torch.std(contact_forces, dim=1, unbiased=False) <= float(self.cfg.contact_force_balance_std_n)
        probe_done = all_attached & balance_ok
        correct_done = probe_done
        regrip_done = all_attached
        lift_done = payload_z >= float(self.cfg.lift_height_m)
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

    def _agent_obs(self, agent_name: str, agent_idx: int) -> torch.Tensor:
        robot = self._robot_entities[agent_name]
        ego = self._robot_base_features(robot)  # [N, 6]
        payload = self._payload_features()  # [N, 6]
        rel_payload = payload - ego  # [N, 6]

        goal = torch.zeros((self._num_envs, 6), dtype=torch.float32, device=self._device)
        goal[:, 0:2] = self._goal_xy
        goal[:, 2] = 0.0
        goal[:, 3:5] = self._goal_xy - ego[:, 0:2]
        goal[:, 5] = 0.0

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
        act = self._last_actions[agent_name]  # [N,6]

        # Assemble to exactly 61 dimensions with stable ordering.
        parts = [ego, rel_payload, goal, neighbors, phase, act]  # total 6+6+6+27+10+6 = 61
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
        # Explicit centralized critic input (do not rely on implicit concatenation fallback).
        per_agent_obs = [self._agent_obs(name, i) for i, name in enumerate(self.possible_agents)]
        obs_cat = torch.cat(per_agent_obs, dim=1)  # [N, 244] for 4x61

        payload = self._payload_features()  # [N,6]
        phase = self._phase_one_hot()  # [N,10]
        time_s = self._episode_time_seconds().view(-1, 1)
        # Construct a 72D global block with stable ordering; fill remainder with zeros for now.
        # 72D global block so total state dim stays 316 (= 244 + 72).
        global_block = torch.zeros((self._num_envs, 72), dtype=torch.float32, device=self._device)
        global_block[:, 0:6] = payload
        global_block[:, 6:16] = phase
        global_block[:, 16:17] = time_s
        global_block[:, 17:19] = self._goal_xy

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
        self._phase_mgr.update(phase_inputs, now_s=self._episode_time_seconds())
        payload_xy = self._payload_features()[:, 0:2]
        goal_dist = torch.linalg.vector_norm(payload_xy - self._goal_xy, dim=1)
        goal_bonus = 0.02 * (1.0 - torch.tanh(goal_dist))

        team_attach_bonus = 0.01 * phase_inputs.all_attached.to(dtype=torch.float32)
        rewards: Dict[str, torch.Tensor] = {}
        overload_limit = float(self.cfg.contact_force_overload_n)
        for name in self.possible_agents:
            act = self._last_actions[name]
            control_pen = 0.001 * torch.sum(act**2, dim=1)
            force_n = self._contact_force_cache.get(name)
            if force_n is None:
                force_n = self._safe_contact_force_norm(self._robot_entities[name])
            contact_bonus = 0.004 * (force_n >= float(self.cfg.contact_force_threshold_n)).to(dtype=torch.float32)
            overload_pen = 0.002 * torch.relu(force_n - overload_limit)
            rewards[name] = (
                -0.01
                - control_pen
                - overload_pen
                + contact_bonus
                + team_attach_bonus
                + goal_bonus
            ).to(dtype=torch.float32)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        done_phase_id = int(PHASE_TO_ID[Phase.DONE])
        terminated = self._phase_mgr.phase_ids == done_phase_id
        if hasattr(self, "episode_length_buf") and hasattr(self, "max_episode_length"):
            truncated = self.episode_length_buf >= int(max(1, int(self.max_episode_length) - 1))
            truncated = truncated.to(device=self._device, dtype=torch.bool)
        else:
            truncated = self._episode_time_seconds() >= float(self.cfg.episode_length_s)
        return terminated, truncated

    def _attachment_prim_path(self, env_id: int, agent_name: str) -> str:
        return f"/World/envs/env_{int(env_id)}/Attachments/{agent_name}"

    def _attach_vacuum(self, env_id: int, agent_name: str, parent_rigid_body_path: str, child_rigid_body_path: str) -> None:
        key = (int(env_id), str(agent_name))
        if key in self._attachments:
            return
        attachment_path = self._attachment_prim_path(env_id, agent_name)
        self._attachment_backend.attach(attachment_path, parent_rigid_body_path, child_rigid_body_path)
        self._attachments[key] = attachment_path
        if 0 <= int(env_id) < self._num_envs:
            self._attachment_count_buf[int(env_id)] += 1

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
