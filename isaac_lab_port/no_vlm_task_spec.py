from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Tuple


class RobotPreset(str, Enum):
    RIDGEBACK_FRANKA = "ridgeback_franka"
    RIDGEBACK_UR5 = "ridgeback_ur5"


class VacuumBackend(str, Enum):
    CPU_SURFACE_GRIPPER = "cpu_surface_gripper"
    GPU_FIXED_JOINT = "gpu_fixed_joint"


@dataclass(frozen=True)
class RobotAsset:
    preset: RobotPreset
    usd_path: str
    notes: str


ROBOT_ASSETS: Dict[RobotPreset, RobotAsset] = {
    RobotPreset.RIDGEBACK_FRANKA: RobotAsset(
        preset=RobotPreset.RIDGEBACK_FRANKA,
        usd_path="Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
        notes="Mobile manipulator baseline with Franka arm.",
    ),
    RobotPreset.RIDGEBACK_UR5: RobotAsset(
        preset=RobotPreset.RIDGEBACK_UR5,
        usd_path="Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd",
        notes="Mobile manipulator variant with UR5 arm.",
    ),
}


@dataclass
class NoVlmTaskSpec:
    # Core setup
    robot_preset: RobotPreset = RobotPreset.RIDGEBACK_FRANKA
    num_robots: int = 4
    num_envs: int = 128
    device: str = "cuda"

    # Vacuum/attachment strategy
    vacuum_backend: VacuumBackend = VacuumBackend.GPU_FIXED_JOINT

    # Problem definition (VLM removed)
    use_belief_ekf: bool = True
    use_cbf: bool = True
    include_probe_phase: bool = True
    include_correct_phase: bool = True
    include_regrip_phase: bool = True

    # Task ranges
    object_mass_range: Tuple[float, float] = (50.0, 280.0)
    speed_limit: float = 0.25
    separation_min: float = 0.18
    contact_force_max: float = 140.0

    def phase_sequence(self) -> List[str]:
        # No OBSERVE/PLAN VLM stages in the new task.
        phases = ["approach", "fine_approach", "contact"]
        if self.include_probe_phase:
            phases.append("probe")
        if self.include_correct_phase:
            phases.append("correct")
        if self.include_regrip_phase:
            phases.append("regrip")
        phases.extend(["lift", "transport", "place", "done"])
        return phases

    def validate(self) -> None:
        if self.num_robots < 2:
            raise ValueError("num_robots must be >= 2 for cooperative transport.")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if self.object_mass_range[0] <= 0.0 or self.object_mass_range[1] <= self.object_mass_range[0]:
            raise ValueError("object_mass_range must be positive and increasing.")
        if self.separation_min <= 0.0:
            raise ValueError("separation_min must be > 0.")
        if self.contact_force_max <= 0.0:
            raise ValueError("contact_force_max must be > 0.")
        if self.robot_preset not in ROBOT_ASSETS:
            raise ValueError(f"Unsupported robot preset: {self.robot_preset}")

        # Surface gripper is known to be CPU-only in current Isaac Sim/Isaac Lab docs.
        if self.vacuum_backend == VacuumBackend.CPU_SURFACE_GRIPPER and self.device == "cuda":
            raise ValueError(
                "cpu_surface_gripper backend is CPU-only; use --device cpu or switch to gpu_fixed_joint."
            )

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["robot_preset"] = self.robot_preset.value
        data["vacuum_backend"] = self.vacuum_backend.value
        data["phase_sequence"] = self.phase_sequence()
        data["robot_asset"] = asdict(ROBOT_ASSETS[self.robot_preset])
        data["robot_asset"]["preset"] = ROBOT_ASSETS[self.robot_preset].preset.value
        return data

