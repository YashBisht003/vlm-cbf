"""Isaac Lab migration scaffold for no-VLM cooperative transport."""

from .no_vlm_task_spec import NoVlmTaskSpec, RobotPreset, VacuumBackend
from .task_registry import SKRL_CFG_ENTRY_POINT, TASK_ID, register_no_vlm_task
from .vacuum_attachment import AutoAttachmentBackend, VacuumAttachmentConfig
from .torch_phase_manager import (
    ID_TO_PHASE,
    PHASE_TO_ID,
    Phase,
    PhaseConfig,
    TorchBatchedNoVlmPhaseManager,
    TorchBatchedPhaseInputs,
)
