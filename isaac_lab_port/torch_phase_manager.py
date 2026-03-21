from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import torch


class Phase(str, Enum):
    APPROACH = "approach"
    FINE_APPROACH = "fine_approach"
    CONTACT = "contact"
    PROBE = "probe"
    CORRECT = "correct"
    REGRIP = "regrip"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    DONE = "done"


PHASE_ORDER: tuple[Phase, ...] = (
    Phase.APPROACH,
    Phase.FINE_APPROACH,
    Phase.CONTACT,
    Phase.PROBE,
    Phase.CORRECT,
    Phase.REGRIP,
    Phase.LIFT,
    Phase.TRANSPORT,
    Phase.PLACE,
    Phase.DONE,
)

PHASE_TO_ID = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}
ID_TO_PHASE = tuple(PHASE_ORDER)


@dataclass
class PhaseConfig:
    include_probe_phase: bool = True
    include_correct_phase: bool = True
    include_regrip_phase: bool = True
    contact_hold_s: float = 0.25
    contact_timeout_s: float = 6.0
    regrip_timeout_s: float = 6.0


@dataclass
class TorchBatchedPhaseInputs:
    approach_ready: torch.Tensor
    fine_approach_ready: torch.Tensor
    all_attached: torch.Tensor
    probe_done: torch.Tensor
    correct_done: torch.Tensor
    regrip_done: torch.Tensor
    lift_done: torch.Tensor
    transport_done: torch.Tensor
    place_done: torch.Tensor

    @classmethod
    def zeros(cls, num_envs: int, device: torch.device | str = "cpu") -> "TorchBatchedPhaseInputs":
        n = int(num_envs)
        z = torch.zeros(n, dtype=torch.bool, device=device)
        return cls(
            approach_ready=z.clone(),
            fine_approach_ready=z.clone(),
            all_attached=z.clone(),
            probe_done=z.clone(),
            correct_done=z.clone(),
            regrip_done=z.clone(),
            lift_done=z.clone(),
            transport_done=z.clone(),
            place_done=z.clone(),
        )

    def validate(self, num_envs: int, device: torch.device) -> None:
        n = int(num_envs)
        for name in (
            "approach_ready",
            "fine_approach_ready",
            "all_attached",
            "probe_done",
            "correct_done",
            "regrip_done",
            "lift_done",
            "transport_done",
            "place_done",
        ):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if t.ndim != 1 or t.shape[0] != n:
                raise ValueError(f"{name} must be shape [{n}], got {tuple(t.shape)}")
            if t.device != device:
                raise ValueError(f"{name} must be on device {device}, got {t.device}")
            if t.dtype != torch.bool:
                setattr(self, name, t.to(dtype=torch.bool))


class TorchBatchedNoVlmPhaseManager:
    """Torch-native phase manager for batched Isaac Lab environments."""

    def __init__(
        self,
        num_envs: int,
        cfg: PhaseConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        self.cfg = cfg if cfg is not None else PhaseConfig()
        self.device = torch.device(device)

        self.phase_ids = torch.full(
            (self.num_envs,),
            fill_value=int(PHASE_TO_ID[Phase.APPROACH]),
            dtype=torch.int8,
            device=self.device,
        )
        self.phase_started_at_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.contact_all_attached_since_s = torch.full(
            (self.num_envs,),
            fill_value=-1.0,
            dtype=torch.float32,
            device=self.device,
        )

    def _as_now_vector(self, now_s: float | torch.Tensor) -> torch.Tensor:
        if isinstance(now_s, torch.Tensor):
            t = now_s.to(device=self.device, dtype=torch.float32).reshape(-1)
            if t.shape[0] != self.num_envs:
                raise ValueError(f"now_s must have length {self.num_envs}, got {t.shape[0]}")
            return t
        return torch.full((self.num_envs,), float(now_s), dtype=torch.float32, device=self.device)

    def _normalize_env_ids(self, env_ids: Optional[Iterable[int]]) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        idx_list = list(env_ids)
        if len(idx_list) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        idx = torch.tensor(idx_list, dtype=torch.long, device=self.device)
        if torch.any(idx < 0) or torch.any(idx >= self.num_envs):
            raise ValueError("env_ids contains out-of-range index")
        return idx

    def _set_phase(self, mask: torch.Tensor, phase: Phase, now_vec: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        pid = int(PHASE_TO_ID[phase])
        self.phase_ids[mask] = pid
        self.phase_started_at_s[mask] = now_vec[mask]
        if phase != Phase.CONTACT:
            self.contact_all_attached_since_s[mask] = -1.0

    def _post_contact_phase(self) -> Phase:
        if self.cfg.include_probe_phase:
            return Phase.PROBE
        if self.cfg.include_correct_phase:
            return Phase.CORRECT
        if self.cfg.include_regrip_phase:
            return Phase.REGRIP
        return Phase.LIFT

    def _post_probe_phase(self) -> Phase:
        if self.cfg.include_correct_phase:
            return Phase.CORRECT
        if self.cfg.include_regrip_phase:
            return Phase.REGRIP
        return Phase.LIFT

    def _post_correct_phase(self) -> Phase:
        if self.cfg.include_regrip_phase:
            return Phase.REGRIP
        return Phase.LIFT

    def reset(
        self,
        now_s: float | torch.Tensor = 0.0,
        env_ids: Optional[Iterable[int]] = None,
    ) -> None:
        idx = self._normalize_env_ids(env_ids)
        if idx.numel() == 0:
            return
        now_vec = self._as_now_vector(now_s)
        self.phase_ids[idx] = int(PHASE_TO_ID[Phase.APPROACH])
        self.phase_started_at_s[idx] = now_vec[idx]
        self.contact_all_attached_since_s[idx] = -1.0

    def phase_names(self) -> list[str]:
        cpu_ids = self.phase_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        return [ID_TO_PHASE[int(pid)].value for pid in cpu_ids]

    def update(self, inp: TorchBatchedPhaseInputs, now_s: float | torch.Tensor) -> torch.Tensor:
        now_vec = self._as_now_vector(now_s)
        inp.validate(self.num_envs, self.device)

        phase_ids0 = self.phase_ids.clone()
        phase_started_at_s0 = self.phase_started_at_s.clone()
        transitioned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def apply_transition(mask: torch.Tensor, phase: Phase) -> None:
            mask = mask & (~transitioned)
            if not torch.any(mask):
                return
            self._set_phase(mask, phase, now_vec)
            transitioned[mask] = True

        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.APPROACH])) & inp.approach_ready,
            Phase.FINE_APPROACH,
        )
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.FINE_APPROACH])) & inp.fine_approach_ready,
            Phase.CONTACT,
        )

        in_contact = phase_ids0 == int(PHASE_TO_ID[Phase.CONTACT])
        attached = in_contact & inp.all_attached
        not_started = self.contact_all_attached_since_s < 0.0
        newly_attached = attached & not_started
        self.contact_all_attached_since_s[newly_attached] = now_vec[newly_attached]

        detached = in_contact & (~inp.all_attached)
        self.contact_all_attached_since_s[detached] = -1.0

        hold_s = float(max(0.0, self.cfg.contact_hold_s))
        attached_since = self.contact_all_attached_since_s
        held_long_enough = attached & (attached_since >= 0.0) & ((now_vec - attached_since) >= hold_s)
        apply_transition(held_long_enough, self._post_contact_phase())

        elapsed = now_vec - phase_started_at_s0
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.CONTACT]))
            & (elapsed >= float(max(0.0, self.cfg.contact_timeout_s))),
            Phase.APPROACH,
        )

        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.PROBE])) & inp.probe_done,
            self._post_probe_phase(),
        )
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.CORRECT])) & inp.correct_done,
            self._post_correct_phase(),
        )
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.REGRIP])) & inp.regrip_done,
            Phase.LIFT,
        )

        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.REGRIP]))
            & (elapsed >= float(max(0.0, self.cfg.regrip_timeout_s))),
            Phase.CONTACT,
        )

        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.LIFT])) & inp.lift_done,
            Phase.TRANSPORT,
        )
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.TRANSPORT])) & inp.transport_done,
            Phase.PLACE,
        )
        apply_transition(
            (phase_ids0 == int(PHASE_TO_ID[Phase.PLACE])) & inp.place_done,
            Phase.DONE,
        )
        return self.phase_ids.clone()
