from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

try:
    import gymnasium as gym
except Exception as exc:
    raise RuntimeError("gymnasium is required for task registration.") from exc

try:
    from . import agents as _agents
except ImportError:
    import agents as _agents

TASK_ID = "Isaac-NoVlm-CoopTransport-Direct-v0"
_agents_module_name = _agents.__name__
if _agents_module_name == "agents":
    _agents_module_name = "isaac_lab_port.agents"
SKRL_CFG_ENTRY_POINT = f"{_agents_module_name}:skrl_mappo_cfg.yaml"


@dataclass
class TaskRegistration:
    task_id: str
    entry_point: str
    kwargs: Dict


def _load_env_types():
    try:
        from .direct_marl_env import NoVlmCoopTransportDirectEnv, NoVlmCoopTransportEnvCfg
    except ImportError:
        from direct_marl_env import NoVlmCoopTransportDirectEnv, NoVlmCoopTransportEnvCfg
    return NoVlmCoopTransportDirectEnv, NoVlmCoopTransportEnvCfg


def register_no_vlm_task(force: bool = False) -> TaskRegistration:
    NoVlmCoopTransportDirectEnv, NoVlmCoopTransportEnvCfg = _load_env_types()
    kwargs = {
        "env_cfg_entry_point": NoVlmCoopTransportEnvCfg,
        "skrl_cfg_entry_point": SKRL_CFG_ENTRY_POINT,
        "skrl_mappo_cfg_entry_point": SKRL_CFG_ENTRY_POINT,
    }

    already = False
    try:
        already = gym.spec(TASK_ID) is not None
    except Exception:
        already = False

    if already and not force:
        spec = gym.spec(TASK_ID)
        return TaskRegistration(
            task_id=TASK_ID,
            entry_point=str(spec.entry_point),
            kwargs=getattr(spec, "kwargs", {}) or {},
        )

    # Drop previous registration if force requested.
    if already and force:
        try:
            del gym.registry[TASK_ID]
        except Exception:
            pass

    gym.register(
        id=TASK_ID,
        entry_point=NoVlmCoopTransportDirectEnv,
        kwargs=kwargs,
        disable_env_checker=True,
    )
    spec = gym.spec(TASK_ID)
    return TaskRegistration(
        task_id=TASK_ID,
        entry_point=str(spec.entry_point),
        kwargs=getattr(spec, "kwargs", {}) or {},
    )


def registration_summary(force: bool = False) -> Dict:
    reg = register_no_vlm_task(force=force)
    out = asdict(reg)
    kwargs = {}
    for k, v in (out.get("kwargs", {}) or {}).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            kwargs[k] = v
        else:
            kwargs[k] = f"{v}"
    out["kwargs"] = kwargs
    out["registered"] = True
    return out
