from __future__ import annotations

import sys

from isaaclab.app import AppLauncher


def main() -> None:
    launcher = AppLauncher(headless=True)
    sim_app = launcher.app
    try:
        import gymnasium as gym

        from direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
        from task_registry import TASK_ID, registration_summary

        registration_summary(force=True)
        cfg = NoVlmCoopTransportEnvCfg(
            scene=NoVlmSceneCfg(num_envs=2, env_spacing=8.0),
            device="cuda:0",
            sim=SimulationCfg(dt=0.02, device="cuda:0"),
        )
        env = gym.make(TASK_ID, cfg=cfg)
        raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
        payload = getattr(raw_env, "_payload_entity", None)
        print(f"payload_type={type(payload).__name__}", flush=True)
        if payload is None:
            print("payload_missing=true", flush=True)
            return
        masses = None
        if hasattr(payload, "root_physx_view") and payload.root_physx_view is not None:
            view = payload.root_physx_view
            if hasattr(view, "get_masses"):
                masses = view.get_masses()
                print(f"root_physx_view_masses={masses}", flush=True)
        data = getattr(payload, "data", None)
        if data is not None:
            for attr in ("default_root_state", "root_state_w"):
                if hasattr(data, attr):
                    obj = getattr(data, attr)
                    print(f"data_attr_{attr}_shape={getattr(obj, 'shape', None)}", flush=True)
        if masses is None:
            print("mass_query_unavailable=true", flush=True)
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    sys.exit(main())
