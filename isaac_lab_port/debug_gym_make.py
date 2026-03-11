from __future__ import annotations

import traceback

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

print("[debug] app launched")

try:
    import gymnasium as gym

    try:
        from task_registry import TASK_ID, register_no_vlm_task
        from direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
    except ImportError:
        from .task_registry import TASK_ID, register_no_vlm_task
        from .direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg

    print("[debug] imports ok")
    reg = register_no_vlm_task(force=True)
    print("[debug] registered", reg)

    cfg = NoVlmCoopTransportEnvCfg(
        scene=NoVlmSceneCfg(num_envs=2, env_spacing=8.0),
        device="cuda:0",
        sim=SimulationCfg(dt=0.02, device="cuda:0"),
    )
    print("[debug] cfg built")

    try:
        env = gym.make(TASK_ID, cfg=cfg)
        print("[debug] gym.make returned", type(env))
        raw = env.unwrapped
        robot0 = raw._robot_entities.get("robot_0")
        if robot0 is not None:
            print("[debug] robot_0 body_names", getattr(robot0, "body_names", None))
        payload = raw._payload_entity
        if payload is not None:
            print("[debug] payload body_names", getattr(payload, "body_names", None))
        env.close()
        print("[debug] env closed")
    except BaseException as exc:
        print("[debug] gym.make exception", type(exc).__name__, exc)
        traceback.print_exc()
finally:
    print("[debug] closing sim")
    simulation_app.close()
