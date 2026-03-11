from __future__ import annotations

import traceback

import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

print("[probe] app launched")

try:
    import gymnasium as gym

    try:
        from task_registry import TASK_ID, register_no_vlm_task
        from direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg
    except ImportError:
        from .task_registry import TASK_ID, register_no_vlm_task
        from .direct_marl_env import NoVlmCoopTransportEnvCfg, NoVlmSceneCfg, SimulationCfg

    register_no_vlm_task(force=True)
    cfg = NoVlmCoopTransportEnvCfg(
        scene=NoVlmSceneCfg(num_envs=2, env_spacing=8.0),
        device="cuda:0",
        sim=SimulationCfg(dt=0.02, device="cuda:0"),
    )
    env = gym.make(TASK_ID, cfg=cfg)
    raw = env.unwrapped
    print("[probe] env created")

    obs, _ = env.reset()
    print("[probe] reset ok", {k: tuple(v.shape) for k, v in obs.items()})

    robot = raw._robot_entities["robot_0"]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    hand_pose = robot.data.body_state_w[:, hand_idx, :7].clone()
    print("[probe] panda_hand idx", hand_idx)
    print("[probe] hand pose env0", hand_pose[0].tolist())

    payload = raw._payload_entity
    payload_pose = hand_pose.clone()
    payload_pose[:, 2] = payload_pose[:, 2] - 0.02
    payload_vel = torch.zeros((raw.num_envs, 6), dtype=torch.float32, device=raw.device)
    env_ids = torch.arange(raw.num_envs, dtype=torch.long, device=raw.device)
    payload.write_root_pose_to_sim(payload_pose, env_ids=env_ids)
    payload.write_root_velocity_to_sim(payload_vel, env_ids=env_ids)
    if hasattr(payload, "write_data_to_sim"):
        payload.write_data_to_sim()
    raw.scene.write_data_to_sim()
    print("[probe] payload moved onto hand")

    zero_actions = {
        name: torch.zeros((raw.num_envs, 6), dtype=torch.float32, device=raw.device)
        for name in raw.possible_agents
    }
    for i in range(10):
        env.step(zero_actions)
        sensor = raw._contact_sensors["robot_0"]
        net = getattr(sensor.data, "net_forces_w", None)
        filt = getattr(sensor.data, "force_matrix_w", None)
        print("[probe] step", i + 1)
        if isinstance(net, torch.Tensor):
            print("[probe] net", net[0].detach().cpu().tolist())
        if isinstance(filt, torch.Tensor):
            print("[probe] filt", filt[0].detach().cpu().tolist())
        print("[probe] cached", raw._contact_force_cache.get("robot_0")[0].item() if raw._contact_force_cache else None)

    env.close()
    print("[probe] env closed")
except BaseException as exc:
    print("[probe] exception", type(exc).__name__, exc)
    traceback.print_exc()
finally:
    print("[probe] closing sim")
    simulation_app.close()
