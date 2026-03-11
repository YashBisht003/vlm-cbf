# Isaac Lab Port (No-VLM)

This folder is the active Isaac-only path for the project. PyBullet is not part of the intended runtime here.

## Scope of this scaffold

- Uses **no VLM** in the problem loop.
- Keeps the core problem: cooperative mobile manipulation under force/safety constraints.
- Adds version-safe bootstrapping for Isaac Lab with import ordering that avoids conda + torch launch issues.

## Current files

- `no_vlm_task_spec.py`: task and backend specification (robot preset, vacuum backend, phase sequence).
- `torch_phase_manager.py`: torch-native batched phase manager (`torch.bool` inputs, `torch.int8` phase ids, `torch.float32` timers).
- `direct_marl_env.py`: DirectMARLEnv skeleton with MAPPO contracts (`possible_agents`, spaces dicts, explicit `_get_states()`).
- `vacuum_attachment.py`: PhysX AutoAttachment backend abstraction (attach/detach via attachment prims).
- `task_registry.py`: explicit gym task registration helper for this custom env.
- `train_skrl_mappo.py`: SKRL bootstrap using `load_isaaclab_env` + `wrap_env`, with optional official-script launcher.
- `agents/skrl_mappo_cfg.yaml`: SKRL MAPPO YAML config entry-point for registry.
- `check_skrl_train_script.py`: preflight heuristic checker for the known `--algorithm` NameError bug in some installs.
- `train_mappo_isaac.py`: executable bootstrap entrypoint with AppLauncher-first import order.
- `mappo_train_isaac.slurm`: starter HPC script template.

## Recommended stable stack

- Isaac Lab `v2.3.2`
- Isaac Sim `5.1`

## Local workstation setup

For this repo-local workstation flow, use:

```bash
bash isaac_lab_port/setup_local_isaac.sh
```

This creates a conda env under `./.conda_envs/` and checks out IsaacLab under `./third_party/`.

Project-side Python extras for the Isaac path live in:

```bash
isaac_lab_port/requirements_isaac_extra.txt
```

## First command to run locally

```powershell
python isaac_lab_port/train_mappo_isaac.py --print-spec-only --robot-preset ridgeback_franka --vacuum-backend gpu_fixed_joint --device cuda
```

Registry sanity check:
```powershell
python isaac_lab_port/train_mappo_isaac.py --print-task-registration
```

SKRL training bootstrap sanity check:
```powershell
python isaac_lab_port/train_skrl_mappo.py --print-only
```

Param Ganga full runtime setup (Isaac Sim 5.1 + Isaac Lab v2.3.2 + SKRL + smoke gate):
```bash
sbatch isaac_lab_port/setup_param_ganga_runtime.slurm
```

64-env smoke test (logs policy/value input shapes + contact-force gate):
```powershell
python isaac_lab_port/train_skrl_mappo.py --headless --num-envs 64 --smoke-test --smoke-steps 500 --smoke-log-every 50
```
Strict smoke gate (fail run if contact force never becomes non-zero):
```powershell
python isaac_lab_port/train_skrl_mappo.py --headless --num-envs 64 --smoke-test --smoke-steps 500 --smoke-log-every 50 --smoke-require-contact
```

Official Isaac Lab SKRL training (recommended):
```powershell
isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-NoVlm-CoopTransport-Direct-v0 --algorithm MAPPO --headless
```

## Notes on vacuum backend choice

- `cpu_surface_gripper`:
  - Keep for validation or low-scale debugging.
  - CPU-only path according to current docs/release notes.
- `gpu_fixed_joint`:
  - Intended scalable path for HPC training.
  - Mirrors the attach/detach strategy used in your PyBullet implementation.

## Next implementation step

1. Build a Direct MARL environment class in Isaac Lab with:
   - multi-robot scene instancing
   - force-safe attach/detach backend abstraction
   - no-VLM phase machine (`approach -> fine_approach -> contact -> probe -> correct -> regrip -> lift -> transport -> place -> done`)
2. Wire SKRL MAPPO trainer to that env.
3. Reuse EKF and CBF logic from the current repo in Isaac-native state/action wrappers.

## Batched phase architecture (important)

`torch_phase_manager.py` now contains:

- `TorchBatchedNoVlmPhaseManager(num_envs=...)`
- `TorchBatchedPhaseInputs` with one bool tensor per transition signal.

Design rule:

- Keep phase state per environment on device (`phase_ids[num_envs]`, `int8`).
- Keep per-environment timers on device (`phase_started_at_s`, `contact_all_attached_since_s`, `float32`).
- Call `update(inputs, now_s_vector)` each simulation step with tensors on the same device.
- In `direct_marl_env.py`, phase time is driven from Isaac env time (`episode_length_buf * step_dt`).

This avoids single-scalar phase/timer coupling and supports mixed phases across environments in parallel training.

## Dimensions

- Per-agent action dim: `6` = `3 base + 3 arm`.
- Per-agent observation dim: `61` (no zero-padding).
- Centralized state dim: `316` = `4 * 61 + 72`.

## DirectMARLEnv API contract (implemented in skeleton)

- Uses `observation_spaces`, `action_spaces`, `shared_observation_spaces` from day one.
- Does not use deprecated `num_observations` / `num_actions`.
- Uses `sim: SimulationCfg` and `scene: InteractiveSceneCfg`-style structure in env cfg.
- Keeps asset cfgs (`robot_0..robot_3`, `payload`) on the env cfg (not nested in scene cfg).
- Defines agent names as `robot_0..robot_3` in `possible_agents`.
- Uses Isaac token convention `{ENV_REGEX_NS}/Robot_i`.
- Implements explicit `_get_states()` for centralized critic state instead of implicit concat fallback.
- `_setup_scene()` includes mandatory `clone_environments(copy_from_source=False)` and CPU `filter_collisions()`.
- `_reset_idx()` includes Isaac articulation-style default root/joint state reset pattern scaffold.
- Fallback robot `ArticulationCfg` now includes:
  - `spawn=UsdFileCfg(...)` with `activate_contact_sensors=True`
  - physics property stubs (`rigid_props`, `articulation_props`)
  - `init_state` and separate implicit actuator groups (`base_velocity`, `arm_position`, `gripper_position`)
- `_apply_action()` now writes actuator commands to robots and calls `write_data_to_sim()` each step.
- Mecanum wheel commands are converted from base linear/angular velocity to wheel joint rad/s using configurable `wheel_radius_m` and `mecanum_yaw_coupling_m`.
- Arm action channels drive additive joint position targets on top of default hold pose.
- `_phase_inputs()` is now contact-driven (distance + force + attachment + payload progress) instead of all-zero placeholders.
- SKRL MAPPO config now uses `models.separate: true` so the value network is instantiated against centralized shared state space.
- SKRL MAPPO agent config enables `RunningStandardScaler` for `state_preprocessor`, `shared_state_preprocessor`, and `value_preprocessor`.
- SKRL MAPPO agent config enables `KLAdaptiveLR` with `kl_threshold: 0.008`.
- `state()` returns a global tensor for current SKRL wrapper compatibility; `state_dict()` is available as a helper for dict-style integrations.
- Observation/state reads explicitly refresh entity buffers via per-entity `update(dt)` before tensor extraction.
