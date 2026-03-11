# VLM-CBF

Research code for cooperative multi-robot object transport with belief-aware safety filtering.

The repo currently contains two code paths:

- Active Isaac Lab path: [`isaac_lab_port/`](isaac_lab_port/)
- Legacy PyBullet path: root-level environment and training files

The Isaac path is the current development target. In that path, the VLM stages are removed and the focus is on cooperative transport, belief tracking, MAPPO training, and CBF-based safety.

## What Is In This Repo

- `belief_ekf.py`
  Belief-space EKF over hidden object properties `[mass, com_x, com_y, com_z, Ixx, Iyy, Izz]`
- `cbf_qp.py`
  Control-barrier-function safety filter
- `neural_cbf.py`
  Neural barrier runtime utilities
- `residual_corrector.py`
  Residual correction model/runtime
- `isaac_lab_port/direct_marl_env.py`
  Isaac Lab multi-agent environment
- `isaac_lab_port/train_skrl_mappo.py`
  Isaac smoke tests and bring-up utilities
- `isaac_lab_port/run_official_skrl_train.py`
  Wrapper for official Isaac Lab SKRL MAPPO training
- `validate_belief_ekf.py`
  Repo-local EKF validation script

## Belief EKF

The belief filter does not consume direct mass or COM sensors. It infers latent object properties from:

- per-robot vertical load measurements `forces_z`
- robot planar support locations
- object planar center estimate

That is why the filter exists: the controller needs latent load properties and uncertainty, not raw force readings.

### What It Does Well

- Tracks mass and planar COM from force-sharing observations
- Produces covariance that can be used for risk-aware CBF tightening
- Exposes diagnostics, including NIS and inertia observability

### Important Limitation

Under the current measurement model, inertia is not strongly observed from the available force measurements. In this repo, inertia states are kept for interface compatibility, but they are prior-anchored when the Jacobian shows no useful observability. This is deliberate: the code should not pretend to estimate inertia confidently when the measurements do not support that claim.

In short:

- Mass and COM estimation: supported
- Inertia estimation: weak / not a primary claim in the current implementation

## EKF Validation

Run:

```bash
python validate_belief_ekf.py
```

Current validation checks:

- mass / COM convergence
- robustness to noisy force measurements
- NIS calibration
- bounded behavior under unobservable inertia

The current patched filter uses:

- adaptive process noise for static latent parameters
- adaptive measurement covariance based on total supported load
- inertia prior regularization when observability is effectively zero

## Isaac Lab Setup

The repo-local Isaac environment used here is:

- Isaac Sim `5.1.0`
- Isaac Lab `v2.3.2`
- SKRL `1.4.3`

The active local environment on this machine is:

- `/home/ub/yash_projects/vlm-cbf/.conda_envs/isaaclab_232`

Isaac Lab source checkout used by this repo:

- `/home/ub/yash_projects/vlm-cbf/third_party/IsaacLab`

## Isaac Bring-Up

Task registration check:

```bash
source /home/ub/anaconda3/etc/profile.d/conda.sh
conda activate /home/ub/yash_projects/vlm-cbf/.conda_envs/isaaclab_232
cd /home/ub/yash_projects/vlm-cbf/third_party/IsaacLab
./isaaclab.sh -p /home/ub/yash_projects/vlm-cbf/isaac_lab_port/train_mappo_isaac.py --print-task-registration
```

Smoke test:

```bash
./isaaclab.sh -p /home/ub/yash_projects/vlm-cbf/isaac_lab_port/train_skrl_mappo.py --headless --num-envs 8 --smoke-test --smoke-steps 25 --smoke-policy scripted_approach
```

Official MAPPO train bring-up:

```bash
./isaaclab.sh -p /home/ub/yash_projects/vlm-cbf/isaac_lab_port/run_official_skrl_train.py --task Isaac-NoVlm-CoopTransport-Direct-v0 --algorithm MAPPO --headless --num_envs 8 --max_iterations 2
```

## Current Status

Working:

- Isaac Lab environment construction
- task registration
- SKRL wrapper integration
- official MAPPO training bootstrap
- contact sensing on `panda_hand`
- belief EKF integration into Isaac env

Not solved yet:

- strong contact persistence under simple scripted control
- convincing inertia estimation from current force-only likelihood
- real hardware sensor validation

## Legacy Path

The root-level PyBullet path is still present for reference, but it is not the preferred path for ongoing development. If you are extending the current system, work in [`isaac_lab_port/`](isaac_lab_port/).

## Notes for Papers / Reports

The defensible claim for the current belief module is:

- the filter maintains a calibrated belief over mass and COM from force-sharing observations, and that covariance can be used downstream for adaptive safety margins

The non-defensible claim, without further work, is:

- accurate inertia estimation from the current measurement stack alone

If this repo is cited, that distinction should be kept explicit.
