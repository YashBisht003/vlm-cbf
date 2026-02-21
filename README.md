# VLM-CBF: Vision-Language Guided Safe Multi-Robot Cooperative Manipulation

Official implementation of the VLM–CBF–MAPPO framework for heterogeneous multi-robot cooperative object transport under safety constraints.

This project integrates:

- Vision-Language Model (VLM) semantic formation reasoning
- Bayesian probe-and-correct load estimation
- Graph Neural Network (GNN) decentralized control
- Multi-Agent PPO (MAPPO) with CTDE training
- Control Barrier Function (CBF) safety filtering
- ISO-compliant force and speed constraints

---

## 🔬 Problem Overview

Robots must cooperatively lift and transport heavy objects with unknown mass distribution.

Key challenges:
- Hidden center-of-mass (COM) offsets
- Heterogeneous robot capacities
- Safety constraints during contact-rich manipulation
- Decentralized coordination

We address this using a two-stage reasoning approach:

1. **Semantic prior (VLM)** → multi-hypothesis formation initialization  
2. **Physics verification (force probe)** → Bayesian selection + residual correction  
3. **Safe decentralized control (GNN + MAPPO + CBF)** → coordinated transport  

---

## 🏗 System Architecture

Pipeline:

1. RGB-D perception → VLM generates K formation hypotheses
2. Robots approach object
3. Probe phase (partial lift)
4. Bayesian hypothesis selection
5. Residual waypoint correction
6. MAPPO-GNN decentralized transport
7. CBF safety filtering (hard constraints)

Training uses:
- Centralized critic (CTDE)
- Shared actor weights
- Intervention-aware reward shaping

Deployment:
- Fully decentralized execution
- No centralized critic
- No server-side VLM during control loop

---

## ⚙️ Installation

### 1. Create Environment

```bash
conda create -n vlmcbf python=3.11
conda activate vlmcbf
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch
- PyBullet
- NumPy
- SciPy

---

## 🚀 Training (Cluster / HPC Recommended)

This project is simulation-heavy and designed for parallel environments.

### Example (Headless Training)

```bash
python train_mappo.py \
    --num-envs 32 \
    --total-timesteps 10_000_000 \
    --device cuda \
    --headless
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--num-envs` | Number of parallel simulation environments |
| `--device` | `cpu` or `cuda` |
| `--headless` | Run without GUI |
| `--resume` | Resume from latest checkpoint |
| `--checkpoint-dir` | Directory for saving models |

For HPC (e.g., Param Ganga), adjust:
- `--num-envs` according to CPU cores
- Use `--headless`
- Enable GPU if available

---

## 🎮 Demo Mode

Run interactive simulation:

```bash
python demo.py --mode interactive
```

For headless evaluation:

```bash
python demo.py --mode eval --checkpoint checkpoints/best.pt
```

---

## 🧠 Probe-and-Correct Phase

During contact:

1. Adaptive probe force computed from worst-case load hypothesis
2. Ground unloading verified
3. Load fractions measured via F/T sensors
4. Bayesian hypothesis selection performed
5. Residual MLP applies fine waypoint correction

This reduces COM estimation error before full lift.

---

## 🛡 Safety Mechanism (CBF-QP)

All control outputs pass through:

```
GNN Actor → Proposed action
          → CBF-QP Safety Filter
          → Safe action
```

Safety guarantees:
- Force limits
- Velocity limits
- Collision avoidance
- Load capacity constraints

Intervention penalty included in reward to reduce clipping frequency.

---

## 📊 Training Objective

Reward components:

- Task progress
- Load balance
- CBF proximity shaping
- CBF intervention penalty
- Sparse success bonus

---

## 📈 Logging

Logs stored in:

```
logs/
```

Includes:
- Episode reward
- CBF intervention frequency
- Load imbalance metric
- Success rate

TensorBoard:

```bash
tensorboard --logdir logs
```

---

## 💻 Recommended Hardware

For large-scale training:

- 16–64 CPU cores (parallel environments)
- 32–128 GB RAM
- Optional GPU for policy optimization

---


```

---

## 📌 Notes

- Designed for research use.
- Simulation-only validation.
- Real-world deployment requires additional calibration and hardware validation.

---

## 🤝 Acknowledgments

Developed as part of research on safe multi-agent reinforcement learning for cooperative robotics.


```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_vlm_llava_lora.py --model-id llava-hf/llava-1.5-7b-hf --train-jsonl vlm_dataset\train.jsonl --val-jsonl vlm_dataset\val.jsonl --image-root vlm_dataset --out llava_lora_out --epochs 3 --lora-rank 16 --lora-alpha 32 --use-4bit --fp16
```

Inference with LoRA adapter:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" infer_vlm_llava.py --model-id llava-hf/llava-1.5-7b-hf --adapter llava_lora_out\adapter --image vlm_dataset\images\sample_000000_view_00.png --out llava_formation.json
```

Formation metric evaluation:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_vlm_formations.py --jsonl vlm_dataset\val.jsonl --image-root vlm_dataset --model-id llava-hf/llava-1.5-7b-hf --adapter llava_lora_out\adapter --out-csv vlm_eval_samples.csv --out-json vlm_eval_metrics.json
```

Zero-shot vs fine-tuned comparison:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_vlm_formations.py --jsonl vlm_dataset\val.jsonl --image-root vlm_dataset --model-id llava-hf/llava-1.5-7b-hf --out-json vlm_eval_zeroshot.json
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_vlm_formations.py --jsonl vlm_dataset\val.jsonl --image-root vlm_dataset --model-id llava-hf/llava-1.5-7b-hf --adapter llava_lora_out\adapter --out-json vlm_eval_finetuned.json
```

## Train GNN Policy (MAPPO)
This trains a decentralized GNN policy with a centralized critic (CTDE).
It can run on CPU, but will be much faster on a GPU.
The critic input includes:
- concatenated per-agent observations
- explicit global state (`object pose`, `goal`, `phase`, `belief mean/cov`, force stats)
- per-agent observation uses the paper-style 101D structure
  (`ego`, `object-relative`, `forces`, `goal`, `neighbors`, `belief`, `safety`, `phase`)

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 1600 --steps-per-update 512 --out mappo_policy.pt --checkpoint-dir checkpoints --save-every 25 --save-latest --log-interval 10
```

Parallel rollout collection (recommended on multi-core CPU/GPU nodes):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --num-envs 8 --updates 1600 --steps-per-update 512 --checkpoint-dir checkpoints
```

Optional neural-CBF pretrain warm start (recommended):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 1600 --steps-per-update 512 --train-neural-cbf --pretrain-neural-cbf-steps 20000 --pretrain-neural-cbf-epochs 3 --checkpoint-dir checkpoints
```

Resume from latest checkpoint:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --resume --checkpoint-dir checkpoints --save-latest
```

Resume from best checkpoint:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --resume-best --checkpoint-dir checkpoints --save-latest --best-metric success_rate
```

CPU-focused run (limit Torch threads explicitly):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 1600 --steps-per-update 384 --torch-threads 4 --checkpoint-dir checkpoints --save-latest
```

GPU-focused run:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --device cuda --updates 1600 --steps-per-update 512 --checkpoint-dir checkpoints --save-latest
```

Train learned residual correction model (used in `Correct` phase):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_residual_corrector.py --out residual_corrector.joblib --samples 40000
```

Generate residual dataset from simulation and train on it:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_residual_dataset.py --out residual_dataset.csv --episodes 3000 --headless
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_residual_corrector.py --dataset-csv residual_dataset.csv --out residual_corrector.joblib
```

Use learned residual model during training/evaluation:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 1600 --residual-model residual_corrector.joblib
```

Checkpoint behavior:
- `mappo_policy_latest.pt` is saved every update (unless `--no-save-latest`).
- `mappo_policy_update_XXXX.pt` is saved every `--save-every` updates.
- `mappo_policy_best.pt` is updated when `--best-metric` improves.
- Resume searches latest, then latest numbered checkpoint, then final output.
- Save verification is enabled by default (`--verify-checkpoints` / `--no-verify-checkpoints`).

Run the trained policy:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_policy.py --model mappo_policy.pt --headless --deterministic
```

Evaluate trained policy over many episodes:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_policy_runs.py --model checkpoints\mappo_policy_best.pt --episodes 500 --headless --out eval_learned_policy.csv
```

## IROS Suite
Run full train/eval/report pipeline:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_iros_suite.py --headless --device cuda --episodes 500 --updates 1600 --resume-train
```

This script runs:
- MAPPO training (or resume)
- Learned policy evaluation
- Heuristic baseline evaluation (with and without CBF)
- Plot generation + summary tables
- `suite_summary.csv` and `suite_summary.md`

Include learned-policy neural CBF ablation:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_iros_suite.py --headless --device cuda --episodes 500 --updates 1600 --resume-train --run-neural-ablation
```

Include probe/correct ablations (`no-probe`, `probe+heuristic`, `probe+learned`):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_iros_suite.py --headless --device cuda --episodes 500 --updates 1600 --resume-train --generate-residual-dataset --train-residual-model --run-probe-ablation
```

Include VLM formation evaluation in suite:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_iros_suite.py --headless --device cuda --episodes 500 --updates 1600 --resume-train --run-vlm-eval --vlm-model-id llava-hf/llava-1.5-7b-hf --vlm-adapter llava_lora_out\adapter
```

## Safety Filter (CBF/QP)
The environment applies a CBF/QP safety filter per robot using OSQP.
If the solver fails, the robot executes a monitored stop profile (`v <- v - k*dt*v`).
The QP includes a slack variable for feasibility and clips final velocity to
the ISO speed norm bound.
Classical CBF constraints in the same QP include:
- speed limit
- robot-robot separation
- contact-force barrier (`dB_force/dt + alpha*B_force >= 0`, belief-adaptive via EKF uncertainty)

Neural force barrier integration (`h_phi([F_i, mu_b, Sigma_b])`):
- The neural barrier is evaluated from force + EKF belief.
- A linearized neural-CBF inequality is added to the same QP (`a^T v + s >= b`) using autograd `dh_phi/dv` at the current control point.
- By default, extra neural speed/separation shaping is disabled (`neural_cbf_tighten_gain=0`, `neural_cbf_sigmoid_gain=0`) so the neural term enters primarily through the QP inequality.
- Online neural-CBF training uses temporal rollout residuals plus safe/unsafe sign regularization, including unsafe transitions.

Belief EKF state (paper-aligned):
- `x_obj = [m, p_COM(x,y,z), I_diag(xx,yy,zz)]`
- belief covariance is used for risk-adaptive CBF tightening.

## Distributed Phase + UDP
Phase synchronization supports a UDP peer-broadcast mode:
- `use_udp_phase=True`
- `use_udp_neighbor_state=True`
- `udp_base_port=<port>`

This keeps phase voting and neighbor-state exchange out of a centralized loop
and logs phase sync delay metrics in `info["phase_sync"]`.
Execution is decentralized but communication-enabled (lightweight P2P UDP),
not communication-free.
Default behavior is strict all-robot consensus; timeout quorum fallback is optional
(`phase_allow_quorum_fallback=True`).

## Vacuum End Effector Model
`carry_mode="constraint"` uses a vacuum-style fixed constraint per end effector.
Attachment requires end-effector/object proximity (`vacuum_attach_dist`), and
grasp can drop when:
- end effector drifts beyond `vacuum_break_dist` (stretch drop)
- combined grasp force capacity is below required object weight (overload drop)

These events are logged in `info["grasp"]` for evaluation/debugging.

## VLM JSON Input (Optional)
You can provide VLM formation output as JSON. The environment accepts either:
- single formation (`waypoints`, `load_labels`, `confidence`)
- multi-hypothesis list (`hypotheses`), where each hypothesis includes
  `waypoints`, `load_labels`, `confidence`, and optional `load_fractions`.

Example:
```json
{
  "hypotheses": [
    {
      "confidence": 0.56,
      "load_fractions": [0.35, 0.35, 0.15, 0.15],
      "waypoints": [
        {"x": 0.45, "y": 0.00, "load": "high"},
        {"x": -0.45, "y": 0.00, "load": "high"},
        {"x": 0.00, "y": 0.30, "load": "low"},
        {"x": 0.00, "y": -0.30, "load": "low"}
      ]
    },
    {
      "confidence": 0.27,
      "load_fractions": [0.48, 0.22, 0.15, 0.15],
      "waypoints": [
        {"x": 0.52, "y": 0.00, "load": "high"},
        {"x": -0.30, "y": 0.00, "load": "high"},
        {"x": 0.00, "y": 0.30, "load": "low"},
        {"x": 0.00, "y": -0.30, "load": "low"}
      ]
    }
  ]
}
```

Use it with:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --vlm-json path\to\formation.json
```

Auto infer formation inside demo:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --auto-vlm --auto-vlm-mode robot4 --vlm-backend llava --vlm-model llava-hf/llava-1.5-7b-hf --vlm-adapter llava_lora_out\adapter --vlm-fusion-k 3
```

Multi-view fusion from 4 robot images (standalone):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" infer_vlm_multiview.py --backend llava --model-id llava-hf/llava-1.5-7b-hf --adapter llava_lora_out\adapter --images view0.png view1.png view2.png view3.png --k 3 --out formation_multiview.json
```

## Quantec URDF Notes
If you use a KUKA Quantec URDF/xacro, prefer passing a tool-frame link name
like `tool0` or `flange`:
`--heavy-ee-link-name tool0` (recommended), or use `--heavy-ee-link <index>`.

## Recommended VLM Training Plan (IROS-ready)
Model choice (recommended): **LLaVA-1.5-7B with LoRA** (matches the paper spec).
Training steps:
1. Generate/collect labeled object images (use `generate_vlm_dataset.py`, then replace labels with expert labels).
2. Format as LLaVA conversations (already done in `annotations.jsonl`).
3. Fine-tune with LoRA (rank 16, alpha 32, 3 epochs is a good starting point).
4. Export JSON formation outputs for inference and use with `--vlm-json` or integrate directly.

If you prefer a different VLM (e.g., Qwen2-VL), tell me and I'll adapt the dataset format.

## Object Size vs Robot Size
Object dimensions are scaled to be **1x-3x the robot visual size** by default.
This is controlled by:
- `object_size_ratio=(1.0, 3.0)`
- `robot_size_mode="base"` (base link AABB), or `"full"` (whole robot AABB)

If you change these settings, regenerate the dataset before training.

## Files
- `vlm_cbf_env.py`: main environment and simulation logic
- `run_demo.py`: launches a GUI demo with a simple coordinator policy
- `marl_obs.py`: observation builder for multi-agent policy training
- `gnn_policy.py`: GNN policy and centralized critic definitions
- `train_mappo.py`: MAPPO-style training script
- `generate_residual_dataset.py`: creates probe/residual training rows from PyBullet scenes
- `train_residual_corrector.py`: trains lightweight residual waypoint correction MLP
- `train_vlm_llava_lora.py`: LLaVA LoRA fine-tuning script
- `run_policy.py`: run a trained policy in the environment
- `eval_runs.py`: batch evaluation and CSV logging
- `eval_policy_runs.py`: batch evaluation for trained policy checkpoints
- `eval_vlm_formations.py`: formation/fallback accuracy evaluation for VLM outputs
- `infer_vlm_llava.py`: LLaVA inference to JSON formation output
- `infer_vlm_multiview.py`: multi-view inference and fusion into ranked hypotheses JSON
- `vlm_multiview.py`: reusable multi-view fusion logic used by demo/inference
- `plot_results.py`: plots CSV evaluation results
- `run_iros_suite.py`: one-command IROS-style train/eval/report pipeline
- `requirements.txt`: dependencies

## Notes / Limitations
- CPU VLM (`train_vlm_cpu.py`) is a lightweight proxy, not a replacement for LLaVA fine-tuning.
- If no VLM output is provided, the environment still falls back to geometric formation.
- Reproduce VLM claims from this repo using `train_vlm_llava_lora.py` + `eval_vlm_formations.py` on a fixed split.
- Dataset training treats each view as an independent supervised sample; runtime can fuse 4 observe-phase robot views into one ranked hypothesis JSON (`infer_vlm_multiview.py` or `run_demo.py --auto-vlm-mode robot4`).
- Multi-hypothesis probe-and-correct is implemented in simulation:
  Bayesian hypothesis selection from early force ratios, then waypoint residual correction before lift.
- The neural CBF currently uses a linearized inequality in QP with a local
  dynamics surrogate and autograd gradient; this is a practical implementation,
  not a full formal proof pipeline for forward invariance.
- Online neural-CBF training is rollout-supervised from temporal safety
  residuals and safety constraints; for stronger claims, use a dedicated
  safe/unsafe dataset and ablation protocol.
- Default `carry_mode="auto"` tries suction constraints first and falls back to
  kinematic carry if attachment does not stabilize within a short window.
- `carry_mode="constraint"` is the most physically faithful, but may need tuning
  for very heavy payloads or noisy contacts.
- Approach timeout quorum fallback is optional, disabled by default for strict
  consensus transitions (`phase_allow_quorum_fallback=False`).
- Formation assignment and phase progression are structured (VLM/Hungarian + consensus FSM);
  MAPPO learns cooperative low-level control within this scaffold, rather than the task graph itself.
- This is an environment scaffold; it is ready to integrate with your policy.

