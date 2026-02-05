# VLM-CBF PyBullet Environment

This project provides a PyBullet simulation environment that matches the
VLM-CBF cooperative manipulation spec:
- 4 mobile manipulators (2 heavy, 2 agile)
- L/T/cuboid objects with randomized mass, COM, and friction
- Phase-based coordination (Observe -> Plan -> Approach -> Contact -> Lift -> Transport -> Place)
- Conflict-free task allocation (Hungarian)
- Safety monitors (speed, separation, contact force)

The environment is intentionally modular so you can plug in your own VLM,
GNN, and CBF implementations. The default runner includes simple heuristic
controllers and a kinematic carry model for stability.

## Requirements
- Python 3.11+
- Windows: PyBullet may require Microsoft C++ Build Tools to build from source
- OSQP is used for the CBF/QP safety filter

## Setup
Use the external venv you created:

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" -m pip install -r requirements.txt
```

If PyBullet fails to install, install Microsoft C++ Build Tools and re-run the
command above.

## Run Demo
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py
```

Optional arguments:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --carry-mode auto --seed 42
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --headless --steps 2000
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --pos-noise 0.01 --yaw-noise 0.02 --force-noise 2 --noisy-obs --noisy-control
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --heavy-urdf path\to\quantec.urdf --heavy-ee-link-name tool0
```

## Mobile Base Dynamics
By default, the simulator uses a differential-drive base for agile robots and an omni base
for heavy robots. The base URDFs live in `assets/` and can be swapped via:
- `base_diff_urdf` and `base_omni_urdf` in `TaskConfig`

The base controller uses wheel velocity control (no kinematic teleport) when
`kinematic_base=False` (default).

## Evaluate
Run multiple episodes and write a CSV summary:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_runs.py --episodes 50 --headless --out eval_results.csv
```

Plot the CSV results:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" plot_results.py --csv eval_results.csv --out eval_plots.png
```

Record MP4 videos every 100 episodes:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" eval_runs.py --episodes 500 --video-every 100 --video-dir videos
```

## Generate VLM Dataset (Synthetic)
This creates image/JSON pairs using the geometric fallback as a proxy label.
Replace with expert labels for your final IROS dataset.

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_vlm_dataset.py --samples 500 --out vlm_dataset
```

Multi-view per object (recommended for VLM fine-tuning):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_vlm_dataset.py --samples 500 --views 4 --out vlm_dataset
```

Best fine-tuning recipe (multi-camera + randomization):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_vlm_dataset.py --samples 2000 --views 6 --multi-camera --random-lighting --random-color --out vlm_dataset
```

Add multi-object scenes and auto train/val split:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_vlm_dataset.py --samples 2000 --views 6 --multi-camera --random-lighting --random-color --distractors 3 --distractor-random --val-ratio 0.1 --out vlm_dataset
```

Object size scaling (1x-3x robot size, visual):
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" generate_vlm_dataset.py --samples 2000 --views 6 --multi-camera --random-lighting --random-color --distractors 3 --distractor-random --val-ratio 0.1 --robot-size-mode base --size-ratio-min 1.0 --size-ratio-max 3.0 --out vlm_dataset
```

## Train CPU VLM-like Regressor (i5-friendly)
This trains a lightweight HOG+Ridge model that outputs the same JSON format as the VLM.

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_vlm_cpu.py --data vlm_dataset --out cpu_vlm_model.joblib
```

Run inference on a single image:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" infer_vlm_cpu.py --model cpu_vlm_model.joblib --image vlm_dataset\images\sample_000000_view_00.png --out formation.json
```

## Train GNN Policy (MAPPO)
This trains a decentralized GNN policy with a centralized critic (CTDE).
It can run on CPU, but will be much faster on a GPU.

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 200 --steps-per-update 512 --out mappo_policy.pt --checkpoint-dir checkpoints --save-every 25 --save-latest
```

Resume from latest checkpoint:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --resume --checkpoint-dir checkpoints --save-latest
```

Resume from best checkpoint:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --resume-best --checkpoint-dir checkpoints --save-latest --best-metric success_rate
```

Run the trained policy:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_policy.py --model mappo_policy.pt --headless --deterministic
```

## Safety Filter (CBF/QP)
The environment applies a CBF/QP safety filter per robot using OSQP.
If the solver fails, the robot executes a monitored stop (safe fallback).

## VLM JSON Input (Optional)
You can provide VLM formation output as JSON. The environment expects 4 waypoints
in the object frame, plus load labels and an optional confidence score.

Example:
```json
{
  "confidence": 0.82,
  "waypoints": [
    {"x": 0.45, "y": 0.00, "load": "high"},
    {"x": -0.45, "y": 0.00, "load": "high"},
    {"x": 0.00, "y": 0.30, "load": "low"},
    {"x": 0.00, "y": -0.30, "load": "low"}
  ]
}
```

Use it with:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --vlm-json path\to\formation.json
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
- `run_policy.py`: run a trained policy in the environment
- `eval_runs.py`: batch evaluation and CSV logging
- `plot_results.py`: plots CSV evaluation results
- `requirements.txt`: dependencies

## Notes / Limitations
- The VLM formation is a stub (confidence-based fallback is implemented).
- Default `carry_mode="auto"` tries suction constraints first and falls back to
  kinematic carry if attachment does not stabilize within a short window.
- `carry_mode="constraint"` is the most physically faithful, but may need tuning
  for very heavy payloads or noisy contacts.
- This is an environment scaffold; it is ready to integrate with your policy.
