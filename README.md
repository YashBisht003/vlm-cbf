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
- LLaVA fine-tuning uses `transformers`, `peft`, and `accelerate`
- 4-bit QLoRA path uses `bitsandbytes` (Linux recommended)

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

Default runtime drive mode is `base_drive_mode="velocity"` (robust for training).
You can switch to `base_drive_mode="wheel"` for direct wheel-drive dynamics.
`kinematic_base=False` remains default in both modes.

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

## Train LLaVA-1.5-7B with LoRA (Reproducible Path)
This is the actual LLaVA+LoRA pipeline (separate from CPU proxy).
Default paper-style knobs in script:
- LoRA rank: `16`
- LoRA alpha: `32`
- Epochs: `3`

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

```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" train_mappo.py --headless --updates 1600 --steps-per-update 512 --out mappo_policy.pt --checkpoint-dir checkpoints --save-every 25 --save-latest --log-interval 10
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

Include VLM formation evaluation in suite:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_iros_suite.py --headless --device cuda --episodes 500 --updates 1600 --resume-train --run-vlm-eval --vlm-model-id llava-hf/llava-1.5-7b-hf --vlm-adapter llava_lora_out\adapter
```

## Safety Filter (CBF/QP)
The environment applies a CBF/QP safety filter per robot using OSQP.
If the solver fails, the robot executes a monitored stop (safe fallback).
The QP includes a slack variable for feasibility and clips final velocity to
the ISO speed norm bound.

## Distributed Phase + UDP
Phase synchronization supports a UDP peer-broadcast mode:
- `use_udp_phase=True`
- `use_udp_neighbor_state=True`
- `udp_base_port=<port>`

This keeps phase voting and neighbor-state exchange out of a centralized loop
and logs phase sync delay metrics in `info["phase_sync"]`.

## Vacuum End Effector Model
`carry_mode="constraint"` uses a vacuum-style fixed constraint per end effector.
Attachment requires end-effector/object proximity (`vacuum_attach_dist`), and
grasp can drop when:
- end effector drifts beyond `vacuum_break_dist` (stretch drop)
- combined grasp force capacity is below required object weight (overload drop)

These events are logged in `info["grasp"]` for evaluation/debugging.

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

Auto infer formation inside demo:
```powershell
& "C:\Users\Yash Bisht\.venvs\pybullet_vlm_cbf\Scripts\python.exe" run_demo.py --auto-vlm --vlm-backend llava --vlm-model llava-hf/llava-1.5-7b-hf --vlm-adapter llava_lora_out\adapter
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
- `train_vlm_llava_lora.py`: LLaVA LoRA fine-tuning script
- `run_policy.py`: run a trained policy in the environment
- `eval_runs.py`: batch evaluation and CSV logging
- `eval_policy_runs.py`: batch evaluation for trained policy checkpoints
- `eval_vlm_formations.py`: formation/fallback accuracy evaluation for VLM outputs
- `infer_vlm_llava.py`: LLaVA inference to JSON formation output
- `plot_results.py`: plots CSV evaluation results
- `run_iros_suite.py`: one-command IROS-style train/eval/report pipeline
- `requirements.txt`: dependencies

## Notes / Limitations
- CPU VLM (`train_vlm_cpu.py`) is a lightweight proxy, not a replacement for LLaVA fine-tuning.
- If no VLM output is provided, the environment still falls back to geometric formation.
- Reproduce VLM claims from this repo using `train_vlm_llava_lora.py` + `eval_vlm_formations.py` on a fixed split.
- Default `carry_mode="auto"` tries suction constraints first and falls back to
  kinematic carry if attachment does not stabilize within a short window.
- `carry_mode="constraint"` is the most physically faithful, but may need tuning
  for very heavy payloads or noisy contacts.
- The environment now includes approach timeout quorum fallback for phase progression
  in difficult randomized layouts (`phase_approach_timeout_s`, `phase_approach_min_ready`).
- This is an environment scaffold; it is ready to integrate with your policy.
