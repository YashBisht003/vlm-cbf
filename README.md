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

## 📄 Citation

If you use this work:

```bibtex
@inproceedings{vlmcbf2026,
  title={Vision-Language Guided Safe Multi-Robot Cooperative Manipulation via Probe-and-Correct},
  author={Bisht, Yash and ...},
  booktitle={IEEE/RSJ IROS},
  year={2026}
}
```

---

## 📌 Notes

- Designed for research use.
- Simulation-only validation.
- Real-world deployment requires additional calibration and hardware validation.

---

## 🤝 Acknowledgments

Developed as part of research on safe multi-agent reinforcement learning for cooperative robotics.

