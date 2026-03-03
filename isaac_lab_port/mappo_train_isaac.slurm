#!/bin/bash
#SBATCH --job-name=isaac_mappo_bootstrap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=isaac_mappo_%j.out
#SBATCH --error=isaac_mappo_%j.err
#SBATCH --chdir=/home/y_bisht_me.iitr/projects/vlm-cbf
#SBATCH --export=ALL

set -euo pipefail

echo ">>> Started on $(hostname) at $(date)"

# Update this path to your Isaac Lab python executable.
PYTHON_BIN="$HOME/miniconda3/envs/isaaclab_232/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected python not found at ${PYTHON_BIN}"
  exit 1
fi

echo ">>> Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

# Sanity check task registration + env config contract.
"${PYTHON_BIN}" isaac_lab_port/train_mappo_isaac.py \
  --print-task-registration

"${PYTHON_BIN}" isaac_lab_port/train_mappo_isaac.py \
  --print-env-contract \
  --device cuda

# Preflight for known SKRL train.py '--algorithm' bug in some Isaac Lab installs.
"${PYTHON_BIN}" isaac_lab_port/check_skrl_train_script.py

# SKRL MAPPO bootstrap (prints resolved setup).
"${PYTHON_BIN}" isaac_lab_port/train_skrl_mappo.py \
  --headless \
  --print-only

# 64-env / 500-step smoke test gate before full training.
"${PYTHON_BIN}" isaac_lab_port/train_skrl_mappo.py \
  --headless \
  --num-envs 64 \
  --smoke-test \
  --smoke-steps 500 \
  --smoke-log-every 50 \
  --smoke-require-contact

# To run full official training path (once environment install is verified), use:
# "${PYTHON_BIN}" isaac_lab_port/train_skrl_mappo.py --headless --run-official-script

echo ">>> Bootstrap finished at $(date)"
