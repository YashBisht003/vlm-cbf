#!/bin/bash
#SBATCH --job-name=isaac_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=isaac_setup_%j.out
#SBATCH --error=isaac_setup_%j.err
#SBATCH --chdir=/home/y_bisht_me.iitr/projects/vlm-cbf
#SBATCH --export=ALL

set -euo pipefail

echo ">>> Isaac setup job started on $(hostname) at $(date)"

# Ensure conda init is available for non-interactive shell.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

bash isaac_lab_port/setup_param_ganga_runtime.sh \
  --env-name isaaclab_232 \
  --isaaclab-dir "$HOME/IsaacLab" \
  --project-dir "/home/y_bisht_me.iitr/projects/vlm-cbf"

echo ">>> Isaac setup job finished at $(date)"
