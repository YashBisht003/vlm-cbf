#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: setup failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.conda_envs/isaaclab_232"
ISAACLAB_DIR="${ROOT_DIR}/third_party/IsaacLab"
PYTHON_VERSION="3.11"
SKIP_ISAACSIM=0

usage() {
  cat <<EOF
Usage:
  bash isaac_lab_port/setup_local_isaac.sh [options]

Options:
  --env-prefix PATH      Conda env prefix path (default: ${ENV_PREFIX})
  --isaaclab-dir PATH    Isaac Lab checkout path (default: ${ISAACLAB_DIR})
  --python-version VER   Python version for conda env (default: ${PYTHON_VERSION})
  --skip-isaacsim        Skip isaacsim pip install step
  -h, --help             Show this help

Notes:
  - Intended for this local workstation path, not Param Ganga.
  - Keeps the conda env and IsaacLab checkout inside this repo by default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --isaaclab-dir)
      ISAACLAB_DIR="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --skip-isaacsim)
      SKIP_ISAACSIM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH."
  exit 2
fi

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION-}"

echo ">>> Host: $(hostname)"
echo ">>> Date: $(date)"
echo ">>> GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

set +u
eval "$(conda shell.bash hook)"
set -u

mkdir -p "$(dirname "${ENV_PREFIX}")"
mkdir -p "$(dirname "${ISAACLAB_DIR}")"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo ">>> Creating project-local conda env at ${ENV_PREFIX}"
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

set +u
conda activate "${ENV_PREFIX}"
set -u
python -m pip install -U pip setuptools wheel

echo ">>> Installing Torch (CUDA 12.8 wheels)"
python -m pip install \
  "torch==2.7.0" \
  "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128

if [[ "${SKIP_ISAACSIM}" -eq 0 ]]; then
  echo ">>> Installing Isaac Sim 5.1.0 python packages"
  python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
fi

if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
  echo ">>> Shallow-cloning IsaacLab v2.3.2 into ${ISAACLAB_DIR}"
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

cd "${ISAACLAB_DIR}"
git fetch --depth 1 origin tag v2.3.2 || true
git checkout v2.3.2
git submodule update --init --recursive

if ./isaaclab.sh --help | grep -q -- "--install"; then
  ./isaaclab.sh --install
  ./isaaclab.sh --install skrl
else
  ./isaaclab.sh -i skrl
fi

cd "${ROOT_DIR}"
export ISAACLAB_PATH="${ISAACLAB_DIR}"

echo ">>> Installing project-side Isaac extras"
python -m pip install -r isaac_lab_port/requirements_isaac_extra.txt

echo ">>> Registration check"
python isaac_lab_port/train_skrl_mappo.py --print-only

echo ">>> Setup completed"
echo ">>> Activate later with: conda activate ${ENV_PREFIX}"
