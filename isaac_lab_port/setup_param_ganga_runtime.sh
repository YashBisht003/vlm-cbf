#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: setup failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

usage() {
  cat <<'EOF'
Usage:
  bash isaac_lab_port/setup_param_ganga_runtime.sh [options]

Options:
  --env-name NAME          Conda env name (default: isaaclab_232)
  --env-prefix PATH        Conda env prefix path (default: PROJECT_DIR/.conda_envs/ENV_NAME)
  --isaaclab-dir PATH      Isaac Lab source directory (default: $HOME/IsaacLab)
  --project-dir PATH       Project repo root (default: current working directory)
  --python-version VER     Python version for conda env (default: 3.11)
  --skip-smoke             Skip 64-env smoke test
  --skip-isaacsim          Skip isaacsim pip install step
  -h, --help               Show this help

Notes:
  - Intended for Linux GPU nodes (Param Ganga).
  - Installs Isaac Sim 5.1 and Isaac Lab v2.3.2 from source.
  - Defaults keep all writable artifacts inside PROJECT_DIR.
EOF
}

ENV_NAME="isaaclab_232"
ENV_PREFIX=""
PROJECT_DIR="$(pwd)"
ISAACLAB_DIR="${PROJECT_DIR}/third_party/IsaacLab"
PYTHON_VERSION="3.11"
SKIP_SMOKE=0
SKIP_ISAACSIM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --env-prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --isaaclab-dir)
      ISAACLAB_DIR="$2"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
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

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this setup script is for Linux only."
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run this on a GPU node."
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH."
  echo "Hint: source your conda init script before running this setup."
  exit 2
fi

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION-}"

echo ">>> Host: $(hostname)"
echo ">>> Date: $(date)"
echo ">>> GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GLIBC_VER="$(ldd --version 2>/dev/null | sed -nE '1 s/.* ([0-9]+\.[0-9]+).*/\1/p')"
if [[ -z "${GLIBC_VER}" ]]; then
  echo "ERROR: failed to parse GLIBC version from ldd --version output."
  exit 2
fi
echo ">>> GLIBC: ${GLIBC_VER}"
python3 - <<PY
import sys
glibc = tuple(int(x) for x in "${GLIBC_VER}".split("."))
if glibc < (2, 35):
    print("ERROR: GLIBC < 2.35. Isaac Sim 5.1 pip install requires >= 2.35.")
    print("Use Isaac Sim binary workflow on this node/image.")
    sys.exit(2)
print(">>> GLIBC check passed (>= 2.35)")
PY

# Activate conda shell functions.
set +u
eval "$(conda shell.bash hook)"
set -u

if [[ -z "${ENV_PREFIX}" ]]; then
  ENV_PREFIX="${PROJECT_DIR}/.conda_envs/${ENV_NAME}"
fi

mkdir -p "$(dirname "${ENV_PREFIX}")"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo ">>> Creating project-local conda env: ${ENV_PREFIX} (python=${PYTHON_VERSION})"
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

set +u
conda activate "${ENV_PREFIX}"
set -u
PY_BIN="$(command -v python)"
echo ">>> Python: ${PY_BIN}"
python -V

echo ">>> Upgrading pip tooling"
python -m pip install -U pip setuptools wheel

echo ">>> Installing Torch (CUDA 12.8 wheels)"
python -m pip install \
  "torch==2.7.0" \
  "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128

if [[ "${SKIP_ISAACSIM}" -eq 0 ]]; then
  echo ">>> Installing Isaac Sim 5.1.0 python packages"
  python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
else
  echo ">>> Skipping Isaac Sim pip install (--skip-isaacsim)"
fi

if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
  echo ">>> Shallow-cloning IsaacLab v2.3.2 into ${ISAACLAB_DIR}"
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

cd "${ISAACLAB_DIR}"
echo ">>> Checking out IsaacLab v2.3.2"
git fetch --depth 1 origin tag v2.3.2 || true
git checkout v2.3.2
git submodule update --init --recursive

echo ">>> Installing Isaac Lab python package + SKRL extension"
if ./isaaclab.sh --help | grep -q -- "--install"; then
  ./isaaclab.sh --install
  ./isaaclab.sh --install skrl
else
  ./isaaclab.sh -i skrl
fi

echo ">>> Runtime import check"
./isaaclab.sh -p - <<'PY'
import sys
import torch
import isaacsim
import isaaclab
import isaaclab_rl
import skrl
print("[check] python:", sys.version.split()[0])
print("[check] torch:", torch.__version__)
print("[check] isaacsim:", getattr(isaacsim, "__version__", "unknown"))
print("[check] isaaclab:", getattr(isaaclab, "__version__", "unknown"))
print("[check] isaaclab_rl module OK")
print("[check] skrl:", getattr(skrl, "__version__", "unknown"))
PY

cd "${PROJECT_DIR}"
echo ">>> Project dir: ${PROJECT_DIR}"

export ISAACLAB_PATH="${ISAACLAB_DIR}"
export CONDA_DEFAULT_ENV="${ENV_PREFIX}"
echo ">>> Checking official SKRL train.py for known --algorithm bug"
python isaac_lab_port/check_skrl_train_script.py

echo ">>> Checking task registration"
python isaac_lab_port/train_skrl_mappo.py --print-only

if [[ "${SKIP_SMOKE}" -eq 0 ]]; then
  echo ">>> Running strict smoke test (64 env, 500 steps)"
  python isaac_lab_port/train_skrl_mappo.py \
    --headless \
    --num-envs 64 \
    --smoke-test \
    --smoke-steps 500 \
    --smoke-log-every 50 \
    --smoke-require-contact
else
  echo ">>> Skipping smoke test (--skip-smoke)"
fi

echo ">>> Setup completed successfully at $(date)"
echo ">>> Activate later with: conda activate ${ENV_PREFIX}"
