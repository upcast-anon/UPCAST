#!/usr/bin/env bash
set -euo pipefail

env_prefix="${JEDI_ENV_PREFIX:-./conda_envs/jedi}"
conda_bin="${CONDA:-conda}"
python_version="${PYTHON_VERSION:-3.10}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-./conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-./pip_cache}"
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "$(dirname "${env_prefix}")"

if ! command -v "${conda_bin}" >/dev/null 2>&1; then
  echo "Cannot find conda: ${conda_bin}" >&2
  exit 1
fi

if [[ ! -d "${env_prefix}" ]]; then
  "${conda_bin}" create -y -p "${env_prefix}" "python=${python_version}" pip
fi

python_bin="${env_prefix}/bin/python"
pip_bin="${env_prefix}/bin/pip"

"${pip_bin}" install --upgrade pip setuptools wheel

# Keep these packages out of the training env. videojedi currently pulls vjepa
# and may constrain torch/torchvision, so this environment is evaluation-only.
"${pip_bin}" install --upgrade videojedi


"${python_bin}" - <<'PY'
import importlib.util

packages = ["videojedi"]
for name in packages:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'ok' if spec is not None else 'missing'}")
PY

echo "JEDi environment ready:"
echo "  ${env_prefix}"
echo "Use with:"
echo "  Configure the metric helpers to use ${python_bin}."
