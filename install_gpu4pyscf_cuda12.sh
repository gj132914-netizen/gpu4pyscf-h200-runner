#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] This installer uses the active venv/conda environment only."
echo "[INFO] No sudo, apt, yum, dnf, service restart, or system CUDA changes are performed."
echo "[INFO] Python path: $(which python)"
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[cuda12]"

echo "[INFO] Running environment check..."
g4pyscf-check

