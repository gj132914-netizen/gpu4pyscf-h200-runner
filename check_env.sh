#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Python path: $(which python)"
python --version

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] CUPY_GPU_MEMORY_LIMIT=${CUPY_GPU_MEMORY_LIMIT:-<unset>}"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "[INFO] MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "[INFO] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"
echo "[INFO] NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-<unset>}"

g4pyscf-check "$@"

