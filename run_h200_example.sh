#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Python path: $(which python)"
python --version

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "[INFO] CUDA_VISIBLE_DEVICES was unset; defaulting to 0 for this shell."
else
  echo "[INFO] Respecting existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

export CUPY_GPU_MEMORY_LIMIT="${CUPY_GPU_MEMORY_LIMIT:-92%}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-12}"

echo "[INFO] CUPY_GPU_MEMORY_LIMIT=${CUPY_GPU_MEMORY_LIMIT}"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[INFO] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[INFO] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "[INFO] NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"

g4pyscf-check

g4pyscf-run \
  --xyz examples/example_water.xyz \
  --backend gpu \
  --gpu-method to_gpu \
  --basis def2-svp \
  --xc b3lyp \
  --df \
  --grid-level 2 \
  --gpu-memory-limit "${CUPY_GPU_MEMORY_LIMIT}" \
  --memory-mb 80000 \
  --threads "${OMP_NUM_THREADS}"

