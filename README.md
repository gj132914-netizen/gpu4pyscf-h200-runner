# gpu4pyscf-h200-runner

H200-oriented PySCF/GPU4PySCF runner for shared Linux CUDA servers. It wraps a
single-molecule XYZ workflow with conservative H200 defaults, early CUDA/CuPy
environment setup, CuPy memory cleanup, OOM-aware retry decisions, nvidia-smi
diagnostics, Linux resource diagnostics, and JSON summaries.

This package is meant for real computational chemistry jobs, but it does not
hide scientific changes. Any retry that changes grid, basis, functional,
density fitting, or backend is printed and recorded in the summary JSON.

## Server Profile

Default profile:

- backend: `gpu`
- gpu_method: `to_gpu`
- fallback_cpu: `false`
- memory_mb: `80000`
- threads: `12`
- gpu_memory_limit: `92%`
- grid_level: `2`
- max_cycle: `200`
- conv_tol: `1e-8`
- density_fit: `true`
- opt, gradient, hessian: `false`
- charge: `0`
- spin: `0`
- xc: `b3lyp`
- basis: `def2-svp`

The intended server is one dedicated NVIDIA H200 with 128 GB RAM, while CPU
resources are shared. CPU fallback is therefore explicit only.

This package is designed for user-space execution on shared Linux CUDA servers.
It must not perform system-level package installation, service restarts,
reboots, or process termination of other users' jobs.

## Install

Linux venv install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[cuda12]"
```

Conda environments are also fine. Activate the intended environment first, then
run the same `pip install` command inside that environment.

Editable local development from Git:

```bash
git clone git@github.com:MY_ACCOUNT/gpu4pyscf-h200-runner.git
cd gpu4pyscf-h200-runner
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[cuda12]"
```

Install from GitHub:

```bash
pip install "git+ssh://git@github.com/MY_ACCOUNT/gpu4pyscf-h200-runner.git@main"
```

Version-tagged install:

```bash
pip install "git+ssh://git@github.com/MY_ACCOUNT/gpu4pyscf-h200-runner.git@v0.1.0"
```

CPU-only install:

```bash
pip install -r requirements-cpu.txt
pip install -e .
```

CUDA extras:

```bash
pip install -e ".[cuda12]"
```

or use `.[cuda11]` / `.[cuda13]` for matching CUDA environments.

Linux helper scripts are provided for convenience. They assume the correct
venv/conda environment is already active and never use `sudo`:

```bash
bash install_gpu4pyscf_cuda12.sh
bash check_env.sh
bash run_h200_example.sh
```

## Environment Check

```bash
g4pyscf-check
g4pyscf-check --gpu-memory-limit 92% --threads 12
```

The check prints Python, package, CuPy CUDA, and nvidia-smi information. Use
`--json` for machine-readable diagnostics. On Linux, it also records `/proc`
and cgroup resource hints such as `/proc/meminfo`, `os.cpu_count()`,
`/sys/fs/cgroup/memory.max`, and the legacy cgroup memory limit path when
available.

Recommended Linux environment exports:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUPY_GPU_MEMORY_LIMIT="92%"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
```

If your scheduler or server has already set `CUDA_VISIBLE_DEVICES`, keep it.
The runner respects an existing value unless you explicitly pass `--gpu-id`.

## Basic CPU Example

```bash
g4pyscf-run \
  --xyz examples/example_water.xyz \
  --backend cpu \
  --basis def2-svp \
  --xc b3lyp \
  --memory-mb 80000 \
  --threads 12
```

## Recommended H200 Command

```bash
g4pyscf-run \
  --xyz examples/example_water.xyz \
  --gpu-memory-limit 92% \
  --backend gpu \
  --gpu-method to_gpu \
  --basis def2-svp \
  --xc b3lyp \
  --df \
  --grid-level 2 \
  --memory-mb 80000 \
  --threads 12
```

You can also use the default config:

```bash
g4pyscf-run --config examples/h200_default_config.json
```

When `--gpu-id 1` is passed, the runner sets `CUDA_VISIBLE_DEVICES=1` before
importing CuPy or GPU4PySCF. Inside Python that selected GPU may appear as
device index 0. That is normal CUDA behavior. If `--gpu-id` is not passed, the
runner does not override an existing `CUDA_VISIBLE_DEVICES`.

## Large Molecule Starter

```bash
g4pyscf-run \
  --xyz molecule.xyz \
  --gpu-memory-limit 92% \
  --backend gpu \
  --gpu-method to_gpu \
  --basis def2-svp \
  --xc b3lyp \
  --df \
  --grid-level 2 \
  --memory-mb 80000 \
  --threads 12 \
  --max-cycle 200 \
  --conv-tol 1e-8
```

Upgrade scientific settings gradually:

1. `b3lyp / def2-svp / grid 2`
2. `b3lyp / def2-tzvp / grid 2`
3. `cam-b3lyp / def2-tzvp / grid 2`
4. `grid 3` only after stable
5. gradient only when needed
6. hessian only with `--force-hessian` and a small enough molecule

## Geometry Optimization

Geometry optimization defaults to CPU for compatibility, followed by a final
single-point using the requested backend.

```bash
g4pyscf-run \
  --xyz molecule.xyz \
  --opt \
  --opt-backend cpu \
  --backend gpu \
  --gpu-method to_gpu \
  --xc b3lyp \
  --basis def2-svp \
  --df \
  --memory-mb 80000 \
  --threads 12 \
  --gpu-memory-limit 92%
```

If geomeTRIC is missing, the runner fails with:

```text
Geometry optimization requires geomeTRIC. Install with: pip install geometric
```

## OOM Retry Behavior

On OOM-like errors, the runner retries in this order:

1. Clean CuPy memory pools and retry the same settings once.
2. Lower `grid_level` by one while it is greater than 1.
3. Fall back large basis sets such as `def2-tzvp` to `def2-svp`.
4. Fall back range-separated or heavier hybrid functionals such as
   `cam-b3lyp`, `wb97x-d`, or `pbe0` to `b3lyp`.
5. Disable density fitting.
6. Retry CPU only when `--fallback-cpu` is explicitly enabled.

Fallbacks are recorded under `retry_decisions` in
`pyscf_h200_output/<job>_summary.json`, for example:

```json
[
  {
    "stage": "scf",
    "reason": "OOM detected",
    "action": "cleanup_gpu_memory_and_retry_same_settings",
    "success": false
  },
  {
    "stage": "scf",
    "reason": "OOM detected",
    "action": "lower_grid_level",
    "from": 2,
    "to": 1,
    "success": true
  }
]
```

The summary also includes `requested_config`, `resolved_config`,
`final_config`, `nvidia_smi_before`, `nvidia_smi_after`,
`nvidia_smi_on_failure`, CuPy memory snapshots, package versions,
`system_resources`, final backend, final GPU method, elapsed time, output paths,
and error text if failed.

GPU process diagnostics are read-only. The runner may report the nvidia-smi
process table, but it never terminates GPU processes automatically.

## Hessian Safety

Hessian is off by default. If `--hessian` is requested and the molecule has more
atoms than `--max-hessian-atoms` (default 60), Hessian is skipped unless
`--force-hessian` is passed. The skip reason is written to JSON.

Do not enable Hessian casually on large molecules. Frequency jobs can consume
much more GPU memory than a single-point calculation.

## Spin

`spin` is `2S = Nalpha - Nbeta`, not multiplicity.

- singlet: `spin 0`
- doublet: `spin 1`
- triplet: `spin 2`

## Outputs

Default output directory: `pyscf_h200_output`

Files:

- `<job>.log`
- `<job>.chk`
- `<job>_summary.json`
- `<job>_final.xyz`
- `<job>_optimized.xyz` when `--opt` is used
- `<job>_gradient.npy` when `--gradient` is used
- `<job>_hessian.npy` when `--hessian` is used

Existing outputs are not overwritten silently. If files already exist and
`--overwrite` is not passed, a timestamp is appended to the job name.

## Safety Notes

- Do not run multiple H200 jobs on the same GPU unless intentionally sharing
  VRAM.
- Set `CUDA_VISIBLE_DEVICES` externally or pass `--gpu-id`.
- Keep CPU threads reasonable on shared CPU nodes. The default is 12, and the
  runner warns when `--threads` is greater than 24.
- Do not use `sudo`, `apt`, `yum`, `dnf`, `systemctl`, reboot, shutdown, or
  service restart commands for this package.
- Do not modify system CUDA, system Python, or global site-packages. Use the
  active venv/conda environment.
- Never kill other users' processes. This package does not implement automatic
  process termination.
- Runtime code does not make network calls.
- Do not commit private molecules, logs, checkpoints, `.env` files, keys, or
  large outputs.
