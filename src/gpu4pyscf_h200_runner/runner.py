"""PySCF/GPU4PySCF execution engine."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RunnerConfig, config_to_dict, setup_environment, validate_run_config
from .diagnostics import (
    cuda_snapshot,
    nvidia_smi_snapshot,
    package_versions_snapshot,
    python_environment_snapshot,
    system_resource_snapshot,
)
from .exceptions import is_oom_error
from .io_utils import atoms_to_pyscf_atom_text, read_xyz, write_mol_xyz
from .memory import cleanup_gpu_memory, get_cupy_memory_info
from .retry import RetryPlanner, retry_message


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class OutputPaths:
    job_name: str
    outdir: Path
    log_file: Path
    chkfile: Path
    summary_json: Path
    final_xyz: Path
    optimized_xyz: Path
    gradient_npy: Path
    hessian_npy: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "outdir": str(self.outdir),
            "log_file": str(self.log_file),
            "chkfile": str(self.chkfile),
            "summary_json": str(self.summary_json),
            "final_xyz": str(self.final_xyz),
            "optimized_xyz": str(self.optimized_xyz),
            "gradient_npy": str(self.gradient_npy),
            "hessian_npy": str(self.hessian_npy),
        }


def make_output_paths(config: RunnerConfig) -> OutputPaths:
    if config.xyz is None:
        raise ValueError("--xyz is required unless it is provided in --config.")
    outdir = config.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    base_job_name = config.job_name or config.xyz.stem

    def build(job_name: str) -> OutputPaths:
        return OutputPaths(
            job_name=job_name,
            outdir=outdir,
            log_file=outdir / f"{job_name}.log",
            chkfile=outdir / f"{job_name}.chk",
            summary_json=outdir / f"{job_name}_summary.json",
            final_xyz=outdir / f"{job_name}_final.xyz",
            optimized_xyz=outdir / f"{job_name}_optimized.xyz",
            gradient_npy=outdir / f"{job_name}_gradient.npy",
            hessian_npy=outdir / f"{job_name}_hessian.npy",
        )

    paths = build(base_job_name)
    if not config.overwrite:
        candidates = [
            paths.log_file,
            paths.chkfile,
            paths.summary_json,
            paths.final_xyz,
            paths.optimized_xyz,
            paths.gradient_npy,
            paths.hessian_npy,
        ]
        if any(path.exists() for path in candidates):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths = build(f"{base_job_name}_{stamp}")
    return paths


def _verbose_level(config: RunnerConfig) -> int:
    if config.quiet:
        return 0
    if config.verbose:
        return 5
    return 4


def _print(config: RunnerConfig, message: str) -> None:
    if not config.quiet:
        print(message)


def _build_mol_from_text(atom_text: str, config: RunnerConfig, paths: OutputPaths):
    from pyscf import gto, lib

    try:
        lib.num_threads(config.threads)
    except Exception as exc:
        _print(config, f"[WARN] Failed to set PySCF thread count: {exc}")

    return gto.M(
        atom=atom_text,
        unit="Angstrom",
        basis=config.basis,
        charge=config.charge,
        spin=config.spin,
        max_memory=config.memory_mb,
        verbose=_verbose_level(config),
        output=str(paths.log_file),
    )


def build_mol_from_xyz(config: RunnerConfig, paths: OutputPaths):
    if config.xyz is None:
        raise ValueError("--xyz is required.")
    atom_text = atoms_to_pyscf_atom_text(read_xyz(config.xyz))
    return _build_mol_from_text(atom_text, config, paths)


def clone_mol_with_config(mol: Any, config: RunnerConfig, paths: OutputPaths):
    coords = mol.atom_coords(unit="Angstrom")
    lines = []
    for idx in range(mol.natm):
        symbol = mol.atom_symbol(idx)
        x, y, z = coords[idx]
        lines.append(f"{symbol:2s} {float(x): .12f} {float(y): .12f} {float(z): .12f}")
    return _build_mol_from_text("\n".join(lines), config, paths)


def apply_common_settings(mf: Any, config: RunnerConfig, paths: OutputPaths) -> Any:
    mf.max_cycle = config.max_cycle
    mf.conv_tol = config.conv_tol
    mf.chkfile = str(paths.chkfile)

    if hasattr(mf, "grids") and mf.grids is not None:
        mf.grids.level = config.grid_level
    if config.level_shift is not None and hasattr(mf, "level_shift"):
        mf.level_shift = config.level_shift
    if config.damping is not None:
        if hasattr(mf, "damp"):
            mf.damp = config.damping
        elif hasattr(mf, "damping"):
            mf.damping = config.damping
    if config.diis_space is not None and hasattr(mf, "diis_space"):
        mf.diis_space = config.diis_space
    if config.newton:
        mf = mf.newton()
    return mf


def maybe_density_fit(mf: Any, config: RunnerConfig) -> Any:
    if not config.density_fit:
        return mf
    if config.auxbasis:
        try:
            return mf.density_fit(auxbasis=config.auxbasis)
        except TypeError:
            _print(config, "[WARN] density_fit(auxbasis=...) failed; using default density_fit().")
            return mf.density_fit()
    return mf.density_fit()


def make_cpu_mf(mol: Any, config: RunnerConfig, paths: OutputPaths) -> Any:
    from pyscf import dft

    mf = dft.RKS(mol) if config.spin == 0 else dft.UKS(mol)
    mf.xc = config.xc
    mf = maybe_density_fit(mf, config)
    return apply_common_settings(mf, config, paths)


def make_gpu_mf_via_to_gpu(mol: Any, config: RunnerConfig, paths: OutputPaths) -> Any:
    mf = make_cpu_mf(mol, config, paths)
    cleanup_gpu_memory(verbose=not config.quiet)
    if not hasattr(mf, "to_gpu"):
        raise RuntimeError("This PySCF object has no to_gpu() method. Check GPU4PySCF installation.")
    return mf.to_gpu()


def make_gpu_mf_direct(mol: Any, config: RunnerConfig, paths: OutputPaths) -> Any:
    cleanup_gpu_memory(verbose=not config.quiet)
    if config.spin == 0:
        from gpu4pyscf.dft import rks

        mf = rks.RKS(mol, xc=config.xc)
    else:
        from gpu4pyscf.dft import uks

        mf = uks.UKS(mol, xc=config.xc)
    mf = maybe_density_fit(mf, config)
    return apply_common_settings(mf, config, paths)


def make_gpu_mf(mol: Any, config: RunnerConfig, paths: OutputPaths) -> tuple[Any, str]:
    errors: list[str] = []
    if config.gpu_method in ("auto", "to_gpu"):
        try:
            return make_gpu_mf_via_to_gpu(mol, config, paths), "to_gpu"
        except Exception as exc:
            errors.append(f"to_gpu failed: {exc}")
            if config.gpu_method == "to_gpu":
                raise
    if config.gpu_method in ("auto", "direct"):
        try:
            return make_gpu_mf_direct(mol, config, paths), "direct"
        except Exception as exc:
            errors.append(f"direct failed: {exc}")
            if config.gpu_method == "direct":
                raise
    raise RuntimeError("GPU mean-field creation failed.\n" + "\n".join(errors))


def make_mf(mol: Any, config: RunnerConfig, paths: OutputPaths) -> tuple[Any, str, str | None]:
    if config.backend == "cpu":
        return make_cpu_mf(mol, config, paths), "cpu", None
    if config.backend == "gpu":
        mf, method = make_gpu_mf(mol, config, paths)
        return mf, "gpu", method
    if config.backend == "auto":
        try:
            mf, method = make_gpu_mf(mol, config, paths)
            return mf, "gpu", method
        except Exception as exc:
            if config.fallback_cpu:
                _print(config, f"[WARN] GPU setup failed and CPU fallback is enabled: {exc}")
                return make_cpu_mf(mol, config, paths), "cpu", None
            raise
    raise ValueError(f"Unknown backend: {config.backend}")


def to_numpy_maybe(value: Any) -> Any:
    if hasattr(value, "get"):
        return value.get()
    return value


def get_homo_lumo(mf: Any) -> tuple[float | None, float | None, float | None]:
    import numpy as np

    try:
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        if isinstance(mo_energy, (list, tuple)):
            energies = np.concatenate([np.asarray(to_numpy_maybe(e)).reshape(-1) for e in mo_energy])
        else:
            energies = np.asarray(to_numpy_maybe(mo_energy)).reshape(-1)
        if isinstance(mo_occ, (list, tuple)):
            occs = np.concatenate([np.asarray(to_numpy_maybe(o)).reshape(-1) for o in mo_occ])
        else:
            occs = np.asarray(to_numpy_maybe(mo_occ)).reshape(-1)
        occupied = energies[occs > 1e-8]
        virtual = energies[occs <= 1e-8]
        homo = float(occupied.max()) if occupied.size else None
        lumo = float(virtual.min()) if virtual.size else None
        gap = float(lumo - homo) if homo is not None and lumo is not None else None
        return homo, lumo, gap
    except Exception as exc:
        print(f"[WARN] HOMO/LUMO extraction failed: {exc}")
        return None, None, None


def run_single_scf(mol: Any, config: RunnerConfig, paths: OutputPaths) -> dict[str, Any]:
    attempt_mol = clone_mol_with_config(mol, config, paths)
    try:
        if config.backend != "cpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        mf, backend_used, gpu_method_used = make_mf(attempt_mol, config, paths)
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        _print(config, f"[INFO] Starting SCF. backend={backend_used}, gpu_method={gpu_method_used}")
        energy = mf.kernel()
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        energy = to_numpy_maybe(energy)
        energy_float = float(energy) if energy is not None else None
        return {
            "mol": attempt_mol,
            "mf": mf,
            "backend_used": backend_used,
            "gpu_method_used": gpu_method_used,
            "energy": energy_float,
        }
    except Exception:
        cleanup_gpu_memory(verbose=not config.quiet)
        raise


def run_scf_with_retry(mol: Any, config: RunnerConfig, paths: OutputPaths, summary: dict[str, Any]) -> dict[str, Any]:
    planner = RetryPlanner()
    current_config = config
    pending_decision: dict[str, Any] | None = None
    summary.setdefault("retry_decisions", [])
    summary.setdefault("nvidia_smi_on_failure", [])

    while True:
        try:
            result = run_single_scf(mol, current_config, paths)
            if pending_decision is not None:
                pending_decision["success"] = True
            result["final_config"] = current_config
            return result
        except Exception as exc:
            if pending_decision is not None:
                pending_decision["success"] = False
                pending_decision["error"] = f"{type(exc).__name__}: {exc}"

            if not is_oom_error(exc):
                raise

            _print(current_config, "[WARN] OOM detected")
            summary.setdefault("oom_errors", []).append(
                {
                    "datetime": datetime.now().isoformat(timespec="seconds"),
                    "config": config_to_dict(current_config),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            summary["nvidia_smi_on_failure"].append(nvidia_smi_snapshot())
            cleanup_gpu_memory(verbose=not current_config.quiet)

            next_config, decision = planner.next_retry(current_config, exc)
            if next_config is None or decision is None:
                summary["final_config"] = config_to_dict(current_config)
                raise

            _print(current_config, f"[WARN] {retry_message(decision)}")
            summary["retry_decisions"].append(decision)
            pending_decision = decision
            current_config = next_config


def optimize_geometry_if_requested(mol: Any, config: RunnerConfig, paths: OutputPaths) -> Any:
    if not config.opt:
        return mol
    try:
        from pyscf.geomopt.geometric_solver import optimize
    except Exception as exc:
        raise RuntimeError("Geometry optimization requires geomeTRIC. Install with: pip install geometric") from exc

    if config.opt_backend == "cpu":
        opt_backend = "cpu"
    elif config.opt_backend == "gpu":
        opt_backend = "gpu"
    elif config.opt_backend == "same":
        opt_backend = config.backend
    else:
        raise ValueError(f"Unknown opt_backend: {config.opt_backend}")

    opt_config = replace(config, backend=opt_backend)
    _print(config, f"[INFO] Starting geometry optimization with backend={opt_backend}")
    try:
        opt_mol = clone_mol_with_config(mol, opt_config, paths)
        mf, backend_used, _ = make_mf(opt_mol, opt_config, paths)
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        optimized = optimize(mf, maxsteps=config.opt_maxsteps)
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        write_mol_xyz(optimized, paths.optimized_xyz, comment=f"Optimized with {config.xc}/{config.basis}")
        _print(config, f"[INFO] Optimized XYZ written: {paths.optimized_xyz}")
        return optimized
    except Exception:
        cleanup_gpu_memory(verbose=not config.quiet)
        raise


def compute_gradient_if_requested(mf: Any, config: RunnerConfig, paths: OutputPaths, backend_used: str) -> dict[str, Any]:
    if not config.gradient:
        return {"gradient_norm": None, "gradient_npy": None}
    import numpy as np

    try:
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        grad_obj = mf.nuc_grad_method() if hasattr(mf, "nuc_grad_method") else mf.Gradients()
        gradient = to_numpy_maybe(grad_obj.kernel())
        gradient = np.asarray(gradient, dtype=float)
        np.save(paths.gradient_npy, gradient)
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        return {"gradient_norm": float(np.linalg.norm(gradient)), "gradient_npy": str(paths.gradient_npy)}
    except Exception:
        cleanup_gpu_memory(verbose=not config.quiet)
        raise


def compute_hessian_if_requested(mf: Any, config: RunnerConfig, paths: OutputPaths, backend_used: str) -> dict[str, Any]:
    if not config.hessian:
        return {"hessian_shape": None, "hessian_npy": None}
    import numpy as np

    try:
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        hessian = to_numpy_maybe(mf.Hessian().kernel())
        hessian = np.asarray(hessian, dtype=float)
        np.save(paths.hessian_npy, hessian)
        if backend_used == "gpu":
            cleanup_gpu_memory(verbose=not config.quiet)
        return {"hessian_shape": list(hessian.shape), "hessian_npy": str(paths.hessian_npy)}
    except Exception:
        cleanup_gpu_memory(verbose=not config.quiet)
        raise


def compute_dipole_if_requested(mf: Any, config: RunnerConfig) -> list[float] | None:
    if not config.dipole:
        return None
    try:
        dipole = to_numpy_maybe(mf.dip_moment(unit="Debye"))
        return [float(x) for x in dipole]
    except Exception as exc:
        _print(config, f"[WARN] Dipole calculation failed: {exc}")
        return None


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _base_summary(config: RunnerConfig, paths: OutputPaths, start_wall: str) -> dict[str, Any]:
    summary = {
        "success": False,
        "job_name": paths.job_name,
        "start_time": start_wall,
        "requested_config": config_to_dict(config),
        "resolved_config": config_to_dict(config),
        "final_config": None,
        "command_line_args": list(config.command_line_args),
        "output_paths": paths.as_dict(),
        "environment_report_enabled": bool(config.write_env_report),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_gpu_memory_limit": os.environ.get("CUPY_GPU_MEMORY_LIMIT"),
        "retry_decisions": [],
        "warnings": [],
    }

    if config.write_env_report:
        summary.update({
            "python_environment": python_environment_snapshot(),
            "package_versions": package_versions_snapshot(),
            "cuda_snapshot": cuda_snapshot(),
            "system_resources": system_resource_snapshot(),
        })

    return summary

def run_job(config: RunnerConfig) -> dict[str, Any]:
    validate_run_config(config)
    setup_environment(config)
    paths = make_output_paths(config)
    config = replace(config, job_name=paths.job_name, outdir=paths.outdir)

    start = time.time()
    start_wall = datetime.now().isoformat(timespec="seconds")
    summary = _base_summary(config, paths, start_wall)

    if config.lowmem:
        _print(config, "[INFO] Low-memory mode enabled")
    if config.threads > 24:
        warning = (
            f"Requested threads={config.threads}, which is high for a shared CPU server. "
            "Default is 12; consider keeping --threads <= 24 unless the scheduler granted more CPUs."
        )
        summary["warnings"].append(warning)
        _print(config, f"[WARN] {warning}")

    try:
        summary["nvidia_smi_before"] = nvidia_smi_snapshot()
        summary["cupy_memory_before"] = get_cupy_memory_info()

        mol = build_mol_from_xyz(config, paths)
        mol = optimize_geometry_if_requested(mol, config, paths)

        runtime_config = config
        if config.hessian and mol.natm > config.max_hessian_atoms and not config.force_hessian:
            reason = "natom exceeds max_hessian_atoms; pass --force-hessian to override"
            _print(config, f"[WARN] Hessian skipped: {reason}")
            summary["hessian_skipped_reason"] = reason
            runtime_config = replace(config, hessian=False)

        scf_result = run_scf_with_retry(mol, runtime_config, paths, summary)
        final_config: RunnerConfig = scf_result["final_config"]
        mf = scf_result["mf"]
        final_mol = scf_result["mol"]
        backend_used = scf_result["backend_used"]
        gpu_method_used = scf_result["gpu_method_used"]
        energy = scf_result["energy"]

        homo, lumo, gap = get_homo_lumo(mf)
        dipole = compute_dipole_if_requested(mf, final_config)
        gradient = compute_gradient_if_requested(mf, final_config, paths, backend_used)
        hessian = compute_hessian_if_requested(mf, final_config, paths, backend_used)

        write_mol_xyz(final_mol, paths.final_xyz, comment=f"Final geometry for {paths.job_name}")

        elapsed = time.time() - start
        summary.update(
            {
                "success": True,
                "end_time": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "final_config": config_to_dict(final_config),
                "final_backend_used": backend_used,
                "final_gpu_method_used": gpu_method_used,
                "converged": bool(getattr(mf, "converged", False)),
                "total_energy_hartree": energy,
                "homo_hartree": homo,
                "lumo_hartree": lumo,
                "gap_hartree": gap,
                "gap_ev": gap * HARTREE_TO_EV if gap is not None else None,
                "dipole_debye_xyz": dipole,
                "gradient_norm": gradient["gradient_norm"],
                "gradient_npy": gradient["gradient_npy"],
                "hessian_shape": hessian["hessian_shape"],
                "hessian_npy": hessian["hessian_npy"],
                "nvidia_smi_after": nvidia_smi_snapshot(),
                "cupy_memory_after": get_cupy_memory_info(),
                "error": None,
            }
        )
        _write_summary(paths.summary_json, summary)
        _print_result(config, paths, summary)
        return summary
    except Exception as exc:
        if is_oom_error(exc):
            cleanup_gpu_memory(verbose=not config.quiet)
        summary.update(
            {
                "success": False,
                "end_time": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.time() - start,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "nvidia_smi_after": nvidia_smi_snapshot(),
                "cupy_memory_after": get_cupy_memory_info(),
            }
        )
        summary.setdefault("nvidia_smi_on_failure", []).append(nvidia_smi_snapshot())
        _write_summary(paths.summary_json, summary)
        _print(config, "\n[ERROR] Job failed.")
        _print(config, str(exc))
        _print(config, f"[INFO] Failure summary written: {paths.summary_json}")
        return summary


def _print_result(config: RunnerConfig, paths: OutputPaths, summary: dict[str, Any]) -> None:
    if config.quiet:
        return
    print("\n========== PySCF / GPU4PySCF RESULT ==========")
    print(f"Job name       : {summary['job_name']}")
    print(f"Backend used   : {summary['final_backend_used']}")
    print(f"GPU method     : {summary['final_gpu_method_used']}")
    print(f"Converged      : {summary['converged']}")
    if summary.get("total_energy_hartree") is not None:
        print(f"Energy         : {summary['total_energy_hartree']:.12f} Hartree")
    if summary.get("gap_ev") is not None:
        print(f"HOMO           : {summary['homo_hartree']:.8f} Hartree")
        print(f"LUMO           : {summary['lumo_hartree']:.8f} Hartree")
        print(f"Gap            : {summary['gap_ev']:.6f} eV")
    if summary.get("gradient_norm") is not None:
        print(f"Gradient norm  : {summary['gradient_norm']:.8e}")
    if summary.get("hessian_shape") is not None:
        print(f"Hessian shape  : {summary['hessian_shape']}")
    if summary.get("dipole_debye_xyz") is not None:
        print(f"Dipole         : {summary['dipole_debye_xyz']} Debye")
    print(f"Log            : {paths.log_file}")
    print(f"Summary JSON   : {paths.summary_json}")
    print(f"Final XYZ      : {paths.final_xyz}")
    print("==============================================\n")
