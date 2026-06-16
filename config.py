"""Configuration loading, CLI parsing, and early environment setup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .retry import is_large_basis


DEFAULT_H200_PROFILE: dict[str, Any] = {
    "xyz": None,
    "job_name": None,
    "outdir": "pyscf_h200_output",
    "gpu_id": None,
    "gpu_memory_limit": "92%",
    "backend": "gpu",
    "gpu_method": "to_gpu",
    "fallback_cpu": False,
    "charge": 0,
    "spin": 0,
    "basis": "def2-svp",
    "xc": "b3lyp",
    "density_fit": True,
    "auxbasis": None,
    "grid_level": 2,
    "max_cycle": 200,
    "conv_tol": 1e-8,
    "memory_mb": 80000,
    "threads": 12,
    "newton": False,
    "level_shift": None,
    "damping": None,
    "diis_space": None,
    "opt": False,
    "opt_backend": "cpu",
    "opt_maxsteps": 100,
    "gradient": False,
    "hessian": False,
    "force_hessian": False,
    "max_hessian_atoms": 60,
    "dipole": False,
    "lowmem": False,
    "overwrite": False,
    "verbose": False,
    "quiet": False,
    "write_env_report": True,
}


@dataclass(frozen=True)
class RunnerConfig:
    xyz: Path | None = None
    job_name: str | None = None
    outdir: Path = Path("pyscf_h200_output")
    gpu_id: int | None = None
    gpu_memory_limit: str | None = "92%"
    backend: str = "gpu"
    gpu_method: str = "to_gpu"
    fallback_cpu: bool = False
    charge: int = 0
    spin: int = 0
    basis: str = "def2-svp"
    xc: str = "b3lyp"
    density_fit: bool = True
    auxbasis: str | None = None
    grid_level: int = 2
    max_cycle: int = 200
    conv_tol: float = 1e-8
    memory_mb: int = 80000
    threads: int = 12
    newton: bool = False
    level_shift: float | None = None
    damping: float | None = None
    diis_space: int | None = None
    opt: bool = False
    opt_backend: str = "cpu"
    opt_maxsteps: int = 100
    gradient: bool = False
    hessian: bool = False
    force_hessian: bool = False
    max_hessian_atoms: int = 60
    dipole: bool = False
    lowmem: bool = False
    overwrite: bool = False
    verbose: bool = False
    quiet: bool = False
    write_env_report: bool = True
    config: Path | None = None
    command_line_args: list[str] = field(default_factory=list)


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def config_to_dict(config: RunnerConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("xyz", "outdir", "config"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def _normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    data = dict(mapping)
    if "df" in data and "density_fit" not in data:
        data["density_fit"] = data.pop("df")
    if "gpu_memory_limit" in data and data["gpu_memory_limit"] is not None:
        data["gpu_memory_limit"] = str(data["gpu_memory_limit"])
    return data


def config_from_mapping(mapping: dict[str, Any], command_line_args: list[str] | None = None) -> RunnerConfig:
    data = _normalize_mapping(mapping)
    allowed = set(RunnerConfig.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
    if "xyz" in data:
        data["xyz"] = _path_or_none(data["xyz"])
    if "outdir" in data:
        data["outdir"] = Path(data["outdir"])
    if "config" in data:
        data["config"] = _path_or_none(data["config"])
    if command_line_args is not None:
        data["command_line_args"] = list(command_line_args)
    return RunnerConfig(**data)


def apply_lowmem_policy(config: RunnerConfig, explicit_args: set[str]) -> RunnerConfig:
    if not config.lowmem:
        return config

    updates: dict[str, Any] = {
        "grid_level": min(config.grid_level, 1),
        "max_cycle": max(config.max_cycle, 200),
        "conv_tol": max(config.conv_tol, 1e-8),
    }
    if "--gradient" not in explicit_args:
        updates["gradient"] = False
    if "--hessian" not in explicit_args and not config.force_hessian:
        updates["hessian"] = False
    if is_large_basis(config.basis):
        updates["basis"] = "def2-svp"
    return replace(config, **updates)


def _build_parser(defaults: dict[str, Any], parents: list[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="H200-optimized PySCF/GPU4PySCF runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
    )
    parser.add_argument("--xyz", type=Path, default=_path_or_none(defaults.get("xyz")))
    parser.add_argument("--job-name", default=defaults.get("job_name"))
    parser.add_argument("--outdir", type=Path, default=Path(defaults.get("outdir", "pyscf_h200_output")))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=bool(defaults.get("overwrite")))

    parser.add_argument("--gpu-id", type=int, default=defaults.get("gpu_id"))
    parser.add_argument("--gpu-memory-limit", default=defaults.get("gpu_memory_limit"))
    parser.add_argument("--threads", type=int, default=int(defaults.get("threads", 12)))
    parser.add_argument("--memory-mb", type=int, default=int(defaults.get("memory_mb", 80000)))

    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"], default=defaults.get("backend", "gpu"))
    parser.add_argument(
        "--gpu-method",
        choices=["to_gpu", "direct", "auto"],
        default=defaults.get("gpu_method", "to_gpu"),
    )
    parser.add_argument(
        "--fallback-cpu",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("fallback_cpu", False)),
    )

    parser.add_argument("--charge", type=int, default=int(defaults.get("charge", 0)))
    parser.add_argument("--spin", type=int, default=int(defaults.get("spin", 0)))
    parser.add_argument("--basis", default=defaults.get("basis", "def2-svp"))
    parser.add_argument("--xc", default=defaults.get("xc", "b3lyp"))

    df_group = parser.add_mutually_exclusive_group()
    df_group.add_argument("--df", dest="density_fit", action="store_true")
    df_group.add_argument("--no-df", dest="density_fit", action="store_false")
    parser.set_defaults(density_fit=bool(defaults.get("density_fit", True)))
    parser.add_argument("--auxbasis", default=defaults.get("auxbasis"))

    parser.add_argument("--grid-level", type=int, default=int(defaults.get("grid_level", 2)))
    parser.add_argument("--max-cycle", type=int, default=int(defaults.get("max_cycle", 200)))
    parser.add_argument("--conv-tol", type=float, default=float(defaults.get("conv_tol", 1e-8)))
    parser.add_argument("--newton", action=argparse.BooleanOptionalAction, default=bool(defaults.get("newton", False)))
    parser.add_argument("--level-shift", type=float, default=defaults.get("level_shift"))
    parser.add_argument("--damping", type=float, default=defaults.get("damping"))
    parser.add_argument("--diis-space", type=int, default=defaults.get("diis_space"))

    parser.add_argument("--opt", action=argparse.BooleanOptionalAction, default=bool(defaults.get("opt", False)))
    parser.add_argument(
        "--opt-backend",
        choices=["cpu", "gpu", "same"],
        default=defaults.get("opt_backend", "cpu"),
    )
    parser.add_argument("--opt-maxsteps", type=int, default=int(defaults.get("opt_maxsteps", 100)))

    parser.add_argument("--gradient", action=argparse.BooleanOptionalAction, default=bool(defaults.get("gradient", False)))
    parser.add_argument("--hessian", action=argparse.BooleanOptionalAction, default=bool(defaults.get("hessian", False)))
    parser.add_argument(
        "--force-hessian",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("force_hessian", False)),
    )
    parser.add_argument(
        "--max-hessian-atoms",
        type=int,
        default=int(defaults.get("max_hessian_atoms", 60)),
    )
    parser.add_argument("--dipole", action=argparse.BooleanOptionalAction, default=bool(defaults.get("dipole", False)))
    parser.add_argument("--lowmem", action=argparse.BooleanOptionalAction, default=bool(defaults.get("lowmem", False)))
    parser.add_argument("--verbose", action="store_true", default=bool(defaults.get("verbose", False)))
    parser.add_argument("--quiet", action="store_true", default=bool(defaults.get("quiet", False)))
    parser.add_argument("--write-env-report", action=argparse.BooleanOptionalAction, default=bool(defaults.get("write_env_report", True)))
    return parser


def parse_config(argv: list[str] | None = None) -> RunnerConfig:
    args_list = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(args_list)

    merged = dict(DEFAULT_H200_PROFILE)
    merged.update(_normalize_mapping(load_config_file(pre_args.config)))
    merged["config"] = pre_args.config

    parser = _build_parser(merged, parents=[pre])
    namespace = parser.parse_args(args_list)
    mapping = vars(namespace)
    mapping["command_line_args"] = args_list
    config = config_from_mapping(mapping)
    explicit_args = {arg for arg in args_list if arg.startswith("--")}
    return apply_lowmem_policy(config, explicit_args)


def validate_run_config(config: RunnerConfig) -> None:
    if config.xyz is None:
        raise ValueError("--xyz is required unless it is provided in --config.")
    if config.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if config.memory_mb < 1:
        raise ValueError("--memory-mb must be at least 1.")
    if config.grid_level < 0:
        raise ValueError("--grid-level must be non-negative.")
    if config.max_cycle < 1:
        raise ValueError("--max-cycle must be at least 1.")


def setup_environment(config: RunnerConfig) -> None:
    """Set environment variables before importing CuPy or GPU4PySCF."""

    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    if config.gpu_memory_limit:
        os.environ["CUPY_GPU_MEMORY_LIMIT"] = str(config.gpu_memory_limit)
    thread_count = str(config.threads)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count

