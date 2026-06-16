"""Environment and GPU diagnostics for the H200 runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    if not shutil.which(cmd[0]):
        return {"command": cmd, "available": False, "error": f"{cmd[0]} not found in PATH"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": cmd,
            "available": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"command": cmd, "available": True, "error": str(exc)}


def nvidia_smi_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    return {
        "nvidia_smi": _run_command(["nvidia-smi"]),
        "nvidia_smi_query": _run_command(query),
        "nvidia_smi_compute_processes": _run_command(process_query),
    }


def python_environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_gpu_memory_limit": os.environ.get("CUPY_GPU_MEMORY_LIMIT"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def package_versions_snapshot() -> dict[str, Any]:
    packages = [
        "pyscf",
        "gpu4pyscf",
        "gpu4pyscf-cuda11x",
        "gpu4pyscf-cuda12x",
        "gpu4pyscf-cuda13x",
        "cupy",
        "cupy-cuda11x",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "numpy",
        "geometric",
        "cutensor-cu11",
        "cutensor-cu12",
        "cutensor-cu13",
    ]
    versions: dict[str, Any] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        except Exception as exc:
            versions[package] = f"error: {exc}"
    return versions


def cuda_snapshot() -> dict[str, Any]:
    """Return CuPy CUDA details. Imports CuPy lazily."""

    try:
        import cupy as cp

        devices = []
        count = int(cp.cuda.runtime.getDeviceCount())
        for idx in range(count):
            props = cp.cuda.runtime.getDeviceProperties(idx)
            name = props.get("name", b"unknown")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            devices.append(
                {
                    "index": idx,
                    "name": name,
                    "total_global_mem": int(props.get("totalGlobalMem", 0)),
                    "multi_processor_count": int(props.get("multiProcessorCount", 0)),
                }
            )
        try:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            memory = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}
        except Exception as exc:
            memory = {"error": str(exc)}
        return {"cupy_importable": True, "device_count": count, "devices": devices, "memory": memory}
    except Exception as exc:
        return {"cupy_importable": False, "error": str(exc)}


def _read_text_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    try:
        return {
            "path": path,
            "available": file_path.exists(),
            "text": file_path.read_text(encoding="utf-8", errors="replace") if file_path.exists() else None,
        }
    except Exception as exc:
        return {"path": path, "available": file_path.exists(), "error": str(exc)}


def _parse_meminfo(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            key = parts[0].rstrip(":")
            try:
                values[key] = int(parts[1])
            except ValueError:
                pass
    return values


def _summarize_cpuinfo(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    processor_count = 0
    model_names: list[str] = []
    for line in text.splitlines():
        if line.startswith("processor"):
            processor_count += 1
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            name = value.strip()
            if name and name not in model_names:
                model_names.append(name)
    return {"processor_entries": processor_count, "model_names": model_names[:4]}


def cgroup_resource_snapshot() -> dict[str, Any]:
    """Read Linux cgroup resource hints without changing system state."""

    memory_max = _read_text_file("/sys/fs/cgroup/memory.max")
    legacy_memory_limit = _read_text_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return {
        "memory_max": memory_max,
        "legacy_memory_limit_in_bytes": legacy_memory_limit,
    }


def system_resource_snapshot() -> dict[str, Any]:
    """Return Linux shared-server resource diagnostics when available."""

    meminfo = _read_text_file("/proc/meminfo")
    cpuinfo = _read_text_file("/proc/cpuinfo")
    return {
        "os_cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        "user_id": getattr(os, "getuid", lambda: None)(),
        "meminfo": meminfo,
        "meminfo_parsed_kb": _parse_meminfo(meminfo.get("text")),
        "cpuinfo": {
            "path": cpuinfo.get("path"),
            "available": cpuinfo.get("available"),
            "error": cpuinfo.get("error"),
            "summary": _summarize_cpuinfo(cpuinfo.get("text")),
        },
        "cgroup": cgroup_resource_snapshot(),
    }


def full_diagnostics() -> dict[str, Any]:
    return {
        "python_environment": python_environment_snapshot(),
        "package_versions": package_versions_snapshot(),
        "cuda": cuda_snapshot(),
        "nvidia_smi": nvidia_smi_snapshot(),
        "system_resources": system_resource_snapshot(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check PySCF/GPU4PySCF/CUDA environment.")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--gpu-memory-limit", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args(argv)

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.gpu_memory_limit:
        os.environ["CUPY_GPU_MEMORY_LIMIT"] = str(args.gpu_memory_limit)
    if args.threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(args.threads)

    data = full_diagnostics()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    print("========== GPU4PySCF H200 Environment Check ==========")
    env = data["python_environment"]
    print(f"Python      : {env['python'].splitlines()[0]}")
    print(f"Executable  : {env['executable']}")
    print(f"Platform    : {env['platform']}")
    print(f"CUDA_VISIBLE_DEVICES : {env['cuda_visible_devices']}")
    print(f"CUPY_GPU_MEMORY_LIMIT: {env['cupy_gpu_memory_limit']}")
    print("\n[Packages]")
    for key, value in data["package_versions"].items():
        status = value if value else "not installed"
        print(f"{key:16s}: {status}")
    print("\n[CuPy CUDA]")
    print(json.dumps(data["cuda"], indent=2))
    print("\n[nvidia-smi query]")
    query = data["nvidia_smi"]["nvidia_smi_query"]
    print(query.get("stdout") or query.get("stderr") or query.get("error") or "No output")
    print("\n[System resources]")
    resources = data["system_resources"]
    print(f"os.cpu_count(): {resources.get('os_cpu_count')}")
    cgroup = resources.get("cgroup", {})
    print(f"cgroup memory.max: {cgroup.get('memory_max', {}).get('text')}")
    print(f"legacy cgroup memory.limit_in_bytes: {cgroup.get('legacy_memory_limit_in_bytes', {}).get('text')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
