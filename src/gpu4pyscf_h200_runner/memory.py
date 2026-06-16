"""CuPy memory helpers.

This module intentionally imports CuPy only inside functions. The CLI applies
CUDA_VISIBLE_DEVICES and CUPY_GPU_MEMORY_LIMIT before these helpers are used.
"""

from __future__ import annotations

from typing import Any


def cleanup_gpu_memory(verbose: bool = True) -> None:
    """Synchronize the current CuPy device and release CuPy memory pools."""

    try:
        import cupy as cp

        try:
            cp.cuda.Device().synchronize()
        except Exception:
            pass
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        try:
            cp.cuda.Device().synchronize()
        except Exception:
            pass
        if verbose:
            print("[INFO] CuPy GPU memory pools cleaned.")
    except Exception as exc:
        if verbose:
            print(f"[WARN] GPU memory cleanup failed: {exc}")


def get_cupy_memory_info() -> dict[str, Any]:
    """Return CuPy runtime memory information when CuPy is importable."""

    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return {
            "available": True,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "free_mb": round(int(free_bytes) / 1024**2, 3),
            "total_mb": round(int(total_bytes) / 1024**2, 3),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}

