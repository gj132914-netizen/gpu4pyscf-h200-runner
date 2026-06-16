"""Exception helpers for GPU memory failures."""

from __future__ import annotations


OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cusolver_status_alloc_failed",
    "cupy.cuda.memory.outofmemoryerror",
    "outofmemoryerror",
    "failed to allocate",
    "memory allocation failed",
    "insufficient memory",
    "device memory",
    "allocator",
)


def is_oom_error(exc: BaseException) -> bool:
    """Return True when an exception looks like a GPU or allocator OOM."""

    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in OOM_PATTERNS)

