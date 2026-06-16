"""OOM retry planning for the H200 runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .exceptions import is_oom_error


LARGE_BASIS_MARKERS = (
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
    "def2-qzvpp",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc",
    "pcseg-2",
    "pcseg-3",
)

XC_FALLBACKS = {
    "cam-b3lyp": "b3lyp",
    "wb97x": "b3lyp",
    "wb97x-d": "b3lyp",
    "wB97X": "b3lyp",
    "wB97X-D": "b3lyp",
    "lc-wpbe": "b3lyp",
    "lcwpbe": "b3lyp",
    "pbe0": "b3lyp",
}


def is_large_basis(basis: str | None) -> bool:
    normalized = (basis or "").lower().replace("_", "-")
    return any(marker in normalized for marker in LARGE_BASIS_MARKERS)


def fallback_xc(xc: str | None) -> str | None:
    if not xc:
        return None
    normalized = xc.lower().replace("_", "-")
    return XC_FALLBACKS.get(normalized)


@dataclass
class RetryPlanner:
    cleanup_retry_done: bool = False
    basis_fallback_done: bool = False
    xc_fallback_done: bool = False
    df_disabled_done: bool = False
    cpu_fallback_done: bool = False

    def next_retry(self, config: Any, exc: BaseException, stage: str = "scf") -> tuple[Any | None, dict[str, Any] | None]:
        """Return the next config and decision record after an OOM exception."""

        if not is_oom_error(exc):
            return None, None

        reason = "OOM detected"

        if not self.cleanup_retry_done:
            self.cleanup_retry_done = True
            return replace(config), {
                "stage": stage,
                "reason": reason,
                "action": "cleanup_gpu_memory_and_retry_same_settings",
                "success": None,
            }

        if getattr(config, "grid_level", 0) > 1:
            old = int(config.grid_level)
            new_config = replace(config, grid_level=old - 1)
            return new_config, {
                "stage": stage,
                "reason": reason,
                "action": "lower_grid_level",
                "from": old,
                "to": old - 1,
                "success": None,
            }

        if not self.basis_fallback_done and is_large_basis(getattr(config, "basis", None)):
            self.basis_fallback_done = True
            old = config.basis
            return replace(config, basis="def2-svp"), {
                "stage": stage,
                "reason": reason,
                "action": "fallback_basis",
                "from": old,
                "to": "def2-svp",
                "success": None,
            }

        new_xc = fallback_xc(getattr(config, "xc", None))
        if not self.xc_fallback_done and new_xc and new_xc != getattr(config, "xc", None):
            self.xc_fallback_done = True
            old = config.xc
            return replace(config, xc=new_xc), {
                "stage": stage,
                "reason": reason,
                "action": "fallback_xc",
                "from": old,
                "to": new_xc,
                "success": None,
            }

        if not self.df_disabled_done and getattr(config, "density_fit", False):
            self.df_disabled_done = True
            return replace(config, density_fit=False), {
                "stage": stage,
                "reason": reason,
                "action": "disable_density_fitting",
                "from": True,
                "to": False,
                "success": None,
            }

        if (
            not self.cpu_fallback_done
            and getattr(config, "fallback_cpu", False)
            and getattr(config, "backend", None) != "cpu"
        ):
            self.cpu_fallback_done = True
            return replace(config, backend="cpu", gpu_method="auto"), {
                "stage": stage,
                "reason": reason,
                "action": "fallback_cpu",
                "from": getattr(config, "backend", None),
                "to": "cpu",
                "success": None,
            }

        return None, None


def retry_message(decision: dict[str, Any]) -> str:
    action = decision.get("action")
    if action == "cleanup_gpu_memory_and_retry_same_settings":
        return "Retrying after CuPy memory cleanup"
    if action == "lower_grid_level":
        return f"Lowering grid_level from {decision.get('from')} to {decision.get('to')}"
    if action == "fallback_basis":
        return f"Falling back basis {decision.get('from')} -> {decision.get('to')}"
    if action == "fallback_xc":
        return f"Changing functional {decision.get('from')} -> {decision.get('to')}"
    if action == "disable_density_fitting":
        return "Disabling density fitting"
    if action == "fallback_cpu":
        return "CPU fallback requested; retrying CPU"
    return f"Retrying with action: {action}"

