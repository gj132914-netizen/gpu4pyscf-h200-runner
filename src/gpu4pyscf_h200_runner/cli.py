"""Command line interface for g4pyscf-run."""

from __future__ import annotations

from .config import parse_config
from .runner import run_job


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        summary = run_job(config)
        return 0 if summary.get("success") else 1
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

