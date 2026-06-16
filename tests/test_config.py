import json
import os
from pathlib import Path

from gpu4pyscf_h200_runner.config import RunnerConfig, config_to_dict, parse_config, setup_environment


def test_default_h200_config():
    config = parse_config(["--xyz", "molecule.xyz"])
    assert config.backend == "gpu"
    assert config.gpu_method == "to_gpu"
    assert config.fallback_cpu is False
    assert config.memory_mb == 80000
    assert config.threads == 12
    assert config.grid_level == 2
    assert config.density_fit is True


def test_json_config_override(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"xyz": "a.xyz", "backend": "cpu", "threads": 4}), encoding="utf-8")
    config = parse_config(["--config", str(config_path)])
    assert config.xyz == Path("a.xyz")
    assert config.backend == "cpu"
    assert config.threads == 4


def test_cli_overrides_config_file(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"xyz": "a.xyz", "backend": "cpu", "threads": 4}), encoding="utf-8")
    config = parse_config(["--config", str(config_path), "--backend", "gpu", "--threads", "9"])
    assert config.backend == "gpu"
    assert config.threads == 9
    assert config_to_dict(config)["xyz"] == "a.xyz"


def test_lowmem_policy_changes_large_basis_without_explicit_gradient():
    config = parse_config(["--xyz", "molecule.xyz", "--basis", "def2-tzvp", "--gradient", "--lowmem"])
    assert config.basis == "def2-svp"
    assert config.grid_level == 1
    assert config.gradient is True


def test_setup_environment_respects_existing_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-assigned")
    setup_environment(RunnerConfig(gpu_id=None, gpu_memory_limit="92%", threads=12))
    assert config_to_dict(RunnerConfig(gpu_id=None))["gpu_id"] is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "scheduler-assigned"


def test_setup_environment_gpu_id_overrides_when_explicit(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-assigned")
    setup_environment(RunnerConfig(gpu_id=1, gpu_memory_limit="92%", threads=12))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
