from gpu4pyscf_h200_runner.config import RunnerConfig
from gpu4pyscf_h200_runner.retry import RetryPlanner


def oom():
    return RuntimeError("CUDA out of memory")


def test_cleanup_retry_is_first():
    planner = RetryPlanner()
    config = RunnerConfig(xyz="molecule.xyz")
    new_config, decision = planner.next_retry(config, oom())
    assert new_config == config
    assert decision["action"] == "cleanup_gpu_memory_and_retry_same_settings"


def test_grid_lowering_after_cleanup():
    planner = RetryPlanner(cleanup_retry_done=True)
    config = RunnerConfig(xyz="molecule.xyz", grid_level=2)
    new_config, decision = planner.next_retry(config, oom())
    assert new_config.grid_level == 1
    assert decision["action"] == "lower_grid_level"


def test_basis_fallback():
    planner = RetryPlanner(cleanup_retry_done=True)
    config = RunnerConfig(xyz="molecule.xyz", grid_level=1, basis="def2-tzvp")
    new_config, decision = planner.next_retry(config, oom())
    assert new_config.basis == "def2-svp"
    assert decision["action"] == "fallback_basis"


def test_functional_fallback():
    planner = RetryPlanner(cleanup_retry_done=True)
    config = RunnerConfig(xyz="molecule.xyz", grid_level=1, basis="def2-svp", xc="cam-b3lyp")
    new_config, decision = planner.next_retry(config, oom())
    assert new_config.xc == "b3lyp"
    assert decision["action"] == "fallback_xc"


def test_cpu_fallback_only_when_requested():
    planner = RetryPlanner(
        cleanup_retry_done=True,
        basis_fallback_done=True,
        xc_fallback_done=True,
        df_disabled_done=True,
    )
    config = RunnerConfig(xyz="molecule.xyz", grid_level=1, density_fit=False, fallback_cpu=False)
    new_config, decision = planner.next_retry(config, oom())
    assert new_config is None
    assert decision is None

    requested = RunnerConfig(xyz="molecule.xyz", grid_level=1, density_fit=False, fallback_cpu=True)
    new_config, decision = planner.next_retry(requested, oom())
    assert new_config.backend == "cpu"
    assert decision["action"] == "fallback_cpu"

