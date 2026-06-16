from gpu4pyscf_h200_runner.exceptions import is_oom_error


def test_oom_strings_are_detected():
    for message in [
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "failed to allocate 1024 bytes",
        "cupy.cuda.memory.OutOfMemoryError: out of memory",
        "insufficient memory on device",
        "allocator failed",
    ]:
        assert is_oom_error(RuntimeError(message))


def test_non_oom_exception_is_not_detected():
    assert not is_oom_error(ValueError("SCF did not converge"))

