"""Environment inspection logic backing the `doctor` command."""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys

__all__ = ["Check", "run_checks"]


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "fail" | "warn"
    detail: str


def _check_python() -> Check:
    v = sys.version_info
    return Check("Python interpreter", "ok", f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def _check_nvcc() -> Check:
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc")
    if nvcc is None and cuda_path:
        candidate = os.path.join(cuda_path, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
        if os.path.exists(candidate):
            nvcc = candidate
    if nvcc is None:
        return Check(
            "nvcc (CUDA compiler)", "fail",
            "not found on PATH and CUDA_PATH/CUDA_HOME unset — "
            "build.rs cannot compile csrc/kernels/*.cu",
        )
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        match = re.search(r"release (\d+\.\d+)", out.stdout)
        version = match.group(1) if match else "unknown"
        return Check("nvcc (CUDA compiler)", "ok", f"release {version} ({nvcc})")
    except (OSError, subprocess.SubprocessError) as e:
        return Check("nvcc (CUDA compiler)", "warn", f"found at {nvcc} but failed to run: {e}")


def _check_cuda_toolkit_path() -> Check:
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda_path:
        return Check(
            "CUDA_PATH / CUDA_HOME", "warn",
            "not set — build.rs requires one of these to locate the toolkit",
        )
    if not os.path.isdir(cuda_path):
        return Check(
            "CUDA_PATH / CUDA_HOME", "fail", f"set to {cuda_path!r} but that path doesn't exist"
        )
    return Check("CUDA_PATH / CUDA_HOME", "ok", cuda_path)


def _check_torch() -> Check:
    try:
        import torch
    except ImportError as e:
        return Check("PyTorch", "fail", f"not importable: {e}")
    cuda_ver = torch.version.cuda or "n/a"
    return Check("PyTorch", "ok", f"{torch.__version__} (built for CUDA {cuda_ver})")


def _check_cuda_available() -> Check:
    try:
        import torch
    except ImportError:
        return Check("torch.cuda.is_available()", "fail", "torch not importable")
    if not torch.cuda.is_available():
        return Check("torch.cuda.is_available()", "fail", "no CUDA device visible to PyTorch")
    return Check("torch.cuda.is_available()", "ok", "True")


def _check_gpu() -> Check:
    try:
        import torch
    except ImportError:
        return Check("GPU", "fail", "torch not importable")
    if not torch.cuda.is_available():
        return Check("GPU", "fail", "no CUDA device visible")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return Check("GPU", "ok", f"{name} (sm_{major}{minor}, {total_mem_gb:.1f} GB)")

def _check_fp8_support() -> Check:
    try:
        import torch
    except ImportError:
        return Check("FP8 dtype support", "fail", "torch not importable")
    if not torch.cuda.is_available():
        return Check("FP8 dtype support", "warn", "no CUDA device — cannot verify")
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 8:
        return Check(
            "FP8 dtype support", "warn",
            f"compute capability sm_{major}x detected — Kernel 12 (fp8_quant) needs sm_89+ "
            "(Ada) for good fp8 throughput",
        )
    return Check("FP8 dtype support", "ok", "compute capability supports fp8 (Ada/Hopper+)")


def _check_native_extension() -> Check:
    try:
        # Import order matters on Windows: PyTorch registers the DLL
        # search directories (its bundled CUDA runtime DLLs) as a side
        # effect of being imported. `_native.pyd` depends on those same
        # DLLs, so importing it before `torch` fails with a generic
        # "DLL load failed" even when the build is perfectly fine.
        import torch  # noqa: F401

        from custom_cuda import _native
    except ImportError as e:
        return Check(
            "custom_cuda._native (Rust extension)", "fail",
            f"not importable — run `maturin develop --release`: {e}",
        )
    required = ["rmsnorm_residual_fwd", "fp8_quant_block_fwd", "viterbi_fwd"]
    missing = [fn for fn in required if not hasattr(_native, fn)]
    if missing:
        return Check(
            "custom_cuda._native (Rust extension)", "warn",
            f"imported, but missing expected symbols: {missing} — extension may be stale, rebuild",
        )
    return Check("custom_cuda._native (Rust extension)", "ok", f"loaded from {_native.__file__}")


def _check_cuda_arch_match() -> Check:
    """Cross-check the arch build.rs targeted (CUDA_ARCH, default sm_89)
    against the actually-detected GPU's compute capability.
    """
    try:
        import torch
    except ImportError:
        return Check("Build arch vs. GPU arch", "warn", "torch not importable")
    if not torch.cuda.is_available():
        return Check("Build arch vs. GPU arch", "warn", "no CUDA device — cannot verify")
    major, minor = torch.cuda.get_device_capability(0)
    detected = f"sm_{major}{minor}"
    built_for = os.environ.get("CUDA_ARCH", "sm_89")
    if detected != built_for:
        return Check(
            "Build arch vs. GPU arch", "warn",
            f"kernels built for {built_for} (default/CUDA_ARCH), GPU is {detected} — "
            f"set CUDA_ARCH={detected} and rebuild for best performance if these differ",
        )
    return Check("Build arch vs. GPU arch", "ok", f"{detected} matches build target")


def run_checks() -> list[Check]:
    return [
        _check_python(),
        _check_cuda_toolkit_path(),
        _check_nvcc(),
        _check_torch(),
        _check_cuda_available(),
        _check_gpu(),
        _check_fp8_support(),
        _check_cuda_arch_match(),
        _check_native_extension(),
    ]
