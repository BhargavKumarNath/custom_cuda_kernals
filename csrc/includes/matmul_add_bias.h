#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused MatMul + Add Bias (Kernel 5): `y = x @ weight.T + bias`
// (nn.Linear convention). See project_plan.md Section 3.5 and
// baselines/matmul_add_bias.py for the reference semantics.
//
// x: `[M, K]`, weight: `[N, K]` (`[out_features, in_features]`), bias:
// `[N]` or `nullptr` (no bias), y: `[M, N]`. All row-major contiguous,
// same dtype (`dtype`, a KernelDType — see common.cuh). Launch is a no-op
// (returns cudaSuccess) if M, N, or K <= 0.
//
// Unlike Kernel 4 (which delegates its GEMM to PyTorch/cuBLAS and only
// hand-writes the fused reduction), this kernel *is* a hand-written tiled
// GEMM with a fused bias epilogue — the actual custom-GEMM deliverable in
// this library. See the .cu file for the tiling design and honest
// performance expectations relative to cuBLAS.
extern "C" cudaError_t launch_matmul_add_bias_fwd(
    const void* x,
    const void* weight,
    const void* bias,
    void* y,
    int64_t m,
    int64_t k,
    int64_t n,
    int32_t dtype,
    cudaStream_t stream);
