#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused RMSNorm + residual addition (Kernel 1). See project_plan.md Section
// 3.1 and baselines/rmsnorm_residual.py for the reference semantics:
//
//   residual_out = x + residual
//   y = residual_out / sqrt(mean(residual_out^2, axis=-1) + eps) * weight
//
// x, residual, weight, y, residual_out are row-major, contiguous, with
// `cols` == weight's length (the normalized/hidden dimension). `dtype` is a
// KernelDType (see common.cuh): 0 = f32, 1 = f16, 2 = bf16 — all five
// tensors share the same dtype. Launch is a no-op (returns cudaSuccess) if
// rows <= 0 or cols <= 0.
extern "C" cudaError_t launch_rmsnorm_residual_fwd(
    const void* x,
    const void* residual,
    const void* weight,
    void* y,
    void* residual_out,
    int64_t rows,
    int64_t cols,
    float eps,
    int32_t dtype,
    cudaStream_t stream);
