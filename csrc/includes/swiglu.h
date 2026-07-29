#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused SwiGLU gated activation (Kernel 2). See project_plan.md Section 3.2
// and baselines/swiglu.py for the reference semantics:
//
//   y = SiLU(gate) * up = (gate * sigmoid(gate)) * up
//
// gate, up, y are flat contiguous buffers of `n` elements (any shape — the
// op is purely elementwise, so the caller flattens). `dtype` is a
// KernelDType (see common.cuh): 0 = f32, 1 = f16, 2 = bf16 — all three
// tensors share the same dtype. Launch is a no-op (returns cudaSuccess) if
// n <= 0.
extern "C" cudaError_t launch_swiglu_fwd(
    const void* gate,
    const void* up,
    void* y,
    int64_t n,
    int32_t dtype,
    cudaStream_t stream);
