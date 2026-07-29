#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// FP8 Dynamic Quantization & Casting (Kernel 12). See project_plan.md
// Section 3.12 and baselines/fp8_quant.py for the reference semantics:
//
//   scale = max(amax / FP8_MAX, EPS)
//   x_fp8 = (x / scale) cast to fp8 (e4m3fn or e5m2)
//
// `x`: `[M, N]`, any supported input dtype (see common.cuh's
// `KernelDType`). `x_fp8`: `[M, N]` fp8 bytes (raw storage — caller's
// tensor dtype must be `torch.float8_e4m3fn`/`torch.float8_e5m2`
// matching `fp8_format`; PyTorch's byte layout for both is the OCP
// encoding CUDA's `cuda_fp8.h` also uses, so a raw `uint8_t*` write here
// is directly readable as that torch dtype). `fp8_format`: one of
// `KernelDType::F8E4M3`/`F8E5M2`.
//
// Two entry points, matching project_plan.md Section 3.12's two scale
// granularities:
//
//   - `launch_fp8_quant_block_fwd`: one `128x128`-tile scale per block
//     (DeepSeek-V3-style). A single kernel launch — each thread block
//     owns one tile end-to-end (local amax reduction, scale derivation,
//     vectorized cast+store), so there is no cross-block synchronization
//     and no intermediate global write of the amax. `scale`:
//     `[ceil(M/128), ceil(N/128)]` float32.
//   - `launch_fp8_quant_tensor_fwd`: one scale for the whole tensor.
//     A true single-KERNEL fusion isn't possible here (the scale
//     depends on a reduction over every block's output, which needs
//     either a second kernel or grid-wide cooperative-groups sync); this
//     is implemented as two kernel launches on the same stream — an
//     amax-reduction pass (grid-stride + block-reduce + a single
//     non-negative-float atomicMax per block into `amax_scratch`,
//     caller-zeroed) followed by a scale+cast pass — exposed as one
//     launcher call. `amax_scratch`: `[1]` int32 scratch (caller-
//     allocated and zeroed). `scale`: `[1]` float32.
//
// Both no-op (return cudaSuccess) if m, n <= 0.
extern "C" cudaError_t launch_fp8_quant_block_fwd(
    const void* x,
    uint8_t* x_fp8,
    float* scale,
    int64_t m,
    int64_t n,
    int32_t dtype,
    int32_t fp8_format,
    cudaStream_t stream);

extern "C" cudaError_t launch_fp8_quant_tensor_fwd(
    const void* x,
    uint8_t* x_fp8,
    float* scale,
    int32_t* amax_scratch,
    int64_t m,
    int64_t n,
    int32_t dtype,
    int32_t fp8_format,
    cudaStream_t stream);
