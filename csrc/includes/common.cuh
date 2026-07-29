#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK_RETURN(expr)              \
    do {                                      \
        cudaError_t _cuda_err = (expr);        \
        if (_cuda_err != cudaSuccess) {        \
            return _cuda_err;                  \
        }                                       \
    } while (0)

// ---------------------------------------------------------------------------
// dtype tag shared across every kernel's extern "C" launcher ABI. Values are
// part of the Rust <-> C++ contract (see src/kernels/dtype.rs) — do not
// renumber without updating both sides.
// ---------------------------------------------------------------------------
// F8E4M3/F8E5M2 are additive tags for Kernel 12's fp8 output tensors —
// no existing kernel's dispatch<T>() switches on them, so this is a
// backwards-compatible extension (every other kernel's switch already
// has a `default: return cudaErrorInvalidValue` arm).
enum class KernelDType : int32_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    F8E5M2 = 4,
};

// ---------------------------------------------------------------------------
// Scalar <-> float conversions, specialized per supported storage dtype.
// Every kernel reduces/accumulates in float regardless of storage dtype, for
// numerical stability (matches the fp32-upcast convention in the PyTorch
// eager baselines under baselines/).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

// ---------------------------------------------------------------------------
// Warp / block level reduction utilities, shared by every kernel that needs
// a per-row (or per-tile) sum reduction (RMSNorm, softmax-style ops, etc.).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

// Block-wide sum reduction. `warp_sums_scratch` must point to at least
// BLOCK_SIZE/32 floats of shared memory. The result is broadcast to every
// thread in the block (all threads may read the return value).
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* warp_sums_scratch) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of the warp size");
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums_scratch[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < BLOCK_SIZE / 32) ? warp_sums_scratch[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            warp_sums_scratch[0] = v;
        }
    }
    __syncthreads();
    return warp_sums_scratch[0];
}

// ---------------------------------------------------------------------------
// Warp / block level max reduction — used by online-softmax-style kernels
// (e.g. Kernel 4's chunked cross-entropy) that need a running max alongside
// a running sum.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

// Block-wide max reduction. Same shared-memory/broadcast contract as
// block_reduce_sum above (reuse the same scratch buffer only if the two
// aren't needed concurrently — otherwise use separate scratch arrays).
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* warp_max_scratch) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of the warp size");
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_max_scratch[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < BLOCK_SIZE / 32) ? warp_max_scratch[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) {
            warp_max_scratch[0] = v;
        }
    }
    __syncthreads();
    return warp_max_scratch[0];
}
