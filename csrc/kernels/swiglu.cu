#include "../includes/common.cuh"
#include "../includes/swiglu.h"

// -----------------------------------------------------------------------
// Kernel 2 — Fused SwiGLU Gated Activation.
//
// v1 (scalar, kept below as `swiglu_kernel_scalar`): flat grid-stride loop,
// one element per thread iteration. benchmarks/swiglu_bench.py showed this
// already clearing the 1.8x speedup target by a wide margin (2.4-6.4x) but
// falling short of the 90% bandwidth target on mid-size shapes — and
// notably, fp16/bf16 bandwidth (77-79%) was *lower* than fp32's (84-85%)
// on the same shapes, which is backwards from what pure DRAM-traffic
// (bytes moved, already normalized for in the benchmark) would predict.
// That pattern points at transaction efficiency, not traffic volume: 2-byte
// scalar loads/stores under-use the memory bus width per instruction.
//
// v2 (`swiglu_kernel_vec4_f32` / `swiglu_kernel_vec8_half` below, now the
// default dispatch path): vectorized loads — float4 (4 elements/thread,
// 16B transactions) for fp32, uint4-packed (8 elements/thread, 16B
// transactions) for fp16/bf16. Requires 16-byte-aligned pointers and is
// applied to the floor(n/width)*width prefix; any remainder (from a
// non-vector-width-divisible n, e.g. the npot_dim edge case) is handled by
// a second, small scalar-kernel launch over the tail. Falls back to the
// pure scalar kernel entirely if any of gate/up/y isn't 16-byte aligned
// (e.g. an odd-element-offset slice) so correctness never depends on
// allocator behavior.
// -----------------------------------------------------------------------

namespace {

constexpr int kBlockSize = 256;
constexpr int64_t kMaxGridDimX = 65535;

__device__ __forceinline__ float silu_mul(float g, float u) {
    return (g / (1.0f + expf(-g))) * u;
}

dim3 grid_for(int64_t work_items) {
    const int64_t blocks_needed = (work_items + kBlockSize - 1) / kBlockSize;
    return dim3(static_cast<unsigned int>(blocks_needed < kMaxGridDimX ? blocks_needed : kMaxGridDimX));
}

bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// ---- scalar (fallback + tail) ------------------------------------------

template <typename T>
__global__ void swiglu_kernel_scalar(
    const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ y, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        y[i] = from_float<T>(silu_mul(to_float(gate[i]), to_float(up[i])));
    }
}

// ---- fp32: float4, 4 elements/thread ------------------------------------

__global__ void swiglu_kernel_vec4_f32(
    const float4* __restrict__ gate, const float4* __restrict__ up, float4* __restrict__ y,
    int64_t n_vec) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float4 g = gate[i];
        const float4 u = up[i];
        float4 out;
        out.x = silu_mul(g.x, u.x);
        out.y = silu_mul(g.y, u.y);
        out.z = silu_mul(g.z, u.z);
        out.w = silu_mul(g.w, u.w);
        y[i] = out;
    }
}

// ---- fp16 / bf16: uint4-packed, 8 elements/thread -----------------------
// A uint4 (16 bytes) holds four 2-wide packed pairs (half2 or
// __nv_bfloat162). Each pair's lanes are unpacked to float, computed, and
// repacked — same fp32 compute convention as the scalar path, just wider
// per-transaction.

template <typename T, typename T2>
struct PackedOps;

template <>
struct PackedOps<__half, __half2> {
    __device__ __forceinline__ static float low(const __half2& p) { return __half2float(__low2half(p)); }
    __device__ __forceinline__ static float high(const __half2& p) { return __half2float(__high2half(p)); }
    __device__ __forceinline__ static __half2 pack(float lo, float hi) {
        return __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    }
};

template <>
struct PackedOps<__nv_bfloat16, __nv_bfloat162> {
    __device__ __forceinline__ static float low(const __nv_bfloat162& p) {
        return __bfloat162float(__low2bfloat16(p));
    }
    __device__ __forceinline__ static float high(const __nv_bfloat162& p) {
        return __bfloat162float(__high2bfloat16(p));
    }
    __device__ __forceinline__ static __nv_bfloat162 pack(float lo, float hi) {
        return __halves2bfloat162(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
    }
};

template <typename T, typename T2>
__global__ void swiglu_kernel_vec8_half(
    const uint4* __restrict__ gate, const uint4* __restrict__ up, uint4* __restrict__ y,
    int64_t n_vec) {
    using Ops = PackedOps<T, T2>;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const uint4 gu = gate[i];
        const uint4 uu = up[i];
        const T2* g2 = reinterpret_cast<const T2*>(&gu);
        const T2* u2 = reinterpret_cast<const T2*>(&uu);

        uint4 out;
        T2* out2 = reinterpret_cast<T2*>(&out);
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const float result_lo = silu_mul(Ops::low(g2[lane]), Ops::low(u2[lane]));
            const float result_hi = silu_mul(Ops::high(g2[lane]), Ops::high(u2[lane]));
            out2[lane] = Ops::pack(result_lo, result_hi);
        }
        y[i] = out;
    }
}

// ---- dispatch -------------------------------------------------------

cudaError_t dispatch_f32(const float* gate, const float* up, float* y, int64_t n, cudaStream_t stream) {
    constexpr int64_t kWidth = 4;
    if (n >= kWidth && is_aligned16(gate) && is_aligned16(up) && is_aligned16(y)) {
        const int64_t n_vec = n / kWidth;
        const int64_t tail_start = n_vec * kWidth;
        swiglu_kernel_vec4_f32<<<grid_for(n_vec), kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(gate), reinterpret_cast<const float4*>(up),
            reinterpret_cast<float4*>(y), n_vec);
        if (tail_start < n) {
            const int64_t tail_n = n - tail_start;
            swiglu_kernel_scalar<float><<<grid_for(tail_n), kBlockSize, 0, stream>>>(
                gate + tail_start, up + tail_start, y + tail_start, tail_n);
        }
    } else {
        swiglu_kernel_scalar<float><<<grid_for(n), kBlockSize, 0, stream>>>(gate, up, y, n);
    }
    return cudaGetLastError();
}

template <typename T, typename T2>
cudaError_t dispatch_half_like(const T* gate, const T* up, T* y, int64_t n, cudaStream_t stream) {
    constexpr int64_t kWidth = 8;
    if (n >= kWidth && is_aligned16(gate) && is_aligned16(up) && is_aligned16(y)) {
        const int64_t n_vec = n / kWidth;
        const int64_t tail_start = n_vec * kWidth;
        swiglu_kernel_vec8_half<T, T2><<<grid_for(n_vec), kBlockSize, 0, stream>>>(
            reinterpret_cast<const uint4*>(gate), reinterpret_cast<const uint4*>(up),
            reinterpret_cast<uint4*>(y), n_vec);
        if (tail_start < n) {
            const int64_t tail_n = n - tail_start;
            swiglu_kernel_scalar<T><<<grid_for(tail_n), kBlockSize, 0, stream>>>(
                gate + tail_start, up + tail_start, y + tail_start, tail_n);
        }
    } else {
        swiglu_kernel_scalar<T><<<grid_for(n), kBlockSize, 0, stream>>>(gate, up, y, n);
    }
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_swiglu_fwd(
    const void* gate,
    const void* up,
    void* y,
    int64_t n,
    int32_t dtype,
    cudaStream_t stream) {
    if (n <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_f32(
                static_cast<const float*>(gate), static_cast<const float*>(up),
                static_cast<float*>(y), n, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN((dispatch_half_like<__half, __half2>(
                static_cast<const __half*>(gate), static_cast<const __half*>(up),
                static_cast<__half*>(y), n, stream)));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN((dispatch_half_like<__nv_bfloat16, __nv_bfloat162>(
                static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up),
                static_cast<__nv_bfloat16*>(y), n, stream)));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
