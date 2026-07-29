#include "../includes/common.cuh"
#include "../includes/token_permute.h"

// -----------------------------------------------------------------------
// Kernel 7 — Token Scatter and Gather (Permute / Unpermute).
//
// Both directions are one-block-per-output-row kernels, vectorized from
// the start (informed by Kernels 2/3/6's lessons: get block size right
// relative to per-row work, and vectorize via 16-byte chunks with an
// alignment/divisibility-gated dispatch + scalar fallback — see
// swiglu.cu/rope.cu for the pattern this reuses).
//
// Permute is pure row movement with no arithmetic, so its vectorized path
// treats every dtype as raw 16-byte (uint4) chunks — no per-element
// to_float()/from_float() conversion needed at all, unlike every other
// kernel in this library. Unpermute (combine) does need per-element math
// (the weighted sum across k gathered rows), so it follows the
// SwiGLU-style float4 (fp32) / uint4-packed half2-or-bf16x2 (fp16/bf16)
// vectorization instead.
//
// Block size is computed per-launch from the actual per-row work
// (`compute_block_size`), not a fixed constant — Kernel 3's vectorized
// RoPE regression (fixed 256-thread blocks launched against an 8-16
// iteration loop, wasting 240+ threads/block) is exactly the mistake this
// avoids up front.
// -----------------------------------------------------------------------

namespace {

constexpr int kMaxBlockSize = 256;
constexpr int kMinBlockSize = 32;

int compute_block_size(int64_t work_items) {
    int64_t rounded = ((work_items + 31) / 32) * 32;
    if (rounded < kMinBlockSize) rounded = kMinBlockSize;
    if (rounded > kMaxBlockSize) rounded = kMaxBlockSize;
    return static_cast<int>(rounded);
}

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0; }

// ---- Permute (gather) ----------------------------------------------------

template <typename T>
__global__ void token_gather_kernel_scalar(
    const T* __restrict__ src,
    const int64_t* __restrict__ indices,
    T* __restrict__ dst,
    int64_t hidden_dim) {
    const int64_t out_row = blockIdx.x;
    const int64_t src_row = indices[out_row];
    const T* src_ptr = src + src_row * hidden_dim;
    T* dst_ptr = dst + out_row * hidden_dim;
    for (int64_t c = threadIdx.x; c < hidden_dim; c += blockDim.x) {
        dst_ptr[c] = src_ptr[c];
    }
}

__global__ void token_gather_kernel_vec_raw(
    const uint4* __restrict__ src,
    const int64_t* __restrict__ indices,
    uint4* __restrict__ dst,
    int64_t row_stride_vec) {
    const int64_t out_row = blockIdx.x;
    const int64_t src_row = indices[out_row];
    const uint4* src_ptr = src + src_row * row_stride_vec;
    uint4* dst_ptr = dst + out_row * row_stride_vec;
    for (int64_t c = threadIdx.x; c < row_stride_vec; c += blockDim.x) {
        dst_ptr[c] = src_ptr[c];
    }
}

template <typename T>
cudaError_t dispatch_gather(
    const T* src, const int64_t* indices, T* dst, int64_t n_dst_rows, int64_t hidden_dim,
    cudaStream_t stream) {
    const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(T);
    const bool vecable = (row_bytes % 16 == 0) && is_aligned16(src) && is_aligned16(dst);
    const dim3 grid(static_cast<unsigned int>(n_dst_rows));

    if (vecable) {
        const int64_t row_stride_vec = static_cast<int64_t>(row_bytes / 16);
        const dim3 block(compute_block_size(row_stride_vec));
        token_gather_kernel_vec_raw<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint4*>(src), indices, reinterpret_cast<uint4*>(dst),
            row_stride_vec);
    } else {
        const dim3 block(compute_block_size(hidden_dim));
        token_gather_kernel_scalar<T><<<grid, block, 0, stream>>>(src, indices, dst, hidden_dim);
    }
    return cudaGetLastError();
}

// ---- Unpermute (weighted combine) ----------------------------------------

template <typename T>
__global__ void token_combine_kernel_scalar(
    const T* __restrict__ expert_output,
    const int64_t* __restrict__ unpermute_index,
    const float* __restrict__ weights,
    T* __restrict__ combined,
    int64_t k,
    int64_t hidden_dim) {
    const int64_t token = blockIdx.x;
    const int64_t* idx_row = unpermute_index + token * k;
    const float* w_row = weights + token * k;
    T* out_row = combined + token * hidden_dim;

    for (int64_t c = threadIdx.x; c < hidden_dim; c += blockDim.x) {
        float acc = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const T* src_row = expert_output + idx_row[j] * hidden_dim;
            acc += w_row[j] * to_float(src_row[c]);
        }
        out_row[c] = from_float<T>(acc);
    }
}

__global__ void token_combine_kernel_vec4_f32(
    const float* __restrict__ expert_output,
    const int64_t* __restrict__ unpermute_index,
    const float* __restrict__ weights,
    float* __restrict__ combined,
    int64_t k,
    int64_t hidden_dim_vec) {
    const int64_t token = blockIdx.x;
    const int64_t* idx_row = unpermute_index + token * k;
    const float* w_row = weights + token * k;
    float4* out_row = reinterpret_cast<float4*>(combined + token * hidden_dim_vec * 4);

    for (int64_t c = threadIdx.x; c < hidden_dim_vec; c += blockDim.x) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int64_t j = 0; j < k; ++j) {
            const float4* src_row =
                reinterpret_cast<const float4*>(expert_output + idx_row[j] * hidden_dim_vec * 4);
            const float4 v = src_row[c];
            const float w = w_row[j];
            acc.x += w * v.x;
            acc.y += w * v.y;
            acc.z += w * v.z;
            acc.w += w * v.w;
        }
        out_row[c] = acc;
    }
}

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
__global__ void token_combine_kernel_vec8_half(
    const T* __restrict__ expert_output,
    const int64_t* __restrict__ unpermute_index,
    const float* __restrict__ weights,
    T* __restrict__ combined,
    int64_t k,
    int64_t hidden_dim_vec) {
    using Ops = PackedOps<T, T2>;
    const int64_t token = blockIdx.x;
    const int64_t* idx_row = unpermute_index + token * k;
    const float* w_row = weights + token * k;
    uint4* out_row = reinterpret_cast<uint4*>(combined + token * hidden_dim_vec * 8);

    for (int64_t c = threadIdx.x; c < hidden_dim_vec; c += blockDim.x) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int64_t j = 0; j < k; ++j) {
            const uint4* src_row =
                reinterpret_cast<const uint4*>(expert_output + idx_row[j] * hidden_dim_vec * 8);
            const uint4 packed = src_row[c];
            const T2* p2 = reinterpret_cast<const T2*>(&packed);
            const float w = w_row[j];
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                acc[lane * 2] += w * Ops::low(p2[lane]);
                acc[lane * 2 + 1] += w * Ops::high(p2[lane]);
            }
        }
        uint4 out_packed;
        T2* op2 = reinterpret_cast<T2*>(&out_packed);
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            op2[lane] = Ops::pack(acc[lane * 2], acc[lane * 2 + 1]);
        }
        out_row[c] = out_packed;
    }
}

cudaError_t dispatch_combine_f32(
    const float* expert_output, const int64_t* unpermute_index, const float* weights, float* combined,
    int64_t n_tokens, int64_t k, int64_t hidden_dim, cudaStream_t stream) {
    const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const bool vecable = (row_bytes % 16 == 0) && is_aligned16(expert_output) && is_aligned16(combined);
    const dim3 grid(static_cast<unsigned int>(n_tokens));

    if (vecable) {
        const int64_t hidden_dim_vec = hidden_dim / 4;
        const dim3 block(compute_block_size(hidden_dim_vec));
        token_combine_kernel_vec4_f32<<<grid, block, 0, stream>>>(
            expert_output, unpermute_index, weights, combined, k, hidden_dim_vec);
    } else {
        const dim3 block(compute_block_size(hidden_dim));
        token_combine_kernel_scalar<float><<<grid, block, 0, stream>>>(
            expert_output, unpermute_index, weights, combined, k, hidden_dim);
    }
    return cudaGetLastError();
}

template <typename T, typename T2>
cudaError_t dispatch_combine_half_like(
    const T* expert_output, const int64_t* unpermute_index, const float* weights, T* combined,
    int64_t n_tokens, int64_t k, int64_t hidden_dim, cudaStream_t stream) {
    const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(T);
    const bool vecable = (row_bytes % 16 == 0) && is_aligned16(expert_output) && is_aligned16(combined);
    const dim3 grid(static_cast<unsigned int>(n_tokens));

    if (vecable) {
        const int64_t hidden_dim_vec = hidden_dim / 8;
        const dim3 block(compute_block_size(hidden_dim_vec));
        token_combine_kernel_vec8_half<T, T2><<<grid, block, 0, stream>>>(
            expert_output, unpermute_index, weights, combined, k, hidden_dim_vec);
    } else {
        const dim3 block(compute_block_size(hidden_dim));
        token_combine_kernel_scalar<T><<<grid, block, 0, stream>>>(
            expert_output, unpermute_index, weights, combined, k, hidden_dim);
    }
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_token_gather_fwd(
    const void* src,
    const int64_t* indices,
    void* dst,
    int64_t n_dst_rows,
    int64_t hidden_dim,
    int32_t dtype,
    cudaStream_t stream) {
    if (n_dst_rows <= 0 || hidden_dim <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_gather<float>(
                static_cast<const float*>(src), indices, static_cast<float*>(dst), n_dst_rows,
                hidden_dim, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch_gather<__half>(
                static_cast<const __half*>(src), indices, static_cast<__half*>(dst), n_dst_rows,
                hidden_dim, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch_gather<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(src), indices, static_cast<__nv_bfloat16*>(dst),
                n_dst_rows, hidden_dim, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t launch_token_combine_fwd(
    const void* expert_output,
    const int64_t* unpermute_index,
    const float* weights,
    void* combined,
    int64_t n_tokens,
    int64_t k,
    int64_t hidden_dim,
    int32_t dtype,
    cudaStream_t stream) {
    if (n_tokens <= 0 || k <= 0 || hidden_dim <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_combine_f32(
                static_cast<const float*>(expert_output), unpermute_index, weights,
                static_cast<float*>(combined), n_tokens, k, hidden_dim, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN((dispatch_combine_half_like<__half, __half2>(
                static_cast<const __half*>(expert_output), unpermute_index, weights,
                static_cast<__half*>(combined), n_tokens, k, hidden_dim, stream)));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN((dispatch_combine_half_like<__nv_bfloat16, __nv_bfloat162>(
                static_cast<const __nv_bfloat16*>(expert_output), unpermute_index, weights,
                static_cast<__nv_bfloat16*>(combined), n_tokens, k, hidden_dim, stream)));
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
