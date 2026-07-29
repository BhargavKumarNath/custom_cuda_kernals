#include "../includes/common.cuh"
#include "../includes/fp8_quant.h"

// -----------------------------------------------------------------------
// Kernel 12 — FP8 Dynamic Quantization & Casting.
//
// Block-granularity path (the primary, headline "single fused pass"):
// one thread block per 128x128 tile, computing that tile's own amax
// (block-wide reduction), deriving its own scale, and casting+storing
// its own output — no cross-block communication, no intermediate global
// write of the amax. Each thread re-reads its assigned elements for the
// cast/store pass rather than caching them from the amax pass (a tile
// is at most 128*128*4 bytes = 64KB, comfortably L2-resident by the
// second pass; caching 64 elements/thread in registers instead would
// add real register pressure for no measured benefit — worth revisiting
// only if benchmarking shows the second global read is a genuine
// bottleneck, not speculatively).
//
// Tensor-granularity path: a true single-kernel fusion isn't possible
// (the scale depends on a reduction over the *entire* tensor, which
// needs either a second kernel or grid-wide cooperative-groups sync);
// implemented as two kernel launches on the same stream, the same
// "two-kernel when a single launch can't do it" precedent Kernel 8's
// partition+merge design set.
//
// Both paths pack 16 fp8 bytes into a uint4 for vectorized stores where
// the destination offset is 16-byte aligned and a full 16-wide run is
// in bounds, falling back to a scalar per-byte store otherwise — the
// same align-and-divisibility-gated vectorized/scalar-fallback pattern
// established in Kernels 2/6/7.
// -----------------------------------------------------------------------

namespace {

constexpr float kMinScale = 1e-12f;

__device__ __forceinline__ uint8_t cvt_fp8(float v, __nv_fp8_interpretation_t fp8_interp) {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, fp8_interp);
}

__device__ __forceinline__ uint4 pack16_fp8(const uint8_t bytes[16]) {
    uint32_t words[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t packed = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            packed |= (static_cast<uint32_t>(bytes[w * 4 + b]) << (b * 8));
        }
        words[w] = packed;
    }
    return make_uint4(words[0], words[1], words[2], words[3]);
}

// ---------------------------------------------------------------------
// Block-granularity (128x128 tile) path.
// ---------------------------------------------------------------------

constexpr int kTileSize = 128;
constexpr int kBlockThreads = 256;
constexpr int kGroupWidth = 16;
constexpr int kGroupsPerRow = kTileSize / kGroupWidth;              // 8
constexpr int kPairsPerTile = kTileSize * kGroupsPerRow;            // 1024
constexpr int kPairsPerThread = kPairsPerTile / kBlockThreads;      // 4

template <typename T>
__global__ void fp8_quant_block_kernel(
    const T* __restrict__ x,
    uint8_t* __restrict__ x_fp8,
    float* __restrict__ scale_out,
    int64_t m,
    int64_t n,
    int num_col_blocks,
    float fp8_max,
    __nv_fp8_interpretation_t fp8_interp) {
    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    const int64_t row0 = static_cast<int64_t>(bi) * kTileSize;
    const int64_t col0 = static_cast<int64_t>(bj) * kTileSize;
    const int64_t row_end = min(row0 + kTileSize, m);
    const int64_t col_end = min(col0 + kTileSize, n);
    const int tid = threadIdx.x;

    float local_amax = 0.0f;
#pragma unroll
    for (int p = 0; p < kPairsPerThread; ++p) {
        const int pair_idx = tid + p * kBlockThreads;
        const int local_row = pair_idx / kGroupsPerRow;
        const int local_group = pair_idx % kGroupsPerRow;
        const int64_t row = row0 + local_row;
        const int64_t col_start = col0 + local_group * kGroupWidth;
        if (row >= row_end) continue;
        const int64_t valid_width = min(static_cast<int64_t>(kGroupWidth), col_end - col_start);
        for (int64_t c = 0; c < valid_width; ++c) {
            local_amax = fmaxf(local_amax, fabsf(to_float(x[row * n + col_start + c])));
        }
    }

    __shared__ float warp_scratch[kBlockThreads / 32];
    const float tile_amax = block_reduce_max<kBlockThreads>(local_amax, warp_scratch);

    __shared__ float s_scale;
    if (tid == 0) {
        const float scale = fmaxf(tile_amax / fp8_max, kMinScale);
        s_scale = scale;
        scale_out[bi * num_col_blocks + bj] = scale;
    }
    __syncthreads();
    const float inv_scale = 1.0f / s_scale;

#pragma unroll
    for (int p = 0; p < kPairsPerThread; ++p) {
        const int pair_idx = tid + p * kBlockThreads;
        const int local_row = pair_idx / kGroupsPerRow;
        const int local_group = pair_idx % kGroupsPerRow;
        const int64_t row = row0 + local_row;
        const int64_t col_start = col0 + local_group * kGroupWidth;
        if (row >= row_end) continue;
        const int64_t valid_width = min(static_cast<int64_t>(kGroupWidth), col_end - col_start);
        const int64_t out_offset = row * n + col_start;
        const bool full_group = (valid_width == kGroupWidth);
        const bool aligned = (reinterpret_cast<uintptr_t>(x_fp8 + out_offset) % 16u) == 0;

        if (full_group && aligned) {
            uint8_t bytes[16];
#pragma unroll
            for (int c = 0; c < kGroupWidth; ++c) {
                const float v = to_float(x[row * n + col_start + c]) * inv_scale;
                bytes[c] = cvt_fp8(v, fp8_interp);
            }
            *reinterpret_cast<uint4*>(x_fp8 + out_offset) = pack16_fp8(bytes);
        } else {
            for (int64_t c = 0; c < valid_width; ++c) {
                const float v = to_float(x[row * n + col_start + c]) * inv_scale;
                x_fp8[out_offset + c] = cvt_fp8(v, fp8_interp);
            }
        }
    }
}

template <typename T>
cudaError_t dispatch_block(
    const T* x, uint8_t* x_fp8, float* scale, int64_t m, int64_t n, float fp8_max,
    __nv_fp8_interpretation_t fp8_interp, cudaStream_t stream) {
    const int num_row_blocks = static_cast<int>((m + kTileSize - 1) / kTileSize);
    const int num_col_blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
    const dim3 grid(static_cast<unsigned int>(num_col_blocks), static_cast<unsigned int>(num_row_blocks));
    const dim3 block(kBlockThreads);
    fp8_quant_block_kernel<T><<<grid, block, 0, stream>>>(
        x, x_fp8, scale, m, n, num_col_blocks, fp8_max, fp8_interp);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------
// Tensor-granularity (whole-matrix scale) path: amax-reduce kernel +
// scale/cast kernel.
// ---------------------------------------------------------------------

constexpr int kReduceBlockThreads = 256;
constexpr int64_t kMaxReduceBlocks = 4096;

template <typename T>
__global__ void fp8_amax_reduce_kernel(const T* __restrict__ x, int64_t total_elems, int32_t* amax_bits) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float local_max = 0.0f;
    for (; idx < total_elems; idx += stride) {
        local_max = fmaxf(local_max, fabsf(to_float(x[idx])));
    }
    __shared__ float warp_scratch[kReduceBlockThreads / 32];
    const float block_max = block_reduce_max<kReduceBlockThreads>(local_max, warp_scratch);
    if (threadIdx.x == 0) {
        atomicMax(amax_bits, __float_as_int(block_max));
    }
}

template <typename T>
__global__ void fp8_scale_cast_flat_kernel(
    const T* __restrict__ x,
    uint8_t* __restrict__ x_fp8,
    float* __restrict__ scale_out,
    const int32_t* __restrict__ amax_bits,
    int64_t total_elems,
    float fp8_max,
    __nv_fp8_interpretation_t fp8_interp,
    bool ptr_aligned) {
    const float amax = __int_as_float(*amax_bits);
    const float scale = fmaxf(amax / fp8_max, kMinScale);
    const float inv_scale = 1.0f / scale;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale_out = scale;
    }

    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * kGroupWidth;
    int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kGroupWidth;
    for (; base < total_elems; base += stride) {
        const int64_t remaining = total_elems - base;
        if (remaining >= kGroupWidth && ptr_aligned) {
            uint8_t bytes[16];
#pragma unroll
            for (int c = 0; c < kGroupWidth; ++c) {
                const float v = to_float(x[base + c]) * inv_scale;
                bytes[c] = cvt_fp8(v, fp8_interp);
            }
            *reinterpret_cast<uint4*>(x_fp8 + base) = pack16_fp8(bytes);
        } else {
            const int64_t n_left = remaining < kGroupWidth ? remaining : static_cast<int64_t>(kGroupWidth);
            for (int64_t c = 0; c < n_left; ++c) {
                const float v = to_float(x[base + c]) * inv_scale;
                x_fp8[base + c] = cvt_fp8(v, fp8_interp);
            }
        }
    }
}

template <typename T>
cudaError_t dispatch_tensor(
    const T* x, uint8_t* x_fp8, float* scale, int32_t* amax_scratch, int64_t m, int64_t n, float fp8_max,
    __nv_fp8_interpretation_t fp8_interp, cudaStream_t stream) {
    const int64_t total_elems = m * n;
    const int64_t blocks_needed = (total_elems + kReduceBlockThreads - 1) / kReduceBlockThreads;
    const int64_t reduce_blocks = blocks_needed < kMaxReduceBlocks ? blocks_needed : kMaxReduceBlocks;
    fp8_amax_reduce_kernel<T><<<static_cast<unsigned int>(reduce_blocks < 1 ? 1 : reduce_blocks),
                                 kReduceBlockThreads, 0, stream>>>(x, total_elems, amax_scratch);

    const bool ptr_aligned = (reinterpret_cast<uintptr_t>(x_fp8) % 16u) == 0;
    const int64_t groups_needed = (total_elems + kGroupWidth - 1) / kGroupWidth;
    const int64_t cast_blocks_needed = (groups_needed + kReduceBlockThreads - 1) / kReduceBlockThreads;
    const int64_t cast_blocks = cast_blocks_needed < kMaxReduceBlocks ? cast_blocks_needed : kMaxReduceBlocks;
    fp8_scale_cast_flat_kernel<T><<<static_cast<unsigned int>(cast_blocks < 1 ? 1 : cast_blocks),
                                     kReduceBlockThreads, 0, stream>>>(
        x, x_fp8, scale, amax_scratch, total_elems, fp8_max, fp8_interp, ptr_aligned);
    return cudaGetLastError();
}

float fp8_max_for(int32_t fp8_format) {
    return static_cast<KernelDType>(fp8_format) == KernelDType::F8E5M2 ? 57344.0f : 448.0f;
}

__nv_fp8_interpretation_t fp8_interp_for(int32_t fp8_format) {
    return static_cast<KernelDType>(fp8_format) == KernelDType::F8E5M2 ? __NV_E5M2 : __NV_E4M3;
}

}  // namespace

extern "C" cudaError_t launch_fp8_quant_block_fwd(
    const void* x,
    uint8_t* x_fp8,
    float* scale,
    int64_t m,
    int64_t n,
    int32_t dtype,
    int32_t fp8_format,
    cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return cudaSuccess;
    }
    const float fp8_max = fp8_max_for(fp8_format);
    const __nv_fp8_interpretation_t fp8_interp = fp8_interp_for(fp8_format);

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_block<float>(
                static_cast<const float*>(x), x_fp8, scale, m, n, fp8_max, fp8_interp, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch_block<__half>(
                static_cast<const __half*>(x), x_fp8, scale, m, n, fp8_max, fp8_interp, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch_block<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(x), x_fp8, scale, m, n, fp8_max, fp8_interp, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t launch_fp8_quant_tensor_fwd(
    const void* x,
    uint8_t* x_fp8,
    float* scale,
    int32_t* amax_scratch,
    int64_t m,
    int64_t n,
    int32_t dtype,
    int32_t fp8_format,
    cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return cudaSuccess;
    }
    const float fp8_max = fp8_max_for(fp8_format);
    const __nv_fp8_interpretation_t fp8_interp = fp8_interp_for(fp8_format);

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_tensor<float>(
                static_cast<const float*>(x), x_fp8, scale, amax_scratch, m, n, fp8_max, fp8_interp, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch_tensor<__half>(
                static_cast<const __half*>(x), x_fp8, scale, amax_scratch, m, n, fp8_max, fp8_interp, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch_tensor<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(x), x_fp8, scale, amax_scratch, m, n, fp8_max, fp8_interp,
                stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
