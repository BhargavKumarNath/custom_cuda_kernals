#include "../includes/common.cuh"
#include "../includes/pairwise_distance.h"

// -----------------------------------------------------------------------
// Kernel 9 — Block Pairwise Distance Matrix Computation. Reuses Kernel
// 5's proven 1D register-blocked tiled-GEMM structure directly (rather
// than starting from a naive tiled kernel we already know from Kernel 5's
// benchmark data is shared-memory-bandwidth-bound at ~1.2-1.4 TFLOPS) —
// A plays the role of `x`, B plays the role of `weight`, and the epilogue
// combines the two precomputed row-norm arrays with the tiled dot product
// (`||a||^2 + ||b||^2 - 2*a.b`) instead of adding a bias, with an explicit
// `max(., 0)` clamp guarding against floating-point cancellation producing
// small negative squared distances for near-identical vectors (see
// baselines/pairwise_distance.py's `near_identical` case).
//
// Norms are precomputed by the Python wrapper (O(M*Dim + N*Dim), cheap
// relative to the O(M*N*Dim) tiled term) rather than fused into the tiled
// loop — the same row of A or B is read by many different output tiles,
// so computing its norm inline would redundantly recompute it once per
// tile that touches it instead of once, total.
// -----------------------------------------------------------------------

namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
// Benchmarking (benchmarks/pairwise_distance_bench.py's dimsweep_* cases)
// found throughput flat at ~3.2-3.8 TFLOPS regardless of `dim`, while
// cuBLAS-backed eager/cdist/compiled climbed steadily with `dim` —
// suggesting kBlockK=8's fixed per-tile sync overhead wasn't amortizing.
// Tried doubling to kBlockK=16 (halves tile count, doubles FMA work per
// tile; requires a strided 2-elements-per-thread load since block thread
// count no longer matches tile size 1:1): measured *no* improvement, and
// a regression at xlarge/dimsweep_128 (doubled shared-memory footprint
// per block likely reduced occupancy without fixing the actual
// bottleneck, which the flat TFLOPS across every M/N/dim combination
// tested — not just large-dim ones — suggests is proportional shared-
// memory bandwidth, the same ceiling Kernel 5 hit). Reverted to
// kBlockK=8; the dim-scaling gap vs. cuBLAS is accepted as an honest
// shortfall rather than chased further (see project_plan.md's Kernel 9
// entry).
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;  // outputs per thread along M
// Block shape: (kBlockN, kBlockM / kThreadM) = (64, 8) = 512 threads.
constexpr int kThreadsPerBlock = kBlockN * (kBlockM / kThreadM);

template <typename T>
__global__ void pairwise_distance_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ a_norm_sq,
    const float* __restrict__ b_norm_sq,
    float* __restrict__ dist_sq,
    int64_t m,
    int64_t n,
    int64_t dim) {
    __shared__ float a_tile[kBlockM][kBlockK];
    __shared__ float b_tile[kBlockK][kBlockN];

    const int64_t block_row0 = static_cast<int64_t>(blockIdx.y) * kBlockM;
    const int64_t block_col0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    const int tx = threadIdx.x;  // 0..kBlockN-1
    const int ty = threadIdx.y;  // 0..(kBlockM/kThreadM)-1
    const int linear_id = ty * kBlockN + tx;

    const int64_t col = block_col0 + tx;

    float acc[kThreadM];
#pragma unroll
    for (int i = 0; i < kThreadM; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t num_k_tiles = (dim + kBlockK - 1) / kBlockK;
    for (int64_t t = 0; t < num_k_tiles; ++t) {
        // a_tile[BLOCK_M][BLOCK_K]: 64*16 = 1024 elements, 2 per thread
        // (512 threads/block) via a strided load loop.
#pragma unroll
        for (int idx = linear_id; idx < kBlockM * kBlockK; idx += kThreadsPerBlock) {
            const int at_row = idx / kBlockK;
            const int at_col = idx % kBlockK;
            const int64_t a_row_g = block_row0 + at_row;
            const int64_t a_col_g = t * kBlockK + at_col;
            a_tile[at_row][at_col] = (a_row_g < m && a_col_g < dim) ? to_float(a[a_row_g * dim + a_col_g]) : 0.0f;
        }

        // b_tile[BLOCK_K][BLOCK_N]: 16*64 = 1024 elements, 2 per thread.
#pragma unroll
        for (int idx = linear_id; idx < kBlockK * kBlockN; idx += kThreadsPerBlock) {
            const int bt_row = idx / kBlockN;
            const int bt_col = idx % kBlockN;
            const int64_t b_row_g = block_col0 + bt_col;  // B is [N, Dim]
            const int64_t b_col_g = t * kBlockK + bt_row;
            b_tile[bt_row][bt_col] = (b_row_g < n && b_col_g < dim) ? to_float(b[b_row_g * dim + b_col_g]) : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            const float b_val = b_tile[kk][tx];
#pragma unroll
            for (int mi = 0; mi < kThreadM; ++mi) {
                acc[mi] += a_tile[ty * kThreadM + mi][kk] * b_val;
            }
        }
        __syncthreads();
    }

    const float bn = (col < n) ? b_norm_sq[col] : 0.0f;
#pragma unroll
    for (int mi = 0; mi < kThreadM; ++mi) {
        const int64_t row = block_row0 + ty * kThreadM + mi;
        if (row < m && col < n) {
            const float an = a_norm_sq[row];
            const float d = an + bn - 2.0f * acc[mi];
            dist_sq[row * n + col] = fmaxf(d, 0.0f);
        }
    }
}

template <typename T>
cudaError_t dispatch(
    const T* a, const T* b, const float* a_norm_sq, const float* b_norm_sq, float* dist_sq, int64_t m,
    int64_t n, int64_t dim, cudaStream_t stream) {
    const dim3 block(kBlockN, kBlockM / kThreadM);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
                     static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    pairwise_distance_kernel<T><<<grid, block, 0, stream>>>(a, b, a_norm_sq, b_norm_sq, dist_sq, m, n, dim);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_pairwise_distance_fwd(
    const void* a,
    const void* b,
    const float* a_norm_sq,
    const float* b_norm_sq,
    float* dist_sq,
    int64_t m,
    int64_t n,
    int64_t dim,
    int32_t dtype,
    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || dim <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(a), static_cast<const float*>(b), a_norm_sq, b_norm_sq,
                dist_sq, m, n, dim, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(a), static_cast<const __half*>(b), a_norm_sq, b_norm_sq,
                dist_sq, m, n, dim, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), a_norm_sq,
                b_norm_sq, dist_sq, m, n, dim, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
