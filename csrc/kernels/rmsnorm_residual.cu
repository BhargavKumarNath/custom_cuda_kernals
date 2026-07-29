#include "../includes/common.cuh"
#include "../includes/rmsnorm_residual.h"

// -----------------------------------------------------------------------
// Kernel 1 — Fused RMSNorm + Residual Addition.
//
// Design history (project_plan.md Section 2 / Phase 1 adaptive strategy):
//
// v1: one block per row, 256-thread block, grid-stride loop over columns,
// (x + residual) recomputed in a second pass rather than cached, no
// vectorization. benchmarks/rmsnorm_residual_bench.py showed this already
// hitting 80-89% of the RTX 4070 Laptop's peak memory bandwidth and a
// 4.7-5.2x speedup vs. eager for fp16/bf16 (fp32: ~2.4x, just under the
// 2.5x target — eager's fp32 path has no dtype-upcast overhead to
// eliminate, unlike fp16/bf16, so there's less headroom there by
// construction).
//
// v2 (tried, rejected): staged (x + residual) into dynamic shared memory
// during pass 1 to avoid the pass-2 re-read, hypothesizing the re-read was
// costing a full extra DRAM round trip. Benchmarked: no measurable
// improvement on the shapes v1 was already near-peak on, and a ~5%
// *regression* on several fp16/bf16 shapes (e.g. batch=4/seq=1024/h=4096:
// 206.9 -> 196.7 GB/s) — the 16-46KB of shared memory reserved per block
// reduces the number of blocks resident per SM, and that occupancy loss
// outweighed the saved global read. The saved read turned out to be nearly
// free already: each row is owned by exactly one block for its whole
// lifetime, so pass 2's "redundant" read of x/residual is typically still
// an L2 hit, not a DRAM round trip. Reverted — the simpler recompute
// version below is both less code and at least as fast.
// -----------------------------------------------------------------------

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void rmsnorm_residual_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ y,
    T* __restrict__ residual_out,
    int64_t cols,
    float eps) {
    const int64_t row = blockIdx.x;
    const T* x_row = x + row * cols;
    const T* r_row = residual + row * cols;
    T* y_row = y + row * cols;
    T* ro_row = residual_out + row * cols;

    __shared__ float warp_sums[kBlockSize / 32];

    // Pass 1: accumulate sum-of-squares of (x + residual) in fp32.
    float sum_sq = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += kBlockSize) {
        const float v = to_float(x_row[c]) + to_float(r_row[c]);
        sum_sq += v * v;
    }

    const float total = block_reduce_sum<kBlockSize>(sum_sq, warp_sums);
    const float rms_inv = rsqrtf(total / static_cast<float>(cols) + eps);

    // Pass 2: recompute (x + residual) — see design note above on why this
    // is not the bottleneck it looks like — write residual_out, apply
    // norm+scale.
    for (int64_t c = threadIdx.x; c < cols; c += kBlockSize) {
        const float v = to_float(x_row[c]) + to_float(r_row[c]);
        ro_row[c] = from_float<T>(v);
        const float w = to_float(weight[c]);
        y_row[c] = from_float<T>(v * rms_inv * w);
    }
}

}  // namespace

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
    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        return cudaSuccess;
    }

    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(kBlockSize);

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            rmsnorm_residual_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float*>(x), static_cast<const float*>(residual),
                static_cast<const float*>(weight), static_cast<float*>(y),
                static_cast<float*>(residual_out), cols, eps);
            break;
        case KernelDType::F16:
            rmsnorm_residual_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(x), static_cast<const __half*>(residual),
                static_cast<const __half*>(weight), static_cast<__half*>(y),
                static_cast<__half*>(residual_out), cols, eps);
            break;
        case KernelDType::BF16:
            rmsnorm_residual_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(residual),
                static_cast<const __nv_bfloat16*>(weight), static_cast<__nv_bfloat16*>(y),
                static_cast<__nv_bfloat16*>(residual_out), cols, eps);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    CUDA_CHECK_RETURN(cudaGetLastError());
    return cudaSuccess;
}
