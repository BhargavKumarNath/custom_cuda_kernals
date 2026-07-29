#include "../includes/common.cuh"
#include "../includes/linear_cross_entropy.h"

// -----------------------------------------------------------------------
// Kernel 4 — Fused Linear Cross Entropy Loss, chunk-update kernel.
//
// v1 (scalar, fp32-only): required the Python wrapper to `.float()` each
// chunk before calling — a full extra copy of every chunk, proportional
// to total N*V regardless of chunk_size. benchmarks/linear_cross_entropy_
// bench.py's chunk-size sweep showed latency nearly flat across
// chunk_size (1 to 126 chunks all within noise of each other) while
// memory scaled smoothly with chunk_size — i.e. the ~10-14% overhead vs.
// eager wasn't a per-chunk-launch cost, it was a constant cost
// proportional to total elements processed. That's exactly the signature
// of the upcast copy.
//
// v2 (templated on storage dtype, like Kernels 1-3): reads chunk elements
// directly at hidden's native dtype via to_float() and eliminates the
// Python-side `.float()` cast entirely — the only fp32 materialization
// left is the internal reduction math, matching every other kernel in
// this library. running_max/running_sum/target_logit accumulators stay
// float32 regardless of storage dtype (same convention as Kernels 1-3).
// -----------------------------------------------------------------------

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void linear_ce_chunk_update_kernel(
    const T* __restrict__ logits_chunk,
    const int64_t* __restrict__ targets,
    int64_t v_start,
    int64_t chunk_v,
    float* __restrict__ running_max,
    float* __restrict__ running_sum,
    float* __restrict__ target_logit) {
    const int64_t row = blockIdx.x;
    const T* row_logits = logits_chunk + row * chunk_v;
    const int64_t target = targets[row];

    __shared__ float warp_scratch[kBlockSize / 32];

    // Pass 1: local max over this chunk, and gather the target logit if
    // this chunk happens to contain the target column.
    float local_max = -INFINITY;
    for (int64_t c = threadIdx.x; c < chunk_v; c += kBlockSize) {
        const float v = to_float(row_logits[c]);
        local_max = fmaxf(local_max, v);
        if (v_start + c == target) {
            target_logit[row] = v;
        }
    }
    const float chunk_max = block_reduce_max<kBlockSize>(local_max, warp_scratch);
    __syncthreads();  // warp_scratch is reused below by block_reduce_sum

    const float old_max = running_max[row];
    const float new_max = fmaxf(old_max, chunk_max);

    // Pass 2: sum of exp(logit - new_max) for this chunk.
    float local_sum = 0.0f;
    for (int64_t c = threadIdx.x; c < chunk_v; c += kBlockSize) {
        local_sum += expf(to_float(row_logits[c]) - new_max);
    }
    const float chunk_sum = block_reduce_sum<kBlockSize>(local_sum, warp_scratch);

    if (threadIdx.x == 0) {
        const float old_sum = running_sum[row];
        // expf(-INFINITY) == 0.0f exactly (IEEE 754), so this is correct
        // on the very first chunk (old_max == -inf) without a branch.
        const float rescale = expf(old_max - new_max);
        running_sum[row] = old_sum * rescale + chunk_sum;
        running_max[row] = new_max;
    }
}

template <typename T>
cudaError_t dispatch(
    const T* logits_chunk,
    const int64_t* targets,
    int64_t v_start,
    int64_t n,
    int64_t chunk_v,
    float* running_max,
    float* running_sum,
    float* target_logit,
    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(n));
    const dim3 block(kBlockSize);
    linear_ce_chunk_update_kernel<T><<<grid, block, 0, stream>>>(
        logits_chunk, targets, v_start, chunk_v, running_max, running_sum, target_logit);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_linear_ce_chunk_update(
    const void* logits_chunk,
    const int64_t* targets,
    int64_t v_start,
    int64_t n,
    int64_t chunk_v,
    float* running_max,
    float* running_sum,
    float* target_logit,
    int32_t dtype,
    cudaStream_t stream) {
    if (n <= 0 || chunk_v <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(logits_chunk), targets, v_start, n, chunk_v, running_max,
                running_sum, target_logit, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(logits_chunk), targets, v_start, n, chunk_v, running_max,
                running_sum, target_logit, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(logits_chunk), targets, v_start, n, chunk_v,
                running_max, running_sum, target_logit, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
