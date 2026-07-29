#include "../includes/common.cuh"
#include "../includes/graph_message_passing.h"

// -----------------------------------------------------------------------
// Kernel 10 — Spatiotemporal Graph Message Passing. CSR-based neighbor
// iteration (both edge sets pre-sorted by destination node in Python —
// cheap O(E) bookkeeping, the same delegation pattern as Kernel 4/7/8/9)
// with warp-per-node parallelism and a grid-stride loop over nodes so a
// modest, fixed-size grid handles arbitrarily many nodes.
//
// Design note on "shared-memory staging to minimize atomic contention"
// (project_plan.md Section 3.10): within a warp, the 32 lanes are split
// across the FEATURE dimension (lane `l` owns feature indices `l, l+32,
// l+64, ...`), not across the neighbor list. Combined with CSR-by-
// destination giving each warp exclusive ownership of one node's full
// incoming-edge range (across both spatial and temporal sets), every
// output element `out[node, f]` is written by exactly one lane, exactly
// once — there is no cross-lane combination step to stage anywhere, in
// shared memory or otherwise, and atomics are eliminated entirely rather
// than merely reduced (the same "eliminate rather than reduce" pattern
// Kernel 6/7 established for weaker asks). This v1 re-reads each
// neighbor's small `col`/`weight` scalars once per feature-pass
// (`feature_dim / 32` times) directly from global memory rather than
// staging them in shared memory first; benchmarking
// (benchmarks/graph_message_passing_bench.py) determines whether that
// redundant scalar traffic is actually worth caching, per this project's
// empirical-iteration policy, rather than adding the complexity
// speculatively.
// -----------------------------------------------------------------------

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;  // 256
constexpr int64_t kMaxBlocks = 4096;
// Each lane owns kFeaturesPerThread strided feature stripes (f, f+32,
// f+64, f+96, ...) processed together per neighbor visit — register-
// blocking the feature dimension the same way Kernel 5/9's kThreadM
// blocks the M dimension. Benchmarking v1 (kFeaturesPerThread=1) showed
// the gap to torch.compile widening as average node degree grew
// (benchmarks/graph_message_passing_bench.py's dsweep_* cases): a single
// warp gathers one node's entire neighbor list serially, and with only
// one independent load in flight per lane per neighbor, there's little
// memory-level parallelism to hide gather latency behind. Issuing 4
// independent loads per neighbor per lane gives the hardware more
// overlapping work per traversal of the (data-dependent-length)
// neighbor list, at the cost of 4 live accumulator registers per lane
// instead of 1.
constexpr int kFeaturesPerThread = 4;

template <typename T>
__global__ void graph_message_passing_kernel(
    const T* __restrict__ x_curr,
    const T* __restrict__ x_prev,
    const int64_t* __restrict__ spatial_indptr,
    const int64_t* __restrict__ spatial_col,
    const float* __restrict__ spatial_weight,
    const int64_t* __restrict__ temporal_indptr,
    const int64_t* __restrict__ temporal_col,
    const float* __restrict__ temporal_weight,
    T* __restrict__ out,
    int64_t num_nodes,
    int64_t feature_dim) {
    const int warp_id_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t warps_per_grid = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
    const int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id_in_block;

    for (int64_t node = global_warp_id; node < num_nodes; node += warps_per_grid) {
        const int64_t s_begin = spatial_indptr[node];
        const int64_t s_end = spatial_indptr[node + 1];
        const int64_t t_begin = temporal_indptr[node];
        const int64_t t_end = temporal_indptr[node + 1];

        for (int64_t f0 = lane; f0 < feature_dim; f0 += kWarpSize * kFeaturesPerThread) {
            float acc[kFeaturesPerThread];
#pragma unroll
            for (int j = 0; j < kFeaturesPerThread; ++j) {
                acc[j] = 0.0f;
            }

            for (int64_t e = s_begin; e < s_end; ++e) {
                const int64_t src = spatial_col[e];
                const float w = spatial_weight[e];
                const T* row = x_curr + src * feature_dim;
#pragma unroll
                for (int j = 0; j < kFeaturesPerThread; ++j) {
                    const int64_t f = f0 + j * kWarpSize;
                    if (f < feature_dim) {
                        acc[j] += w * to_float(row[f]);
                    }
                }
            }
            for (int64_t e = t_begin; e < t_end; ++e) {
                const int64_t src = temporal_col[e];
                const float w = temporal_weight[e];
                const T* row = x_prev + src * feature_dim;
#pragma unroll
                for (int j = 0; j < kFeaturesPerThread; ++j) {
                    const int64_t f = f0 + j * kWarpSize;
                    if (f < feature_dim) {
                        acc[j] += w * to_float(row[f]);
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < kFeaturesPerThread; ++j) {
                const int64_t f = f0 + j * kWarpSize;
                if (f < feature_dim) {
                    out[node * feature_dim + f] = from_float<T>(acc[j]);
                }
            }
        }
    }
}

template <typename T>
cudaError_t dispatch(
    const T* x_curr, const T* x_prev, const int64_t* spatial_indptr, const int64_t* spatial_col,
    const float* spatial_weight, const int64_t* temporal_indptr, const int64_t* temporal_col,
    const float* temporal_weight, T* out, int64_t num_nodes, int64_t feature_dim, cudaStream_t stream) {
    const int64_t warps_needed = (num_nodes + 0);  // one warp per node
    const int64_t blocks_needed = (warps_needed + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int64_t blocks = blocks_needed < kMaxBlocks ? blocks_needed : kMaxBlocks;
    const dim3 grid(static_cast<unsigned int>(blocks < 1 ? 1 : blocks));
    const dim3 block(kBlockSize);
    graph_message_passing_kernel<T><<<grid, block, 0, stream>>>(
        x_curr, x_prev, spatial_indptr, spatial_col, spatial_weight, temporal_indptr, temporal_col,
        temporal_weight, out, num_nodes, feature_dim);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_graph_message_passing_fwd(
    const void* x_curr,
    const void* x_prev,
    const int64_t* spatial_indptr,
    const int64_t* spatial_col,
    const float* spatial_weight,
    const int64_t* temporal_indptr,
    const int64_t* temporal_col,
    const float* temporal_weight,
    void* out,
    int64_t num_nodes,
    int64_t feature_dim,
    int32_t dtype,
    cudaStream_t stream) {
    if (num_nodes <= 0 || feature_dim <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(x_curr), static_cast<const float*>(x_prev), spatial_indptr,
                spatial_col, spatial_weight, temporal_indptr, temporal_col, temporal_weight,
                static_cast<float*>(out), num_nodes, feature_dim, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(x_curr), static_cast<const __half*>(x_prev), spatial_indptr,
                spatial_col, spatial_weight, temporal_indptr, temporal_col, temporal_weight,
                static_cast<__half*>(out), num_nodes, feature_dim, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(x_curr), static_cast<const __nv_bfloat16*>(x_prev),
                spatial_indptr, spatial_col, spatial_weight, temporal_indptr, temporal_col, temporal_weight,
                static_cast<__nv_bfloat16*>(out), num_nodes, feature_dim, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
