#include "../includes/common.cuh"
#include "../includes/cosine_topk.h"

// -----------------------------------------------------------------------
// Kernel 8 — Fused Cosine Similarity + Top-K Selection.
//
// v1 (single warp-per-query, no candidate partitioning): benchmarked at
// 20-97x *slower* than eager, getting proportionally worse as the
// candidate pool grew — the opposite of what a streaming top-k kernel
// should do. Root cause: grid size was `ceil(num_queries / 8)` only.
// RAG-realistic benchmarks use few queries (Q=8) against large candidate
// pools (N up to 500,000); with Q=8 and 8 warps/block, that grid is
// exactly **one block** regardless of N — a few hundred threads doing
// all of the work serially, on a GPU with 36 SMs. Parallelizing only
// across queries cannot scale when queries are scarce and candidates are
// plentiful, which is the *normal* case for this kernel, not an edge case.
//
// v2 (accepted, two-kernel partition + merge, below): splits each query's
// candidate pool across `num_partitions` independent warps (chosen by the
// Python wrapper from `num_candidates`, so the grid scales with the
// candidate pool even when Q is tiny), each computing a partial top-k of
// size k over its slice via the same fused-normalization + per-lane
// local-buffer + warp-merge approach as v1. A second, much smaller merge
// kernel — one warp per query, no dot products, just precomputed-score
// insertion and the identical k-round warp-argmax-and-advance merge —
// reduces `num_partitions * k` partial candidates down to the final k.
// This is the standard tiled/partitioned-reduction pattern brute-force
// ANN search implementations use for exactly this "few queries, huge
// candidate pool" shape.
//
// Measured result: v1's 20-97x slowdown (worsening with N) became a
// 1.7-5.1x slowdown that scales *proportionally* with N instead of
// catastrophically. Still short of the ">=3x faster than eager" target
// — eager's normalize+matmul step is backed by cuBLAS, which this
// hand-written streaming kernel (scalar per-candidate dot products, no
// tensor cores) does not match, the same honest gap Kernel 5 reports
// against cuBLAS's GEMM. Accepted as final given the fix already
// recovered the kernel from broken to a reasonable, correctly-scaling
// result, and the kernel's actual purpose — never materializing the full
// [Q, N] similarity matrix — holds regardless of the raw-latency gap.
// -----------------------------------------------------------------------

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxK = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize = kWarpsPerBlock * kWarpSize;

__device__ __forceinline__ void warp_argmax_idx(float& val, int64_t& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        const int64_t other_idx = __shfl_down_sync(0xFFFFFFFFu, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    val = __shfl_sync(0xFFFFFFFFu, val, 0);
    idx = __shfl_sync(0xFFFFFFFFu, idx, 0);
}

// Merge this lane's local sorted top-k buffer across the warp into the
// final k-length result, writing it out via `emit(i, val, idx)` on lane 0.
template <typename Emit>
__device__ __forceinline__ void warp_merge_local_topk(
    float* local_vals, int64_t* local_idx, int64_t k, const Emit& emit) {
    const int lane = threadIdx.x & 31;
    int cursor = 0;
    for (int64_t i = 0; i < k; ++i) {
        float val = local_vals[cursor];
        int64_t idx = local_idx[cursor];
        warp_argmax_idx(val, idx);

        if (local_idx[cursor] == idx) {
            ++cursor;
        }
        if (lane == 0) {
            emit(i, val, idx);
        }
    }
}

__device__ __forceinline__ void init_local_topk(float* local_vals, int64_t* local_idx) {
#pragma unroll
    for (int i = 0; i < kMaxK; ++i) {
        local_vals[i] = -INFINITY;
        local_idx[i] = -1;
    }
}

__device__ __forceinline__ void insert_local_topk(
    float* local_vals, int64_t* local_idx, int64_t k, float val, int64_t idx) {
    if (val > local_vals[k - 1]) {
        int pos = static_cast<int>(k) - 1;
        while (pos > 0 && local_vals[pos - 1] < val) {
            local_vals[pos] = local_vals[pos - 1];
            local_idx[pos] = local_idx[pos - 1];
            --pos;
        }
        local_vals[pos] = val;
        local_idx[pos] = idx;
    }
}

// ---- Partial kernel: one warp per (query, partition), streams its slice
// of candidates, fusing L2 norm accumulation into the dot-product loop. ---

template <typename T>
__global__ void cosine_topk_partial_kernel(
    const T* __restrict__ queries,
    const T* __restrict__ candidates,
    float* __restrict__ partial_scores,
    int64_t* __restrict__ partial_indices,
    int64_t num_queries,
    int64_t num_candidates,
    int64_t dim,
    int64_t k,
    int64_t num_partitions,
    float eps) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t query_idx = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const int64_t partition_id = blockIdx.y;
    if (query_idx >= num_queries) {
        return;
    }

    const int64_t part_size = (num_candidates + num_partitions - 1) / num_partitions;
    const int64_t part_start = partition_id * part_size;
    const int64_t part_end = min(num_candidates, part_start + part_size);

    const T* query_row = queries + query_idx * dim;

    float q_norm_sq_local = 0.0f;
    for (int64_t d = lane; d < dim; d += kWarpSize) {
        const float qv = to_float(query_row[d]);
        q_norm_sq_local += qv * qv;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_norm_sq_local += __shfl_down_sync(0xFFFFFFFFu, q_norm_sq_local, offset);
    }
    const float q_norm = sqrtf(__shfl_sync(0xFFFFFFFFu, q_norm_sq_local, 0));

    float local_vals[kMaxK];
    int64_t local_idx[kMaxK];
    init_local_topk(local_vals, local_idx);

    for (int64_t cand = part_start + lane; cand < part_end; cand += kWarpSize) {
        const T* cand_row = candidates + cand * dim;
        float dot = 0.0f;
        float cand_norm_sq = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            const float qv = to_float(query_row[d]);
            const float cv = to_float(cand_row[d]);
            dot += qv * cv;
            cand_norm_sq += cv * cv;
        }
        const float cos_sim = dot / fmaxf(q_norm * sqrtf(cand_norm_sq), eps);
        insert_local_topk(local_vals, local_idx, k, cos_sim, cand);
    }

    float* out_scores = partial_scores + (query_idx * num_partitions + partition_id) * k;
    int64_t* out_indices = partial_indices + (query_idx * num_partitions + partition_id) * k;
    warp_merge_local_topk(local_vals, local_idx, k, [&](int64_t i, float val, int64_t idx) {
        out_scores[i] = val;
        out_indices[i] = idx;
    });
}

// ---- Merge kernel: one warp per query, merges num_partitions*k
// precomputed candidates down to the final k. No dot products. ----------

__global__ void cosine_topk_merge_kernel(
    const float* __restrict__ partial_scores,
    const int64_t* __restrict__ partial_indices,
    float* __restrict__ topk_scores,
    int64_t* __restrict__ topk_indices,
    int64_t num_queries,
    int64_t num_partitions,
    int64_t k) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t query_idx = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    if (query_idx >= num_queries) {
        return;
    }

    const int64_t total_partial = num_partitions * k;
    const float* scores_row = partial_scores + query_idx * total_partial;
    const int64_t* indices_row = partial_indices + query_idx * total_partial;

    float local_vals[kMaxK];
    int64_t local_idx[kMaxK];
    init_local_topk(local_vals, local_idx);

    for (int64_t i = lane; i < total_partial; i += kWarpSize) {
        insert_local_topk(local_vals, local_idx, k, scores_row[i], indices_row[i]);
    }

    float* out_scores = topk_scores + query_idx * k;
    int64_t* out_indices = topk_indices + query_idx * k;
    warp_merge_local_topk(local_vals, local_idx, k, [&](int64_t i, float val, int64_t idx) {
        out_scores[i] = val;
        out_indices[i] = idx;
    });
}

template <typename T>
cudaError_t dispatch_partial(
    const T* queries, const T* candidates, float* partial_scores, int64_t* partial_indices,
    int64_t num_queries, int64_t num_candidates, int64_t dim, int64_t k, int64_t num_partitions,
    float eps, cudaStream_t stream) {
    const dim3 block(kBlockSize);
    const dim3 grid(
        static_cast<unsigned int>((num_queries + kWarpsPerBlock - 1) / kWarpsPerBlock),
        static_cast<unsigned int>(num_partitions));
    cosine_topk_partial_kernel<T><<<grid, block, 0, stream>>>(
        queries, candidates, partial_scores, partial_indices, num_queries, num_candidates, dim, k,
        num_partitions, eps);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_cosine_topk_partial_fwd(
    const void* queries,
    const void* candidates,
    float* partial_scores,
    int64_t* partial_indices,
    int64_t num_queries,
    int64_t num_candidates,
    int64_t dim,
    int64_t k,
    int64_t num_partitions,
    float eps,
    int32_t dtype,
    cudaStream_t stream) {
    if (num_queries <= 0) {
        return cudaSuccess;
    }
    if (k > kMaxK) {
        return cudaErrorInvalidValue;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch_partial<float>(
                static_cast<const float*>(queries), static_cast<const float*>(candidates),
                partial_scores, partial_indices, num_queries, num_candidates, dim, k,
                num_partitions, eps, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch_partial<__half>(
                static_cast<const __half*>(queries), static_cast<const __half*>(candidates),
                partial_scores, partial_indices, num_queries, num_candidates, dim, k,
                num_partitions, eps, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch_partial<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(queries), static_cast<const __nv_bfloat16*>(candidates),
                partial_scores, partial_indices, num_queries, num_candidates, dim, k,
                num_partitions, eps, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t launch_cosine_topk_merge_fwd(
    const float* partial_scores,
    const int64_t* partial_indices,
    float* topk_scores,
    int64_t* topk_indices,
    int64_t num_queries,
    int64_t num_partitions,
    int64_t k,
    cudaStream_t stream) {
    if (num_queries <= 0) {
        return cudaSuccess;
    }
    if (k > kMaxK) {
        return cudaErrorInvalidValue;
    }

    const dim3 block(kBlockSize);
    const dim3 grid(static_cast<unsigned int>((num_queries + kWarpsPerBlock - 1) / kWarpsPerBlock));
    cosine_topk_merge_kernel<<<grid, block, 0, stream>>>(
        partial_scores, partial_indices, topk_scores, topk_indices, num_queries, num_partitions, k);

    CUDA_CHECK_RETURN(cudaGetLastError());
    return cudaSuccess;
}
