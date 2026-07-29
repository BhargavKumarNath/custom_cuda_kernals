#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused Cosine Similarity + Top-K Selection (Kernel 8). See
// project_plan.md Section 3.8 and baselines/cosine_topk.py for the
// reference semantics:
//
//   cos_sim[q, n] = dot(query_q, candidate_n) / max(||query_q|| * ||candidate_n||, eps)
//   topk_scores, topk_indices = topk(cos_sim, k, dim=-1)
//
// Two-kernel design (see the .cu file's header comment for why a single
// warp-per-query kernel catastrophically underutilizes the GPU when the
// query count is small, the realistic RAG case): `launch_cosine_topk_partial_fwd`
// partitions the candidate pool across `num_partitions` independent warps
// per query, each producing a partial top-k of size k; `launch_cosine_topk_merge_fwd`
// merges the `num_partitions` partial top-k lists (`num_partitions * k`
// precomputed scores, already <= a few thousand) into the final top-k.
// The full `[Q, N]` similarity matrix is never materialized in either
// kernel. The Python wrapper (custom_cuda/kernels/cosine_topk.py) chooses
// `num_partitions` and allocates the intermediate `[Q, num_partitions, k]`
// buffers, orchestrating both launches (same architecture pattern as
// Kernel 4's chunking loop).

// queries: `[Q, D]`, candidates: `[N, D]`, at any supported dtype
// (`dtype` — see common.cuh). partial_scores/partial_indices:
// `[Q, num_partitions, k]` (float32 / int64). `k` must be `<= 32`.
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
    cudaStream_t stream);

// Merges `[Q, num_partitions, k]` partial results into the final
// `[Q, k]` top-k. No dot products here — pure index/score merge.
extern "C" cudaError_t launch_cosine_topk_merge_fwd(
    const float* partial_scores,
    const int64_t* partial_indices,
    float* topk_scores,
    int64_t* topk_indices,
    int64_t num_queries,
    int64_t num_partitions,
    int64_t k,
    cudaStream_t stream);
