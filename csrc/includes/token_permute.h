#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Token Scatter and Gather / Permute-Unpermute (Kernel 7). See
// project_plan.md Section 3.7 and baselines/token_permute.py for the
// reference semantics and the index-computation-vs-kernel architecture
// split (permutation indices are computed in Python via `argsort`; these
// kernels do the bandwidth-critical row movement).

// Permute: `dst[i] = src[indices[i]]`, a pure row gather. Since this
// operation does no per-element arithmetic, it is dtype-agnostic at the
// byte level — the vectorized path treats every dtype's row as raw
// 16-byte chunks. `src`: `[S, H]`, `indices`: `[N]` (values in [0, S)),
// `dst`: `[N, H]`. Launch is a no-op if n_dst_rows or hidden_dim <= 0.
extern "C" cudaError_t launch_token_gather_fwd(
    const void* src,
    const int64_t* indices,
    void* dst,
    int64_t n_dst_rows,
    int64_t hidden_dim,
    int32_t dtype,
    cudaStream_t stream);

// Unpermute (weighted combine):
// `combined[t] = sum_j weights[t,j] * expert_output[unpermute_index[t,j]]`.
// `expert_output`: `[N, H]`, `unpermute_index`: `[T, k]` (values in
// [0, N)), `weights`: `[T, k]`, always float32 (matches Kernel 6's
// convention), `combined`: `[T, H]`. Launch is a no-op if n_tokens,
// k, or hidden_dim <= 0.
extern "C" cudaError_t launch_token_combine_fwd(
    const void* expert_output,
    const int64_t* unpermute_index,
    const float* weights,
    void* combined,
    int64_t n_tokens,
    int64_t k,
    int64_t hidden_dim,
    int32_t dtype,
    cudaStream_t stream);
