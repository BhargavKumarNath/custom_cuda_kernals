#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Block Pairwise Distance Matrix Computation (Kernel 9). See
// project_plan.md Section 3.9 and baselines/pairwise_distance.py for the
// reference semantics:
//
//   dist_sq[i, j] = max(||A[i]||^2 + ||B[j]||^2 - 2 * dot(A[i], B[j]), 0)
//
// A: `[M, Dim]`, B: `[N, Dim]`, at any supported dtype (`dtype` — see
// common.cuh). `a_norm_sq`/`b_norm_sq`: `[M]`/`[N]`, precomputed float32
// squared row norms (computed by the Python wrapper — cheap O(M*Dim +
// N*Dim) work delegated the same way Kernel 4 delegates its matmul,
// keeping this kernel focused on the O(M*N*Dim) tiled dot-product term).
// `dist_sq`: `[M, N]`, always float32. Reuses the 1D register-blocked
// tiled-GEMM structure proven in Kernel 5 (`x @ weight.T` there, `A`
// against `B` here), with a norm-combination + `max(., 0)`
// cancellation-guard epilogue in place of the bias add. Launch is a
// no-op if m, n, or dim <= 0.
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
    cudaStream_t stream);
