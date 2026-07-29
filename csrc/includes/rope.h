#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused Rotary Position Embedding (Kernel 3), half-split convention. See
// project_plan.md Section 3.3 and baselines/rope.py for the reference
// semantics and documented scope (half-split only, position_ids =
// arange(seq_len), cos/sin always fp32):
//
//   x1, x2 = x[..., :d/2], x[..., d/2:]
//   out[..., :d/2] = x1*cos - x2*sin
//   out[..., d/2:] = x2*cos + x1*sin
//
// q: [batch, seq_len, n_q_heads, head_dim], k: [batch, seq_len,
// n_kv_heads, head_dim] (GQA: n_kv_heads <= n_q_heads), both row-major
// contiguous, same dtype. cos_table/sin_table: [seq_len, head_dim/2],
// always float32 regardless of q/k's dtype. q_out/k_out: caller-allocated
// outputs, same shape/dtype as q/k respectively. `dtype` is a KernelDType
// (0=f32, 1=f16, 2=bf16) applying to q/k/q_out/k_out. Single kernel
// launch rotates both q and k. Launch is a no-op if batch/seq_len/head_dim
// <= 0.
extern "C" cudaError_t launch_rope_fwd(
    const void* q,
    const void* k,
    const void* cos_table,
    const void* sin_table,
    void* q_out,
    void* k_out,
    int64_t batch,
    int64_t seq_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    int32_t dtype,
    cudaStream_t stream);
