#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Parallel Viterbi Algorithm (Kernel 11). See project_plan.md Section
// 3.11 and baselines/viterbi.py for the reference semantics: batched
// Viterbi decoding of a single (shared-across-the-batch) HMM.
//
//   delta[b,0,s]  = log_pi[s] + log_emission[b,0,s]
//   delta[b,t,s]  = max_k(delta[b,t-1,k] + log_trans[k,s]) + log_emission[b,t,s]
//   psi[b,t,s]    = argmax_k(delta[b,t-1,k] + log_trans[k,s])
//   best_score[b] = max_s delta[b,T-1,s]
//   best_path[b,T-1]  = argmax_s delta[b,T-1,s]
//   best_path[b,t]    = psi[b,t+1,best_path[b,t+1]]        (t = T-2..0)
//
// One persistent-kernel launch (O(1), not O(T)): one block per batch
// item, internally looping over all T timesteps with `__syncthreads()`
// standing in for what would otherwise be T separate kernel launches.
// `log_trans` is cached in shared memory once and stays resident for
// the whole recursion (`[S,S]`, always float32 — a shared, tiny array,
// not a per-example feature tensor to quantize, same convention as
// Kernel 9's row norms / Kernel 10's edge weights). `log_emission`:
// `[B,T,S]`, any supported dtype (see common.cuh) — the one tensor
// whose size scales with the problem, hence the one dtype varies. `psi`:
// `[B,T,S]` int64 scratch (backpointers; row 0 unused), allocated by the
// caller. `best_path`: `[B,T]` int64. `best_score`: `[B]` float32.
// Launch is a no-op if batch, seq_len, or num_states <= 0; fails with
// cudaErrorInvalidValue if `num_states` exceeds 1024 (one block's max
// thread count — this design parallelizes only across states within a
// block, not within a state's own transition scan, so num_states beyond
// this needs a different design not implemented here).
extern "C" cudaError_t launch_viterbi_fwd(
    const void* log_emission,
    const float* log_trans,
    const float* log_pi,
    int64_t* psi,
    int64_t* best_path,
    float* best_score,
    int64_t batch,
    int64_t seq_len,
    int64_t num_states,
    int32_t dtype,
    cudaStream_t stream);
