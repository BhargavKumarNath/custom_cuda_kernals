#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Fused Linear Cross Entropy Loss (Kernel 4) — chunk-update kernel. See
// project_plan.md Section 3.4 and baselines/linear_cross_entropy.py for
// the reference semantics and documented architecture: the vocab-dimension
// matmul chunk (`hidden @ weight_chunk.T`) is computed by PyTorch/cuBLAS
// in the Python wrapper (custom_cuda/kernels/linear_cross_entropy.py),
// which never materializes more than one `[N, chunk_v]` chunk at a time;
// this kernel fuses the online-softmax running-max/running-sum update and
// the target-logit gather for that chunk into a single pass, which is
// where the multi-kernel-launch elementwise chain (max, sub, exp, sum,
// gather) would otherwise cost extra global memory traffic per chunk.
//
// logits_chunk: `[N, chunk_v]`, at hidden/weight's native compute dtype
// (`dtype`, a KernelDType — see common.cuh) — v1 required the caller to
// upcast to float32 first; v2 reads the native dtype directly via
// to_float() like every other kernel in this library, since benchmarking
// showed the upcast copy was a fixed cost proportional to total elements
// processed, independent of chunk_size (see the .cu file's design-history
// comment). targets: `[N]` int64, global class indices (or
// `ignore_index`, which by construction never falls inside
// `[v_start, v_start+chunk_v)` and so is simply never matched — no
// special-casing needed in the kernel). running_max/running_sum/
// target_logit: `[N]` float32, in-out, must be initialized by the caller
// before the first chunk (running_max = -inf, running_sum = 0,
// target_logit = 0) and threaded through every chunk call in increasing
// `v_start` order.
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
    cudaStream_t stream);
