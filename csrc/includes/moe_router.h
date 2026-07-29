#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Mixture of Experts (MoE) Top-K Router (Kernel 6), softmax-gating
// convention. See project_plan.md Section 3.6 and baselines/moe_router.py
// for the reference semantics and documented scope (softmax only, not
// sigmoid):
//
//   probs = softmax(logits, dim=-1)
//   topk_weights, topk_indices = topk(probs, k, dim=-1)
//   if renormalize: topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
//
// logits: `[num_tokens, num_experts]`, at any supported dtype (`dtype` —
// see common.cuh). topk_weights: `[num_tokens, k]`, always float32.
// topk_indices: `[num_tokens, k]`, int64. Supports num_experts up to 256
// (8 per warp lane) — see the .cu file; launch returns cudaErrorInvalidValue
// if num_experts exceeds that. Launch is a no-op if num_tokens <= 0.
extern "C" cudaError_t launch_moe_router_fwd(
    const void* logits,
    float* topk_weights,
    int64_t* topk_indices,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int32_t dtype,
    int32_t renormalize,
    cudaStream_t stream);
