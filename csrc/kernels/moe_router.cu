#include "../includes/common.cuh"
#include "../includes/moe_router.h"

// -----------------------------------------------------------------------
// Kernel 6 — MoE Top-K Router. One warp per token: each lane holds up to
// kMaxLocal experts in registers, softmax (max + sum-of-exp) is computed
// via warp-shuffle reduction, and top-k is selected by `k` iterations of
// a warp-wide argmax-and-mask — each iteration finds the current global
// max via warp shuffle, the one lane that owns it invalidates that slot,
// and the loop repeats. Entirely register/shuffle-resident per token: no
// shared memory, no global-memory round trip for intermediate softmax
// values, and no CPU-GPU synchronization (dynamic `k` and `num_experts`
// are ordinary kernel arguments, not host-side branches).
//
// Every case here executes in well under a millisecond (E and k are
// small by construction), so this kernel lives in the launch-overhead-
// bound regime rather than being compute- or bandwidth-bound.
// benchmarks/moe_router_bench.py's first pass (kWarpsPerBlock=8, 256
// threads/block) beat eager on most shapes but only by ~1.1-2.2x against
// the >=3x target, and was noisy enough to show one shape (Mixtral-scale,
// fp32) as a slight *regression*. Doubling to kWarpsPerBlock=16 (512
// threads/block — halving the number of blocks launched for a given
// token count) improved every shape, most dramatically on the case that
// mattered most (Mixtral-scale fp16: 2.2x -> 3.35x, clearing the target),
// with the rest landing around 1.4-2.3x. Not pushed further (32+
// warps/block) given diminishing expected returns once already
// launch-overhead-bound rather than occupancy-bound.
// -----------------------------------------------------------------------

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxLocal = 8;  // supports up to 256 experts (8 per lane)
constexpr int kWarpsPerBlock = 16;
constexpr int kBlockSize = kWarpsPerBlock * kWarpSize;

__device__ __forceinline__ void warp_argmax(float& val, int& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        const int other_idx = __shfl_down_sync(0xFFFFFFFFu, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    val = __shfl_sync(0xFFFFFFFFu, val, 0);
    idx = __shfl_sync(0xFFFFFFFFu, idx, 0);
}

template <typename T>
__global__ void moe_router_kernel(
    const T* __restrict__ logits,
    float* __restrict__ topk_weights,
    int64_t* __restrict__ topk_indices,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    bool renormalize) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t token = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    if (token >= num_tokens) {
        return;
    }

    const T* row = logits + token * num_experts;

    float local_val[kMaxLocal];
    int local_idx[kMaxLocal];

#pragma unroll
    for (int j = 0; j < kMaxLocal; ++j) {
        const int expert = lane + j * kWarpSize;
        if (expert < num_experts) {
            local_val[j] = to_float(row[expert]);
            local_idx[j] = expert;
        } else {
            local_val[j] = -INFINITY;
            local_idx[j] = -1;
        }
    }

    // Softmax max reduction.
    float thread_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < kMaxLocal; ++j) {
        thread_max = fmaxf(thread_max, local_val[j]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFFu, thread_max, offset));
    }
    thread_max = __shfl_sync(0xFFFFFFFFu, thread_max, 0);

    // Softmax sum-of-exp reduction; local_val becomes exp(logit - max).
    float thread_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxLocal; ++j) {
        if (local_idx[j] >= 0) {
            local_val[j] = expf(local_val[j] - thread_max);
            thread_sum += local_val[j];
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFFu, thread_sum, offset);
    }
    thread_sum = __shfl_sync(0xFFFFFFFFu, thread_sum, 0);

    // Normalize to softmax probabilities.
#pragma unroll
    for (int j = 0; j < kMaxLocal; ++j) {
        if (local_idx[j] >= 0) {
            local_val[j] /= thread_sum;
        }
    }

    // Iterative top-k: each round, every lane proposes its own best
    // remaining expert, warp_argmax picks the global winner, and the
    // (unique) owning lane invalidates that slot before the next round.
    float weight_sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        float best_val = -INFINITY;
        int best_idx = -1;
        int best_slot = -1;
#pragma unroll
        for (int j = 0; j < kMaxLocal; ++j) {
            if (local_idx[j] >= 0 && local_val[j] > best_val) {
                best_val = local_val[j];
                best_idx = local_idx[j];
                best_slot = j;
            }
        }

        float sel_val = best_val;
        int sel_idx = best_idx;
        warp_argmax(sel_val, sel_idx);

        if (best_slot >= 0 && best_idx == sel_idx) {
            local_idx[best_slot] = -1;
        }

        if (lane == 0) {
            weight_sum += sel_val;
            topk_weights[token * k + i] = sel_val;
            topk_indices[token * k + i] = sel_idx;
        }
    }

    if (renormalize && lane == 0) {
        for (int64_t i = 0; i < k; ++i) {
            topk_weights[token * k + i] /= weight_sum;
        }
    }
}

template <typename T>
cudaError_t dispatch(
    const T* logits, float* topk_weights, int64_t* topk_indices, int64_t num_tokens,
    int64_t num_experts, int64_t k, bool renormalize, cudaStream_t stream) {
    const dim3 block(kBlockSize);
    const dim3 grid(static_cast<unsigned int>((num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock));
    moe_router_kernel<T><<<grid, block, 0, stream>>>(
        logits, topk_weights, topk_indices, num_tokens, num_experts, k, renormalize);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t launch_moe_router_fwd(
    const void* logits,
    float* topk_weights,
    int64_t* topk_indices,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int32_t dtype,
    int32_t renormalize,
    cudaStream_t stream) {
    if (num_tokens <= 0) {
        return cudaSuccess;
    }
    if (num_experts > kMaxLocal * kWarpSize) {
        return cudaErrorInvalidValue;
    }

    const bool renorm = renormalize != 0;

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(logits), topk_weights, topk_indices, num_tokens,
                num_experts, k, renorm, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(logits), topk_weights, topk_indices, num_tokens,
                num_experts, k, renorm, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(logits), topk_weights, topk_indices, num_tokens,
                num_experts, k, renorm, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
