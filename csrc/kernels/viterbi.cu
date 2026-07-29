#include "../includes/common.cuh"
#include "../includes/viterbi.h"

// -----------------------------------------------------------------------
// Kernel 11 — Parallel Viterbi Algorithm. A single persistent-kernel
// launch: one block per batch item, looping over all `seq_len`
// timesteps internally with `__syncthreads()` in place of what a naive
// implementation would do as `seq_len` separate kernel launches
// (project_plan.md Section 3.11's stated bottleneck). `log_trans` is
// staged into shared memory once at kernel start and stays resident for
// every timestep of the recursion.
//
// Parallelization is across states within a block (one thread per
// state `s`); each thread's per-timestep "max over previous state k"
// scan is a simple serial loop over `k` rather than a further
// warp-cooperative reduction — a first-pass design choice, documented
// up front rather than discovered after the fact (the same "document
// the deviation" practice as Kernel 10's feature-vs-neighbor-split
// note): splitting the k-reduction across threads too would need one
// warp per *output* state (`num_states` warps per block), which only
// fits within CUDA's 1024-thread/block limit for `num_states <= 32`,
// not the full edge-case range this kernel is tested against (up to
// 100). Warp-shuffle reduction is used for the one reduction that's
// unconditionally block-wide regardless of `num_states` — the final
// argmax over states that seeds the backtrack.
// -----------------------------------------------------------------------

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;

__device__ __forceinline__ void warp_argmax(float& val, int& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        const int other_idx = __shfl_down_sync(0xFFFFFFFFu, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <typename T>
__global__ void viterbi_kernel(
    const T* __restrict__ log_emission,
    const float* __restrict__ log_trans,
    const float* __restrict__ log_pi,
    int64_t* __restrict__ psi,
    int64_t* __restrict__ best_path,
    float* __restrict__ best_score,
    int64_t seq_len,
    int64_t num_states) {
    const int64_t b = blockIdx.x;
    const int s = threadIdx.x;
    const bool active = s < num_states;

    extern __shared__ char smem_raw[];
    float* s_log_trans = reinterpret_cast<float*>(smem_raw);
    float* delta_a = s_log_trans + num_states * num_states;
    float* delta_b = delta_a + num_states;

    for (int64_t idx = threadIdx.x; idx < num_states * num_states; idx += blockDim.x) {
        s_log_trans[idx] = log_trans[idx];
    }

    const T* emission_b = log_emission + b * seq_len * num_states;
    int64_t* psi_b = psi + b * seq_len * num_states;
    int64_t* path_b = best_path + b * seq_len;

    if (active) {
        delta_a[s] = log_pi[s] + to_float(emission_b[s]);
    }
    __syncthreads();

    float* delta_prev = delta_a;
    float* delta_curr = delta_b;

    for (int64_t t = 1; t < seq_len; ++t) {
        if (active) {
            float best_val = -INFINITY;
            int best_k = 0;
            const float* trans_col_base = s_log_trans + s;
#pragma unroll 4
            for (int64_t k = 0; k < num_states; ++k) {
                const float val = delta_prev[k] + trans_col_base[k * num_states];
                if (val > best_val) {
                    best_val = val;
                    best_k = static_cast<int>(k);
                }
            }
            delta_curr[s] = best_val + to_float(emission_b[t * num_states + s]);
            psi_b[t * num_states + s] = best_k;
        }
        __syncthreads();
        float* tmp = delta_prev;
        delta_prev = delta_curr;
        delta_curr = tmp;
        __syncthreads();
    }

    float val = active ? delta_prev[s] : -INFINITY;
    int idx = active ? s : -1;
    warp_argmax(val, idx);

    __shared__ float warp_best_val[kMaxThreadsPerBlock / kWarpSize];
    __shared__ int warp_best_idx[kMaxThreadsPerBlock / kWarpSize];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_best_val[warp_id] = val;
        warp_best_idx[warp_id] = idx;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_best_val[lane] : -INFINITY;
        int i = (lane < num_warps) ? warp_best_idx[lane] : -1;
        warp_argmax(v, i);
        if (lane == 0) {
            warp_best_val[0] = v;
            warp_best_idx[0] = i;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        best_score[b] = warp_best_val[0];
        path_b[seq_len - 1] = warp_best_idx[0];
        for (int64_t t = seq_len - 2; t >= 0; --t) {
            path_b[t] = psi_b[(t + 1) * num_states + path_b[t + 1]];
        }
    }
}

template <typename T>
cudaError_t dispatch(
    const T* log_emission, const float* log_trans, const float* log_pi, int64_t* psi, int64_t* best_path,
    float* best_score, int64_t batch, int64_t seq_len, int64_t num_states, cudaStream_t stream) {
    const int padded_states = static_cast<int>(((num_states + kWarpSize - 1) / kWarpSize) * kWarpSize);
    if (padded_states > kMaxThreadsPerBlock) {
        return cudaErrorInvalidValue;
    }

    const size_t smem_bytes =
        static_cast<size_t>(num_states) * static_cast<size_t>(num_states) * sizeof(float) +
        2 * static_cast<size_t>(num_states) * sizeof(float);
    if (smem_bytes > 48 * 1024) {
        const cudaError_t attr_err =
            cudaFuncSetAttribute(viterbi_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes));
        if (attr_err != cudaSuccess) {
            return attr_err;
        }
    }

    const dim3 grid(static_cast<unsigned int>(batch));
    const dim3 block(static_cast<unsigned int>(padded_states));
    viterbi_kernel<T><<<grid, block, smem_bytes, stream>>>(
        log_emission, log_trans, log_pi, psi, best_path, best_score, seq_len, num_states);
    return cudaGetLastError();
}

}  // namespace

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
    cudaStream_t stream) {
    if (batch <= 0 || seq_len <= 0 || num_states <= 0) {
        return cudaSuccess;
    }

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32:
            CUDA_CHECK_RETURN(dispatch<float>(
                static_cast<const float*>(log_emission), log_trans, log_pi, psi, best_path, best_score, batch,
                seq_len, num_states, stream));
            break;
        case KernelDType::F16:
            CUDA_CHECK_RETURN(dispatch<__half>(
                static_cast<const __half*>(log_emission), log_trans, log_pi, psi, best_path, best_score, batch,
                seq_len, num_states, stream));
            break;
        case KernelDType::BF16:
            CUDA_CHECK_RETURN(dispatch<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(log_emission), log_trans, log_pi, psi, best_path, best_score,
                batch, seq_len, num_states, stream));
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}
