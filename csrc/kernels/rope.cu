#include "../includes/common.cuh"
#include "../includes/rope.h"

// -----------------------------------------------------------------------
// Kernel 3 — Fused Rotary Position Embedding (half-split convention).
//
// v1 (scalar, kept below as `rope_kernel_scalar`): one block per (batch,
// seq, head) row of q or k, grid-stride loop over the half-head-dim.
// benchmarks/rope_bench.py showed this already clearing the 2x speedup
// target by a wide margin (3.3-9.1x) but, like SwiGLU's v1, fp16/bf16
// bandwidth trailed fp32 on identical shapes (e.g. batch=16/seq=1024:
// fp32 86.1% of peak vs. fp16 67.8%/bf16 67.4%) — the same
// transaction-width signature that vectorization fixed for Kernel 2.
//
// v2 (`rope_kernel_vec` below, now the default dispatch path): each
// thread processes VecWidth consecutive rotation *pairs* at once — a
// float4 load of 4 consecutive x1 values and a separate float4 load of
// the 4 correspondingly-consecutive x2 values (fp32), or the analogous
// uint4-packed 8-wide half2/bf16x2 load (fp16/bf16). x1 and x2 live
// `half_dim` elements apart in memory (not adjacent), so this vectorizes
// along the within-half contiguous axis rather than pairing x1/x2 into
// one transaction the way SwiGLU paired gate/up.
//
// Dispatch requires `half_dim * sizeof(T)` to be a multiple of 16 bytes
// (guarantees every row's x1 pointer, x2 pointer i.e. `src + half_dim`,
// and the cos/sin row all stay 16-byte-aligned given 16-byte-aligned base
// pointers — this is a stronger, row-independent condition, unlike a
// flat-array vectorization where only the base pointer matters) *and*
// `half_dim % VecWidth == 0` (no remainder to handle — real head_dims are
// always powers of two >= 16, so this holds for every case exercised by
// tests/benchmarks). Falls back to the scalar kernel entirely otherwise.
// -----------------------------------------------------------------------

namespace {

constexpr int kBlockSize = 256;

// The vectorized kernels' grid-stride loop runs `half_dim / VecWidth`
// iterations per row — typically 8-16 for real head_dims (64/128) — far
// fewer than kBlockSize=256. A first vectorized attempt kept blockDim.x
// fixed at 256 and *regressed* bandwidth on every shape (see
// benchmarks/rope_bench.py history in project_plan.md): with only 8-16 of
// 256 launched threads ever executing the loop body, the vast majority of
// each block's threads did nothing, cutting effective occupancy. A 32
// (one warp) block size — still covering every realistic half_dim_vec in
// a single iteration via the grid-stride loop, just without the wasted
// 224+ idle threads/block — recovered the win; see rope_bench.py results.
constexpr int kVecBlockSize = 32;

__device__ __forceinline__ int64_t row_seq_idx(int64_t row, int64_t heads, int64_t seq_len) {
    return (row / heads) % seq_len;
}

// ---- scalar (fallback) --------------------------------------------------

template <typename T>
__global__ void rope_kernel_scalar(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int64_t seq_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t rows_q) {
    const int64_t row = blockIdx.x;
    const int64_t half_dim = head_dim / 2;

    const T *src;
    T *dst;
    int64_t seq_idx;
    if (row < rows_q) {
        seq_idx = row_seq_idx(row, n_q_heads, seq_len);
        src = q + row * head_dim;
        dst = q_out + row * head_dim;
    } else {
        const int64_t k_row = row - rows_q;
        seq_idx = row_seq_idx(k_row, n_kv_heads, seq_len);
        src = k + k_row * head_dim;
        dst = k_out + k_row * head_dim;
    }

    const float* cos_row = cos_table + seq_idx * half_dim;
    const float* sin_row = sin_table + seq_idx * half_dim;

    for (int64_t i = threadIdx.x; i < half_dim; i += kBlockSize) {
        const float x1 = to_float(src[i]);
        const float x2 = to_float(src[i + half_dim]);
        const float c = cos_row[i];
        const float s = sin_row[i];
        dst[i] = from_float<T>(x1 * c - x2 * s);
        dst[i + half_dim] = from_float<T>(x2 * c + x1 * s);
    }
}

// ---- fp32: float4, 4 pairs/iteration ------------------------------------

__global__ void rope_kernel_vec4_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int64_t seq_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t rows_q,
    int64_t half_dim_vec) {
    const int64_t row = blockIdx.x;
    const int64_t half_dim = half_dim_vec * 4;

    const float *src;
    float *dst;
    int64_t seq_idx;
    if (row < rows_q) {
        seq_idx = row_seq_idx(row, n_q_heads, seq_len);
        src = q + row * head_dim;
        dst = q_out + row * head_dim;
    } else {
        const int64_t k_row = row - rows_q;
        seq_idx = row_seq_idx(k_row, n_kv_heads, seq_len);
        src = k + k_row * head_dim;
        dst = k_out + k_row * head_dim;
    }

    const float4* x1v = reinterpret_cast<const float4*>(src);
    const float4* x2v = reinterpret_cast<const float4*>(src + half_dim);
    float4* o1v = reinterpret_cast<float4*>(dst);
    float4* o2v = reinterpret_cast<float4*>(dst + half_dim);
    const float4* cv = reinterpret_cast<const float4*>(cos_table + seq_idx * half_dim);
    const float4* sv = reinterpret_cast<const float4*>(sin_table + seq_idx * half_dim);

    for (int64_t iv = threadIdx.x; iv < half_dim_vec; iv += kVecBlockSize) {
        const float4 x1 = x1v[iv];
        const float4 x2 = x2v[iv];
        const float4 c = cv[iv];
        const float4 s = sv[iv];
        float4 o1, o2;
        o1.x = x1.x * c.x - x2.x * s.x;
        o1.y = x1.y * c.y - x2.y * s.y;
        o1.z = x1.z * c.z - x2.z * s.z;
        o1.w = x1.w * c.w - x2.w * s.w;
        o2.x = x2.x * c.x + x1.x * s.x;
        o2.y = x2.y * c.y + x1.y * s.y;
        o2.z = x2.z * c.z + x1.z * s.z;
        o2.w = x2.w * c.w + x1.w * s.w;
        o1v[iv] = o1;
        o2v[iv] = o2;
    }
}

// ---- fp16 / bf16: uint4-packed, 8 pairs/iteration -----------------------

template <typename T, typename T2>
struct PackedOps;

template <>
struct PackedOps<__half, __half2> {
    __device__ __forceinline__ static float low(const __half2& p) { return __half2float(__low2half(p)); }
    __device__ __forceinline__ static float high(const __half2& p) { return __half2float(__high2half(p)); }
    __device__ __forceinline__ static __half2 pack(float lo, float hi) {
        return __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    }
};

template <>
struct PackedOps<__nv_bfloat16, __nv_bfloat162> {
    __device__ __forceinline__ static float low(const __nv_bfloat162& p) {
        return __bfloat162float(__low2bfloat16(p));
    }
    __device__ __forceinline__ static float high(const __nv_bfloat162& p) {
        return __bfloat162float(__high2bfloat16(p));
    }
    __device__ __forceinline__ static __nv_bfloat162 pack(float lo, float hi) {
        return __halves2bfloat162(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
    }
};

template <typename T, typename T2>
__global__ void rope_kernel_vec8_half(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int64_t seq_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t rows_q,
    int64_t half_dim_vec) {
    using Ops = PackedOps<T, T2>;
    const int64_t row = blockIdx.x;
    const int64_t half_dim = half_dim_vec * 8;

    const T *src;
    T *dst;
    int64_t seq_idx;
    if (row < rows_q) {
        seq_idx = row_seq_idx(row, n_q_heads, seq_len);
        src = q + row * head_dim;
        dst = q_out + row * head_dim;
    } else {
        const int64_t k_row = row - rows_q;
        seq_idx = row_seq_idx(k_row, n_kv_heads, seq_len);
        src = k + k_row * head_dim;
        dst = k_out + k_row * head_dim;
    }

    const uint4* x1v = reinterpret_cast<const uint4*>(src);
    const uint4* x2v = reinterpret_cast<const uint4*>(src + half_dim);
    uint4* o1v = reinterpret_cast<uint4*>(dst);
    uint4* o2v = reinterpret_cast<uint4*>(dst + half_dim);
    // cos/sin are always float32: 8 elements = two float4 = one 32-byte
    // span. Read as two consecutive float4 to stay within existing types.
    const float4* cv = reinterpret_cast<const float4*>(cos_table + seq_idx * half_dim);
    const float4* sv = reinterpret_cast<const float4*>(sin_table + seq_idx * half_dim);

    for (int64_t iv = threadIdx.x; iv < half_dim_vec; iv += kVecBlockSize) {
        const uint4 x1u = x1v[iv];
        const uint4 x2u = x2v[iv];
        const T2* x1p = reinterpret_cast<const T2*>(&x1u);
        const T2* x2p = reinterpret_cast<const T2*>(&x2u);
        const float4 c0 = cv[iv * 2];
        const float4 c1 = cv[iv * 2 + 1];
        const float4 s0 = sv[iv * 2];
        const float4 s1 = sv[iv * 2 + 1];
        const float cf[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float sf[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};

        uint4 o1u, o2u;
        T2* o1p = reinterpret_cast<T2*>(&o1u);
        T2* o2p = reinterpret_cast<T2*>(&o2u);
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const float x1_lo = Ops::low(x1p[lane]);
            const float x1_hi = Ops::high(x1p[lane]);
            const float x2_lo = Ops::low(x2p[lane]);
            const float x2_hi = Ops::high(x2p[lane]);
            const float c_lo = cf[lane * 2];
            const float c_hi = cf[lane * 2 + 1];
            const float s_lo = sf[lane * 2];
            const float s_hi = sf[lane * 2 + 1];
            o1p[lane] = Ops::pack(x1_lo * c_lo - x2_lo * s_lo, x1_hi * c_hi - x2_hi * s_hi);
            o2p[lane] = Ops::pack(x2_lo * c_lo + x1_lo * s_lo, x2_hi * c_hi + x1_hi * s_hi);
        }
        o1v[iv] = o1u;
        o2v[iv] = o2u;
    }
}

// ---- dispatch -------------------------------------------------------

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0; }

}  // namespace

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
    cudaStream_t stream) {
    if (batch <= 0 || seq_len <= 0 || head_dim <= 0) {
        return cudaSuccess;
    }
    const int64_t rows_q = batch * seq_len * n_q_heads;
    const int64_t rows_k = batch * seq_len * n_kv_heads;
    const int64_t total_rows = rows_q + rows_k;
    if (total_rows <= 0) {
        return cudaSuccess;
    }

    const int64_t half_dim = head_dim / 2;
    const float* cos_f = static_cast<const float*>(cos_table);
    const float* sin_f = static_cast<const float*>(sin_table);
    const dim3 grid(static_cast<unsigned int>(total_rows));
    const dim3 block(kBlockSize);
    const dim3 vec_block(kVecBlockSize);
    const bool base_aligned =
        is_aligned16(q) && is_aligned16(k) && is_aligned16(q_out) && is_aligned16(k_out) &&
        is_aligned16(cos_f) && is_aligned16(sin_f);

    switch (static_cast<KernelDType>(dtype)) {
        case KernelDType::F32: {
            const bool vecable = base_aligned && (half_dim % 4 == 0) && ((half_dim * 4) % 16 == 0);
            if (vecable) {
                rope_kernel_vec4_f32<<<grid, vec_block, 0, stream>>>(
                    static_cast<const float*>(q), static_cast<const float*>(k), cos_f, sin_f,
                    static_cast<float*>(q_out), static_cast<float*>(k_out), seq_len, n_q_heads,
                    n_kv_heads, head_dim, rows_q, half_dim / 4);
            } else {
                rope_kernel_scalar<float><<<grid, block, 0, stream>>>(
                    static_cast<const float*>(q), static_cast<const float*>(k), cos_f, sin_f,
                    static_cast<float*>(q_out), static_cast<float*>(k_out), seq_len, n_q_heads,
                    n_kv_heads, head_dim, rows_q);
            }
            break;
        }
        case KernelDType::F16: {
            const bool vecable = base_aligned && (half_dim % 8 == 0) && ((half_dim * 2) % 16 == 0);
            if (vecable) {
                rope_kernel_vec8_half<__half, __half2><<<grid, vec_block, 0, stream>>>(
                    static_cast<const __half*>(q), static_cast<const __half*>(k), cos_f, sin_f,
                    static_cast<__half*>(q_out), static_cast<__half*>(k_out), seq_len, n_q_heads,
                    n_kv_heads, head_dim, rows_q, half_dim / 8);
            } else {
                rope_kernel_scalar<__half><<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(q), static_cast<const __half*>(k), cos_f, sin_f,
                    static_cast<__half*>(q_out), static_cast<__half*>(k_out), seq_len, n_q_heads,
                    n_kv_heads, head_dim, rows_q);
            }
            break;
        }
        case KernelDType::BF16: {
            const bool vecable = base_aligned && (half_dim % 8 == 0) && ((half_dim * 2) % 16 == 0);
            if (vecable) {
                rope_kernel_vec8_half<__nv_bfloat16, __nv_bfloat162><<<grid, vec_block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), cos_f,
                    sin_f, static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
                    seq_len, n_q_heads, n_kv_heads, head_dim, rows_q, half_dim / 8);
            } else {
                rope_kernel_scalar<__nv_bfloat16><<<grid, block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k), cos_f,
                    sin_f, static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
                    seq_len, n_q_heads, n_kv_heads, head_dim, rows_q);
            }
            break;
        }
        default:
            return cudaErrorInvalidValue;
    }

    CUDA_CHECK_RETURN(cudaGetLastError());
    return cudaSuccess;
}
