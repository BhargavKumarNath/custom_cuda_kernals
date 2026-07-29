"""Structured metadata for all 12 kernels, driving every CLI command.

Deliberately data, not a parser over `project_plan.md`: the prose spec is
the source of truth for humans, this module is the source of truth for
the CLI. Kept free of `torch`/kernel imports at module scope so commands
that don't touch CUDA (`list`, `info`, `help`) stay fast to start —
`profile_builder` callables below do their imports lazily, inside the
function body, only when `profile` actually needs them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

__all__ = ["KernelSpec", "ProfileTarget", "REGISTRY", "get_kernel", "list_kernels"]


@dataclasses.dataclass(frozen=True)
class ProfileTarget:
    """A ready-to-call representative workload for `profile`/`compare`."""

    fn: Callable[[], object]
    description: str


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    id: str
    number: int
    phase: str
    title: str
    purpose: str
    dtypes: tuple[str, ...]
    memory_bound: str
    optimization_techniques: tuple[str, ...]
    success_criteria: str
    baseline_module: str
    kernel_module: str
    kernel_functions: tuple[str, ...]
    test_files: tuple[str, ...]
    bench_script: str
    bench_result_csvs: tuple[str, ...]
    plot_script: str
    profile_builder: Callable[[str], ProfileTarget]


# ---------------------------------------------------------------------------
# Profile-target builders — one small representative call per kernel, used
# by `profile` (torch.profiler trace) and `compare` (quick eager-vs-kernel
# spot check). Each does its own lazy imports.
# ---------------------------------------------------------------------------


def _profile_rmsnorm_residual(device: str) -> ProfileTarget:
    from baselines.rmsnorm_residual import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.rmsnorm_residual import rmsnorm_residual

    case = STANDARD_CASES[0]
    x, residual, weight = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: rmsnorm_residual(x, residual, weight), f"rmsnorm_residual({case.name})"
    )


def _profile_swiglu(device: str) -> ProfileTarget:
    from baselines.swiglu import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.swiglu import swiglu

    case = STANDARD_CASES[0]
    gate, up = make_inputs(case, device=device)
    return ProfileTarget(lambda: swiglu(gate, up), f"swiglu({case.name})")


def _profile_rope(device: str) -> ProfileTarget:
    from baselines.rope import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.rope import rope

    case = STANDARD_CASES[0]
    q, k, cos, sin = make_inputs(case, device=device)
    return ProfileTarget(lambda: rope(q, k, cos, sin), f"rope({case.name})")


def _profile_linear_cross_entropy(device: str) -> ProfileTarget:
    from baselines.linear_cross_entropy import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy

    case = STANDARD_CASES[0]
    hidden, weight, targets = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: linear_cross_entropy(hidden, weight, targets), f"linear_cross_entropy({case.name})"
    )


def _profile_matmul_add_bias(device: str) -> ProfileTarget:
    from baselines.matmul_add_bias import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.matmul_add_bias import matmul_add_bias

    case = STANDARD_CASES[0]
    x, weight, bias = make_inputs(case, device=device)
    return ProfileTarget(lambda: matmul_add_bias(x, weight, bias), f"matmul_add_bias({case.name})")


def _profile_moe_router(device: str) -> ProfileTarget:
    from baselines.moe_router import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.moe_router import moe_router

    case = STANDARD_CASES[0]
    logits = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: moe_router(logits, case.k, case.renormalize), f"moe_router({case.name})"
    )


def _profile_token_permute(device: str) -> ProfileTarget:
    from baselines.token_permute import GATHER_STANDARD_CASES, make_gather_inputs
    from custom_cuda.kernels.token_permute import token_gather

    case = GATHER_STANDARD_CASES[0]
    src, indices = make_gather_inputs(case, device=device)
    return ProfileTarget(lambda: token_gather(src, indices), f"token_gather({case.name})")


def _profile_cosine_topk(device: str) -> ProfileTarget:
    from baselines.cosine_topk import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.cosine_topk import cosine_topk

    case = STANDARD_CASES[0]
    queries, candidates = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: cosine_topk(queries, candidates, case.k), f"cosine_topk({case.name})"
    )


def _profile_pairwise_distance(device: str) -> ProfileTarget:
    from baselines.pairwise_distance import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.pairwise_distance import pairwise_distance_sq

    case = STANDARD_CASES[0]
    a, b = make_inputs(case, device=device)
    return ProfileTarget(lambda: pairwise_distance_sq(a, b), f"pairwise_distance_sq({case.name})")


def _profile_graph_message_passing(device: str) -> ProfileTarget:
    from baselines.graph_message_passing import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.graph_message_passing import spatiotemporal_message_passing

    case = STANDARD_CASES[0]
    inputs = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: spatiotemporal_message_passing(*inputs),
        f"spatiotemporal_message_passing({case.name})",
    )


def _profile_viterbi(device: str) -> ProfileTarget:
    from baselines.viterbi import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.viterbi import viterbi_decode

    case = STANDARD_CASES[0]
    inputs = make_inputs(case, device=device)
    return ProfileTarget(lambda: viterbi_decode(*inputs), f"viterbi_decode({case.name})")


def _profile_fp8_quant(device: str) -> ProfileTarget:
    from baselines.fp8_quant import STANDARD_CASES, make_inputs
    from custom_cuda.kernels.fp8_quant import fp8_quant

    case = STANDARD_CASES[0]
    x = make_inputs(case, device=device)
    return ProfileTarget(
        lambda: fp8_quant(x, case.fp8_format, case.granularity), f"fp8_quant({case.name})"
    )


# ---------------------------------------------------------------------------
# The registry itself.
# ---------------------------------------------------------------------------

_KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        id="rmsnorm_residual",
        number=1,
        phase="Phase 1: Core Transformer Operations",
        title="Fused RMSNorm + Residual Addition",
        purpose="y = RMSNorm(x + residual) * gamma, fusing the residual add, sum-of-squares "
        "reduction, and scale into a single pass; optionally emits the pre-norm residual sum.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Bandwidth-bound (negligible FLOP count) — a naive implementation "
        "reads+writes the residual add, then reads+writes the norm, doubling traffic.",
        optimization_techniques=(
            "single-pass block reduction (warp-shuffle + shared-memory cross-warp)",
            "float4/half2 vectorized loads and stores",
            "fused residual-write and normalized-write from one register tile",
            "one kernel launch instead of two",
        ),
        success_criteria=">=2.5x speedup vs. unfused eager; >=80% of peak memory bandwidth; "
        "eliminates one intermediate tensor allocation per call.",
        baseline_module="baselines.rmsnorm_residual",
        kernel_module="custom_cuda.kernels.rmsnorm_residual",
        kernel_functions=("rmsnorm_residual",),
        test_files=("tests/test_rmsnorm_residual.py", "tests/test_rmsnorm_residual_kernel.py"),
        bench_script="benchmarks/rmsnorm_residual_bench.py",
        bench_result_csvs=("benchmarks/rmsnorm_residual_results.csv",),
        plot_script="scripts/plot_rmsnorm_residual.py",
        profile_builder=_profile_rmsnorm_residual,
    ),
    KernelSpec(
        id="swiglu",
        number=2,
        phase="Phase 1: Core Transformer Operations",
        title="Fused SwiGLU Gated Activation",
        purpose="SwiGLU(gate, up) = SiLU(gate) * up, fusing the activation and elementwise "
        "multiply into a single kernel over the (large) FFN intermediate tensor.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Bandwidth-roofline-bound (trivial FLOP/byte ratio); an unfused "
        "implementation is two elementwise kernels, each paying a full read+write.",
        optimization_techniques=(
            "single-pass elementwise fusion",
            "float4/half2 vectorized global memory access",
            "register-resident intermediate (no scratch tensor)",
        ),
        success_criteria=">=1.8x speedup vs. two-kernel eager; zero intermediate allocations; "
        ">=90% of peak memory bandwidth.",
        baseline_module="baselines.swiglu",
        kernel_module="custom_cuda.kernels.swiglu",
        kernel_functions=("swiglu",),
        test_files=("tests/test_swiglu.py", "tests/test_swiglu_kernel.py"),
        bench_script="benchmarks/swiglu_bench.py",
        bench_result_csvs=("benchmarks/swiglu_results.csv",),
        plot_script="scripts/plot_swiglu.py",
        profile_builder=_profile_swiglu,
    ),
    KernelSpec(
        id="rope",
        number=3,
        phase="Phase 1: Core Transformer Operations",
        title="Fused Rotary Position Embedding (RoPE)",
        purpose="Rotate Q and K projections together in one kernel launch given precomputed "
        "sin/cos tables, supporting interleaved-pairs and half-split layouts plus GQA.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Bandwidth-bound; naive view/slice/concat implementations issue several "
        "small launches with non-coalesced access through the reshape/transpose views.",
        optimization_techniques=(
            "single fused kernel rotating Q and K together",
            "sin/cos staged through shared memory / coalesced broadcast loads",
            "float2/float4 vectorized access matched to RoPE layout",
            "register-level rotation (no temp buffer)",
        ),
        success_criteria="one launch replaces >=4 elementwise ops; >=2x speedup vs. eager; "
        "max abs error <1e-3 in fp16 vs. fp64 reference; correct under GQA head-count mismatch.",
        baseline_module="baselines.rope",
        kernel_module="custom_cuda.kernels.rope",
        kernel_functions=("rope",),
        test_files=("tests/test_rope.py", "tests/test_rope_kernel.py"),
        bench_script="benchmarks/rope_bench.py",
        bench_result_csvs=("benchmarks/rope_results.csv",),
        plot_script="scripts/plot_rope.py",
        profile_builder=_profile_rope,
    ),
    KernelSpec(
        id="linear_cross_entropy",
        number=4,
        phase="Phase 2: Compute & Loss Optimizations",
        title="Fused Linear Cross Entropy Loss",
        purpose="Fuse the vocab projection (logits = h @ W^T) with cross-entropy using a "
        "chunked, online-softmax algorithm that never materializes full [tokens, vocab] logits.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="VRAM-bound — with 128K+ vocabularies the full logits tensor "
        "(e.g. 8192 tokens x 128,256 vocab x 4B ~= 4.2GB fp32) is often the largest "
        "transient allocation.",
        optimization_techniques=(
            "vocab-dimension tiling with running max/sum (online softmax)",
            "chunked matmul fused with the reduction (no tile's logits outlive their contribution)",
            "fused backward computing the input gradient in the same chunked structure",
        ),
        success_criteria=">=4x reduction in peak VRAM for the loss step vs. materializing full "
        "logits; throughput within 90% of a raw matmul-bound reference.",
        baseline_module="baselines.linear_cross_entropy",
        kernel_module="custom_cuda.kernels.linear_cross_entropy",
        kernel_functions=("linear_cross_entropy",),
        test_files=(
            "tests/test_linear_cross_entropy.py", "tests/test_linear_cross_entropy_kernel.py",
        ),
        bench_script="benchmarks/linear_cross_entropy_bench.py",
        bench_result_csvs=(
            "benchmarks/linear_cross_entropy_results.csv",
            "benchmarks/linear_cross_entropy_chunk_sweep.csv",
            "benchmarks/linear_cross_entropy_memory.csv",
        ),
        plot_script="scripts/plot_linear_cross_entropy.py",
        profile_builder=_profile_linear_cross_entropy,
    ),
    KernelSpec(
        id="matmul_add_bias",
        number=5,
        phase="Phase 2: Compute & Loss Optimizations",
        title="Fused MatMul Add Bias",
        purpose="y = x @ W^T + b, with the bias-add fused directly into the GEMM epilogue "
        "instead of as a separate elementwise pass.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Compute-bound at realistic LLM matrix shapes (register-blocked tiled "
        "GEMM); the library's reference hand-written GEMM used to benchmark epilogue fusion.",
        optimization_techniques=(
            "shared-memory tiled GEMM (block tiling over M/N/K)",
            "register blocking/accumulation",
            "epilogue fusion writing accumulator + bias[col] directly from registers",
            "float4 vectorized stores",
        ),
        success_criteria="bias-add overhead reduced to statistical noise vs. a separate "
        "elementwise kernel; >=15% speedup over cuBLAS GEMM + discrete bias-add for LLM shapes.",
        baseline_module="baselines.matmul_add_bias",
        kernel_module="custom_cuda.kernels.matmul_add_bias",
        kernel_functions=("matmul_add_bias",),
        test_files=("tests/test_matmul_add_bias.py", "tests/test_matmul_add_bias_kernel.py"),
        bench_script="benchmarks/matmul_add_bias_bench.py",
        bench_result_csvs=("benchmarks/matmul_add_bias_results.csv",),
        plot_script="scripts/plot_matmul_add_bias.py",
        profile_builder=_profile_matmul_add_bias,
    ),
    KernelSpec(
        id="moe_router",
        number=6,
        phase="Phase 3: MoE Routing & Permutation",
        title="MoE Top-K Router",
        purpose="Given router logits [tokens, experts], compute the gating distribution "
        "(softmax/sigmoid) and select the top-k experts per token with normalized gate weights.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Latency-bound / launch-bound — sits on the MoE critical path; naive "
        "topk(softmax(logits)) is multiple launches and often forces a host sync.",
        optimization_techniques=(
            "single fused softmax + top-k kernel",
            "one block (or warp, for small num_experts) per token",
            "warp-level iterative warp-shuffle max-reduction selection network",
        ),
        success_criteria=">=3x speedup vs. softmax+topk eager chain; zero host-device sync "
        "in routing; exact match to torch.topk for k in {1,2,4,8}.",
        baseline_module="baselines.moe_router",
        kernel_module="custom_cuda.kernels.moe_router",
        kernel_functions=("moe_router",),
        test_files=("tests/test_moe_router.py", "tests/test_moe_router_kernel.py"),
        bench_script="benchmarks/moe_router_bench.py",
        bench_result_csvs=("benchmarks/moe_router_results.csv",),
        plot_script="scripts/plot_moe_router.py",
        profile_builder=_profile_moe_router,
    ),
    KernelSpec(
        id="token_permute",
        number=7,
        phase="Phase 3: MoE Routing & Permutation",
        title="Token Scatter/Gather (Permute / Unpermute)",
        purpose="Reorder token embeddings into contiguous per-expert buffers per router "
        "assignment (permute/gather) and invert it after expert compute, fusing the "
        "weighted-gate combine into the unpermute path.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Pure bandwidth-bound gather/scatter — frequently the actual MoE-layer "
        "bottleneck at inference batch sizes, more so than the expert FFN matmuls themselves.",
        optimization_techniques=(
            "coalesced row-gather driven by a precomputed permutation index buffer",
            "float4 vectorized per-row copy",
            "fused weighted-combine on the unpermute path",
            "shared-memory staging for narrow hidden dimensions",
        ),
        success_criteria=">=2x speedup vs. index_select/index_add pair; >=75% of peak "
        "memory bandwidth; no intermediate buffer beyond the permuted output.",
        baseline_module="baselines.token_permute",
        kernel_module="custom_cuda.kernels.token_permute",
        kernel_functions=("token_gather", "token_combine"),
        test_files=("tests/test_token_permute.py", "tests/test_token_permute_kernel.py"),
        bench_script="benchmarks/token_permute_bench.py",
        bench_result_csvs=(
            "benchmarks/token_permute_gather_results.csv",
            "benchmarks/token_permute_combine_results.csv",
        ),
        plot_script="scripts/plot_token_permute.py",
        profile_builder=_profile_token_permute,
    ),
    KernelSpec(
        id="cosine_topk",
        number=8,
        phase="Phase 4: RAG & Vector Search Accelerators",
        title="Fused Cosine Similarity + Top-K Selection",
        purpose="Compute cosine similarity between queries and a candidate pool and return "
        "the top-k most similar indices/scores in one kernel, without materializing the full "
        "[queries, candidates] score matrix.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Bandwidth-bound streaming top-k; naive normalize->matmul->topk "
        "materializes the full similarity matrix even though only the top few entries matter.",
        optimization_techniques=(
            "L2 normalization fused into the dot-product accumulation loop",
            "two-kernel partition + merge design (grid scales with candidate pool, not query "
            "count)",
            "per-partition on-chip top-k via sorted-array insertion + warp-shuffle argmax merge",
        ),
        success_criteria=">=3x speedup vs. normalize->matmul->topk baseline up to ~1M "
        "candidates; exact match to a brute-force reference; no full score-matrix materialization.",
        baseline_module="baselines.cosine_topk",
        kernel_module="custom_cuda.kernels.cosine_topk",
        kernel_functions=("cosine_topk",),
        test_files=("tests/test_cosine_topk.py", "tests/test_cosine_topk_kernel.py"),
        bench_script="benchmarks/cosine_topk_bench.py",
        bench_result_csvs=("benchmarks/cosine_topk_results.csv",),
        plot_script="scripts/plot_cosine_topk.py",
        profile_builder=_profile_cosine_topk,
    ),
    KernelSpec(
        id="pairwise_distance",
        number=9,
        phase="Phase 4: RAG & Vector Search Accelerators",
        title="Block Pairwise Distance Matrix Computation",
        purpose="Compute the full squared-Euclidean pairwise distance matrix D[i,j] between "
        "two vector sets using a shared-memory tiled block algorithm (register-blocked GEMM "
        "structure), with an explicit clamp guarding cancellation.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Compute-bound tiled-GEMM-style kernel (reuses Kernel 5's structure); "
        "torch.cdist doesn't exploit shared-memory tile reuse across the (i,j) grid.",
        optimization_techniques=(
            "tiled ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b with shared-memory-staged A/B tiles",
            "register-blocked dot-product accumulation",
            "explicit max(.,0) cancellation-guard clamp",
        ),
        success_criteria=">=2x speedup vs. torch.cdist up to 16Kx16K; >=70% of peak memory "
        "bandwidth; no negative distances after the clamp.",
        baseline_module="baselines.pairwise_distance",
        kernel_module="custom_cuda.kernels.pairwise_distance",
        kernel_functions=("pairwise_distance_sq",),
        test_files=("tests/test_pairwise_distance.py", "tests/test_pairwise_distance_kernel.py"),
        bench_script="benchmarks/pairwise_distance_bench.py",
        bench_result_csvs=("benchmarks/pairwise_distance_results.csv",),
        plot_script="scripts/plot_pairwise_distance.py",
        profile_builder=_profile_pairwise_distance,
    ),
    KernelSpec(
        id="graph_message_passing",
        number=10,
        phase="Phase 5: Graph & Sequence Algorithms",
        title="Spatiotemporal Graph Message Passing",
        purpose="One step of neighborhood aggregation over spatial (within-timestep) and "
        "temporal (previous-timestep) edge sets, aggregating weighted neighbor features into "
        "updated node embeddings.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Irregular, memory-bound gather with poor locality; scatter_add-based "
        "implementations incur atomic contention on high-degree nodes.",
        optimization_techniques=(
            "CSR-based neighbor iteration (sorted by destination node) with warp-per-node "
            "parallelism",
            "feature-dimension split across warp lanes — each output element owned by exactly "
            "one lane, eliminating atomics entirely rather than merely reducing them",
            "register-blocked feature dimension (4 strided stripes per lane)",
        ),
        success_criteria=">=2x speedup vs. PyTorch Geometric's scatter_add-based message "
        "passing on 100K+-node graphs, avg degree 10-50; exact match to a scatter-based reference.",
        baseline_module="baselines.graph_message_passing",
        kernel_module="custom_cuda.kernels.graph_message_passing",
        kernel_functions=("spatiotemporal_message_passing",),
        test_files=(
            "tests/test_graph_message_passing.py", "tests/test_graph_message_passing_kernel.py",
        ),
        bench_script="benchmarks/graph_message_passing_bench.py",
        bench_result_csvs=("benchmarks/graph_message_passing_results.csv",),
        plot_script="scripts/plot_graph_message_passing.py",
        profile_builder=_profile_graph_message_passing,
    ),
    KernelSpec(
        id="viterbi",
        number=11,
        phase="Phase 5: Graph & Sequence Algorithms",
        title="Parallel Viterbi Algorithm (Hidden Markov Models)",
        purpose="Batched Viterbi decoding of a single HMM shared across the batch — most-"
        "likely hidden state sequence via a persistent kernel that loops over all timesteps "
        "internally instead of one launch per timestep.",
        dtypes=("float32", "float16", "bfloat16"),
        memory_bound="Launch-latency-bound in the naive per-timestep-loop form; the fused "
        "kernel is compute-light but eliminates essentially all Python/launch overhead.",
        optimization_techniques=(
            "single persistent-kernel launch looping over all timesteps internally",
            "one block per batch item; transition matrix held resident in shared memory",
            "warp-shuffle max-reduction for the final argmax-over-states step seeding the "
            "backtrack",
        ),
        success_criteria=">=5x speedup vs. per-timestep Python/PyTorch loop for sequence "
        "lengths >=512; exact match to a from-scratch fp64 Viterbi DP; O(1) kernel launches.",
        baseline_module="baselines.viterbi",
        kernel_module="custom_cuda.kernels.viterbi",
        kernel_functions=("viterbi_decode",),
        test_files=("tests/test_viterbi.py", "tests/test_viterbi_kernel.py"),
        bench_script="benchmarks/viterbi_bench.py",
        bench_result_csvs=("benchmarks/viterbi_results.csv",),
        plot_script="scripts/plot_viterbi.py",
        profile_builder=_profile_viterbi,
    ),
    KernelSpec(
        id="fp8_quant",
        number=12,
        phase="Phase 6: Precision & Quantization",
        title="FP8 Dynamic Quantization & Casting",
        purpose="Compute a dynamic per-block (128x128) or per-tensor scale from the live "
        "activation's amax and cast fp32/fp16/bf16 to fp8 (e4m3fn/e5m2) in a fused pass.",
        dtypes=("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"),
        memory_bound="Bandwidth-bound; a naive implementation is a separate amax-reduction "
        "kernel followed by a separate scale-and-cast kernel, doubling traffic over the tensor.",
        optimization_techniques=(
            "block granularity: single fused kernel, one block per 128x128 tile, local "
            "amax reduction immediately followed by scale + vectorized cast+store",
            "tensor granularity: amax-reduce (atomicMax) then scale+cast, two launches",
            "16 fp8 bytes packed into a uint4 for vectorized stores, scalar fallback otherwise",
        ),
        success_criteria=">=2x speedup and >=50% memory-traffic reduction vs. separate "
        "amax + cast kernels; round-trip error bounded by the fp8 format's quantization step.",
        baseline_module="baselines.fp8_quant",
        kernel_module="custom_cuda.kernels.fp8_quant",
        kernel_functions=("fp8_quant",),
        test_files=("tests/test_fp8_quant.py", "tests/test_fp8_quant_kernel.py"),
        bench_script="benchmarks/fp8_quant_bench.py",
        bench_result_csvs=("benchmarks/fp8_quant_results.csv",),
        plot_script="scripts/plot_fp8_quant.py",
        profile_builder=_profile_fp8_quant,
    ),
)

REGISTRY: dict[str, KernelSpec] = {spec.id: spec for spec in _KERNELS}


def list_kernels() -> tuple[KernelSpec, ...]:
    return _KERNELS


def get_kernel(kernel_id: str) -> KernelSpec | None:
    return REGISTRY.get(kernel_id)
