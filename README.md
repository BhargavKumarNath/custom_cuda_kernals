# Custom CUDA Kernels

**12 hand-written CUDA kernels for LLM, RAG, MoE, and graph workloads - wrapped in Rust and exposed to PyTorch as zero-copy, GPU-resident drop-in ops.**

![Build](https://img.shields.io/badge/build-maturin%20%7C%20passing-2ea44f)
![Rust](https://img.shields.io/badge/rust-1.75%2B-dea584?logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/cuda-11.8%2B-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

This repository is a custom CUDA C++ kernel library developed from the ground up for LLM inference, retrieval systems, Mixture-of-Experts (MoE) models, and graph-based workloads. It implements a range of GPU kernels, including tiled GEMM, RMSNorm, SwiGLU, RoPE, fused cross-entropy loss, MoE routing and token permutation, cosine similarity top-k search, pairwise distance computation, spatiotemporal graph message passing, Viterbi decoding, and FP8 quantization.

Each kernel is integrated directly with PyTorch tensors through a Rust CFFI layer, without relying on additional C wrappers or intermediate memory copies. For CUDA tensors stored contiguously in memory, the library operates directly on the underlying GPU pointers using `data_ptr()`, allowing computations to run directly in VRAM with minimal overhead.

All kernels were evaluated on a real GPU environment using an RTX 4070 Laptop GPU with 256 GB/s peak memory bandwidth. Benchmarking was performed using CUDA events with L2 cache flushing between iterations to ensure more reliable measurements, rather than relying on simple wall-clock timing around cached execution paths.

The project documents both successful optimizations and approaches that did not achieve the expected performance improvements. The complete development process, including initial implementations, profiling results, optimization experiments, and performance analysis, is available in [`project_plan.md`](./project_plan.md).

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Architecture](#system-architecture)
3. [Engineering Principles](#engineering-principles)
4. [Integration Proof: A Real Llama-3-8B Block](#integration-proof-a-real-llama-3-8b-block)
5. [The 12 Kernels](#the-12-kernels)
6. [Developer CLI (`custom_cuda_cli`)](#developer-cli-custom_cuda_cli)
7. [Installation and Usage](#installation-and-usage)
8. [Benchmarks and Visualizations](#benchmarks-and-visualizations)

---

## Motivation

PyTorch eager execution is not limited by Python performance. The main bottlenecks often come from how operations are mapped onto GPU hardware. In transformer workloads, three issues appear repeatedly: **memory bandwidth usage**, **unnecessary memory allocation**, and **CUDA kernel launch overhead**.

**Memory bandwidth limitations**. Modern GPUs can perform significantly more computations per second than they can move data through memory. For example, an RTX 4070 Mobile GPU can deliver tens of TFLOPS of compute while being limited to around 256 GB/s of memory bandwidth. Operations such as RMSNorm, SwiGLU, and RoPE are typically bandwidth-bound because they perform relatively few calculations compared to the amount of data they read and write.

When these operations are executed as separate PyTorch operations, intermediate results are repeatedly written back to and loaded from GPU memory. A fused kernel can combine these steps into a single operation, reducing unnecessary memory transfers by processing the data in one pass.

**Reducing intermediate memory usage**. Some operations can create large temporary tensors that significantly increase GPU memory consumption. For example, computing cross-entropy loss over a vocabulary of 128K tokens requires generating a logits tensor with shape [`batch·seq, 128256`] before applying softmax and calculating the loss. For large batches or long context lengths, this intermediate tensor can consume several gigabytes of VRAM despite being discarded immediately afterward.

Implementing the computation directly inside a custom kernel avoids materializing unnecessary intermediate results, reducing memory usage and allowing larger workloads to fit on the GPU.

**CUDA kernel launch overhead**. Each PyTorch operation typically results in a separate CUDA kernel launch. While the overhead of a single launch is small, repeated launches can become significant in workloads with many sequential operations. For example, a Viterbi decoder implemented with a Python loop may launch a separate kernel for every timestep in the sequence. With thousands of timesteps, the accumulated dispatch overhead can become a measurable portion of the runtime.

Custom kernels can reduce this overhead by combining multiple operations into a single execution. Persistent kernels can also keep the computation within one kernel launch by handling iterative work directly on the GPU.

The goal of this project is not to rely on specialized hardware features, but to improve GPU utilization by designing kernels that match the underlying workload. By combining operations where appropriate and providing a direct memory path between PyTorch tensors and CUDA kernels, the library reduces unnecessary data movement, intermediate allocations, and execution overhead.
## System Architecture

Four layers, each with one job:

```
  PyTorch
  torch.Tensor (CUDA, contiguous) - fp32 / fp16 / bfloat16 / fp8
        │
        │   data_ptr(), shape, dtype, current CUDA stream
        ▼
  Python Wrappers                                    custom_cuda/kernels/*.py
  allocate output tensor(s), call the native extension - no shape/dtype
  logic lives here; it's all pushed down into the Rust layer below
        │
        │   Bound<'_, PyAny> -> validated CudaTensorView
        ▼
  Rust CFFI + PyO3                                          src/kernels/*.rs
  validate device / dtype / contiguity / shape, map failures to PyErr
  pull the tensor's raw device pointer + the current torch CUDA stream
  call the extern "C" launcher - zero host-device copies, zero
  intermediate allocation
        │
        │   extern "C" launch_<kernel>_fwd(ptr, ..., stream)
        ▼
  C++ / CUDA Kernels                                     csrc/kernels/*.cu
  warp-shuffle reductions, register-blocked GEMM tiling, shared-memory
  staging, vectorized uint4 loads/stores, persistent kernels
```

The CUDA kernels in `csrc/kernels/*.cu` operate directly on the same device memory allocated and managed by PyTorch. There is no DLPack conversion, CPU memory staging, or additional allocation involved in the execution path. The Rust layer is responsible only for validating that a tensor can be safely passed to a raw CUDA kernel, including checking the device, data type, memory layout, and shape. It also retrieves the CUDA stream currently used by PyTorch so that custom kernel execution remains correctly synchronized with the rest of the model's computation.

The project does not rely on `setuptools`, `CMakeLists.txt`, or a separate native extension build process. Instead, the CUDA compilation is handled through `build.rs`, a standard Rust build script that runs automatically during the `cargo` or `maturin` build process.

During compilation, `build.rs` discovers the CUDA source files under `csrc/kernels/`, locates the installed CUDA toolkit using `CUDA_PATH` or `CUDA_HOME`, and compiles each kernel with `nvcc` using optimized build settings (`-O3`, `--use_fast_math`, `-arch=sm_89`, and `-std=c++17`).

On Windows, CUDA compilation requires `cl.exe` as the host compiler. Since it is not always available through the default system path, the build script uses the `cc` crate's MSVC detection mechanism to locate the compiler automatically and passes it to `nvcc` through the `-ccbin` option. This avoids requiring users to manually open a Visual Studio Developer Command Prompt before building.

The compiled CUDA object files are packaged into a static library and linked directly into the PyO3-based Python extension. As a result, running `maturin develop` handles the complete build process: compiling CUDA kernels, linking the Rust bindings, and installing the resulting `custom_cuda._native` extension module into the Python environment.

## Engineering Principles

The following principles were applied consistently across all 12 kernels in the repository.

**Zero-copy GPU residency.** Every kernel operates directly on the tensor memory already allocated by PyTorch on the GPU. No kernel performs `.cpu()`, `.to(device)`, or transfers data through host memory during execution. The Rust layer handles validation and pointer extraction, while the actual tensor data remains on the GPU throughout the computation.

**GPU-level benchmarking instead of wall-clock timing.** All performance results reported in this repository and in `project_plan.md` are generated from the scripts in `benchmarks/`. Measurements use CUDA events (`torch.cuda.Event(enable_timing=True)`) to capture GPU execution time accurately rather than relying on CPU wall-clock timers.

To reduce measurement noise, the benchmarks include L2 cache flushing between iterations using a 256 MB scratch buffer, preventing subsequent runs from benefiting from cached data. Each benchmark includes a warmup phase followed by at least 100 measured iterations, with results reported using the median and interquartile range.

**Optimization driven by profiling results.** Each kernel was first implemented with a straightforward design, benchmarked, and then optimized based on observed bottlenecks. Optimizations were evaluated through measurement rather than assumptions about expected improvements.

Some optimization attempts did not provide benefits and were removed. For example:

- Kernel 1's shared-memory row-staging approach showed no improvement because the targeted memory access pattern was already benefiting from L2 cache, while the added shared-memory usage reduced occupancy and caused around a 5% slowdown.

- Kernel 9's increase of kBlockK from 8 to 16 produced no throughput improvement and reduced performance for some input sizes.

- Kernel 3's initial vectorization attempt introduced a launch configuration issue that made performance worse. The issue was identified, corrected, and documented instead of being included as an ineffective optimization.

These experiments and their measured results are documented in project_plan.md.

**Transparent performance reporting.** Performance targets are reported based on actual measurements, including cases where a kernel does not outperform the baseline. Kernel 5 (fused MatMul + bias) and Kernel 8 (fused cosine similarity top-k) do not meet their target performance compared to cuBLAS-backed implementations, while Kernel 9 falls short for larger embedding dimensions.

These results are documented along with the reasons behind them. In some cases, vendor libraries such as cuBLAS have highly optimized implementations using specialized hardware features like tensor cores, making it unrealistic for a custom kernel without those optimizations to achieve the same throughput. Reporting these limitations provides a more accurate view of the project's performance.

**Correctness before optimization.** Performance measurements are only collected after each kernel passes its correctness tests. The test suite contains 1,665 tests covering fp32, fp16, and bf16 precision, non-contiguous tensors, and edge cases such as empty batches, single-dimensional inputs, single-token workloads, and non-power-of-two dimensions.

Across the 12 kernels, 1,539 tests execute independently without requiring **torch.compile**. The remaining tests compare results against **torch.compile** implementations for cases where that comparison provides a meaningful reference

## Integration Proof: A Real Llama-3-8B Block

The individual kernels in this repository are validated using synthetic test tensors, which verifies their correctness in isolation. To demonstrate their use in a realistic setting, `examples/llama_block.py` implements a Llama 3 8B decoder block in two configurations using the same model architecture and identical weights.

The first implementation uses standard PyTorch eager operations, while the second replaces the corresponding operations with the custom implementations of Kernel 1 (RMSNorm + Residual), Kernel 2 (SwiGLU), and Kernel 3 (RoPE).

The decoder block uses the same configuration as the released Llama 3 8B model, including `hidden_size=4096`, `intermediate_size=14336`, 32 query heads, 8 key-value heads for grouped-query attention, `rope_theta=500000`, and `rms_norm_eps=1e-5`.

All remaining components, including the QKV projections, output projection, feed-forward matrix multiplications, and attention computation, are implemented using identical PyTorch code in both versions. Since these operations are unchanged, any measured performance difference can be attributed to the three custom kernels rather than differences in GEMM or attention implementations.

Before benchmarking, the outputs of both decoder blocks are compared to verify numerical agreement. Performance measurements are only collected after this correctness check passes.

```
$ python examples/llama_block.py

Verifying baseline and custom blocks agree numerically before benchmarking...
Correctness check passed - outputs match within bf16 tolerance.

            Llama-3-8B Block: PyTorch Eager vs. Custom CUDA Kernels
┌─────────────┬─────────────────────┬────────────────────┬────────────────────┐
│ Metric      │   Baseline (PyTorch │        Custom CUDA │        Improvement │
│             │              Eager) │            Kernels │                    │
├─────────────┼─────────────────────┼────────────────────┼────────────────────┤
│ Latency     │           65.401 ms │          50.836 ms │      1.287x faster │
├─────────────┼─────────────────────┼────────────────────┼────────────────────┤
│ Peak Memory │           2268.2 MB │          1740.2 MB │ +528.0 MB (+23.3%) │
└─────────────┴─────────────────────┴────────────────────┴────────────────────┘
```

On an RTX 4070 Laptop GPU, replacing the three PyTorch operations with their custom kernel implementations reduced the decoder block's forward-pass runtime by **1.29×** and lowered peak VRAM usage by **528 MB (23.3%)** for a configuration of `batch_size=1`, `seq_len=4096`, and `bf16`.

The overall speedup is more modest than the individual kernel benchmarks presented later in this document, which is expected. Only three operations within the decoder block were replaced, while the computationally dominant components, including the matrix multiplications and attention layers, remain standard PyTorch implementations in both versions.

The individual kernel benchmarks measure the performance improvement of each operation in isolation. The decoder block benchmark serves a different purpose: it demonstrates that replacing these kernels within a realistic model produces a measurable improvement in end-to-end execution while keeping the rest of the implementation unchanged.

## The 12 Kernels

Every kernel below was built through the same six-step process: write a correctness baseline and pytest suite first, implement a first-pass CUDA kernel, bind it through Rust/PyO3, validate correctness across fp32/fp16/bf16 and edge cases, benchmark on real hardware with L2 flushing, then plot the results. Full design history - including the failed first attempts - is in `project_plan.md`.

### Core Transformer Ops

**Kernel 1 - Fused RMSNorm + Residual Addition**
`y, residual_out = RMSNorm(x + residual) * weight, x + residual` in one launch instead of a separate add-then-normalize pair.
- *Techniques:* single-pass block reduction (warp-shuffle `__shfl_down_sync` + shared-memory cross-warp combine) for the sum-of-squares statistic; `float4`/vectorized loads and stores; fused residual-write and normalized-write from one register tile.
- *Result:* **4.2-4.8x faster** (fp16/bf16), **2.4x** (fp32) vs. eager, **78-89% of peak memory bandwidth**. A shared-memory row-staging variant was tried and rejected - the redundant reread it targeted was already an L2 hit.

**Kernel 2 - Fused SwiGLU Gated Activation**
`y = SiLU(gate) * up`, fusing the activation and the elementwise multiply over the FFN's large intermediate tensor.
- *Techniques:* single-pass elementwise fusion; `float4` (fp32) / `uint4`-packed `half2`/`bf16x2` (fp16/bf16) vectorized access with a 16-byte-alignment check gating a scalar fallback for any non-vector-width-divisible remainder.
- *Result:* **2.4-6.4x faster** vs. a two-kernel eager baseline. Vectorization closed a mid-size bandwidth gap from 75% to 85.5% of peak on the shapes it targeted.

**Kernel 3 - Fused Rotary Position Embedding (RoPE)**
Rotates Q and K together in one launch given precomputed sin/cos tables, correctly handling grouped-query attention's mismatched Q/K head counts.
- *Techniques:* one kernel launch rotates both Q and K (dispatching per-row on which tensor a given block index falls into); vectorized half-head-dim access; block size fixed to one warp for the vectorized path after profiling caught a launch-config bug in the first vectorized attempt.
- *Result:* **3.3-9.1x faster** vs. eager. Fixing the vectorized kernel's block size (a real bug, not a design tradeoff) improved fp16/bf16 bandwidth by 8-16 percentage points across nearly every shape.

### Compute & Loss Optimizations

**Kernel 4 - Fused Linear Cross-Entropy Loss**
Fuses the vocabulary projection with cross-entropy using chunked online-softmax, so the full `[tokens, vocab]` logits tensor - multiple gigabytes at Llama-3's 128K vocabulary - is never materialized.
- *Techniques:* vocab-dimension tiling with a running max/sum carried across chunks; the chunk's matmul is left to cuBLAS via PyTorch, with the CUDA kernel fusing the online-softmax update and target-logit gather for that chunk into one pass.
- *Result:* **12.5-31.3x peak-VRAM reduction** on realistic Llama-3-vocab shapes (target was ≥4x). A native-dtype rewrite (avoiding an `.float()` upcast copy in the first version) brought latency overhead from ~10-14% down to roughly zero.

**Kernel 5 - Fused MatMul + Add Bias**
`y = x @ Wᵀ + b` with the bias-add fused into the GEMM epilogue instead of a separate elementwise pass.
- *Techniques:* shared-memory-tiled GEMM with 1D register blocking - each thread accumulates 8 output elements along M instead of 1, so each shared-memory read is reused across 8 FMAs.
- *Result (honest miss):* register blocking took the kernel from ~1.3 TFLOPS to **1.8-3.0 TFLOPS**, but that's still short of both the naive-unfused target and cuBLAS's throughput - cuBLAS's fp16/bf16 path uses tensor cores, which no amount of CUDA-core tiling matches without WMMA/MMA intrinsics. Reported as a miss rather than reframed against a softer baseline.

### MoE Routing & Permutation

**Kernel 6 - MoE Top-K Router**
Fuses softmax gating and top-k expert selection into one kernel, entirely register-resident with zero host-device synchronization.
- *Techniques:* one warp per token; softmax via warp-shuffle max/sum reduction; top-k via `k` rounds of warp-wide argmax-and-mask, no shared memory.
- *Result:* first pass beat eager by only 1.1-2.2x against a ≥3x target - profiling showed the kernel was launch-overhead-bound (every shape finishes in well under a millisecond), not compute-bound. Doubling warps-per-block from 8 to 16 halved the blocks launched and took the Mixtral-scale fp16 case to **3.35x**, clearing the target.

**Kernel 7 - Token Scatter/Gather (Permute / Unpermute)**
Reorders token embeddings into contiguous per-expert buffers for MoE dispatch, and gathers them back with the weighted-gate combine fused into the same pass.
- *Techniques:* both directions expressed as pure gathers, never scatter-add/atomics; permutation indices computed once via `argsort`, the CUDA kernels handle only the bandwidth-critical row movement; alignment-gated vectorized `uint4` copies with a scalar fallback.
- *Result:* permute lands within a few percent of PyTorch's own `index_select` (85-97% of peak bandwidth); the fused unpermute+combine beats eager's naive fancy-indexing chain by **3.2-8.6x** (75-92% of peak bandwidth), since eager materializes a full intermediate tensor that the fused kernel never allocates.

### RAG & Vector Search

**Kernel 8 - Fused Cosine Similarity + Top-K**
Computes cosine similarity between queries and a candidate pool and returns the top-k matches without ever materializing the full `[queries, candidates]` score matrix.
- *Techniques:* L2 normalization fused into the dot-product accumulation loop; a two-kernel partition-then-merge design so the grid scales with candidate-pool size even when the query count is tiny.
- *Result:* the first version was 20-97x *slower* than eager - a grid-sizing bug meant parallelism scaled with query count only, and realistic RAG shapes have few queries against huge candidate pools. The partition/merge redesign recovered it to 1.7-5.1x slower than eager (short of the ≥3x-faster target, the same cuBLAS-tensor-core gap as Kernel 5) - a dramatic fix, reported honestly as still short of target.

**Kernel 9 - Block Pairwise Distance Matrix**
Computes the full squared-Euclidean distance matrix between two vector sets using a shared-memory tiled block algorithm, with an explicit clamp guarding against floating-point cancellation.
- *Techniques:* reuses Kernel 5's register-blocked tiled-GEMM structure; `dist_sq = max(‖a‖² + ‖b‖² − 2·a·b, 0)` epilogue.
- *Result:* **1.7-1.8x faster** than `torch.cdist` at large matrix sizes with modest embedding dimensions, but 2.2x *slower* at large embedding dimensions (≥1536) - cuBLAS's GEMM scales better with reduction depth than a hand-tiled kernel without tensor cores. A `kBlockK=8→16` widening was tried to close the gap and measured no improvement; reverted.

### Graph & Sequential Algorithms

**Kernel 10 - Spatiotemporal Graph Message Passing**
One step of neighborhood aggregation over spatial and temporal edge sets, for GNN workloads over sensor/traffic-network-style graphs.
- *Techniques:* CSR-based neighbor iteration (edges pre-sorted by destination node); the feature dimension, not the neighbor list, is split across a warp's lanes, so every output element is owned by exactly one lane - atomics are eliminated entirely rather than merely reduced.
- *Result:* **2.3-4.1x faster** than a `scatter_add`-based reference for graphs with 20,000+ nodes, meeting the ≥2x target; falls short at very small graphs (2,000 nodes) where launch overhead dominates, reported as-is.

**Kernel 11 - Parallel Viterbi Algorithm**
Batched HMM decoding via a single persistent kernel that loops over the entire sequence internally, instead of one kernel launch per timestep.
- *Techniques:* one block per batch item; the transition matrix is staged into shared memory once and stays resident for the whole recursion; warp-shuffle max-reduction for the final argmax-over-states step.
- *Result:* **20-120x faster** than a per-timestep Python loop across every sequence length, batch size, and state count tested - the target was ≥5x. Kernel launch count drops from O(sequence length) to exactly 1.

### Precision & Quantization

**Kernel 12 - FP8 Dynamic Quantization & Casting**
Computes a dynamic per-block (128×128 tile) or per-tensor scale from a tensor's live amax and casts to fp8 (e4m3/e5m2) in a fused pass, DeepSeek-V3-style.
- *Techniques:* one thread block per 128×128 tile does its own amax reduction, scale derivation, and cast+store with zero cross-block communication; 16 fp8 bytes packed into a `uint4` for vectorized stores.
- *Result:* **3.1-11.7x faster** than a separate amax-then-cast reference (target was ≥2x), reaching **up to 87% of the theoretical single-pass memory-bandwidth ceiling** - strong evidence the fused kernel's second read is being served from L2 cache rather than a second DRAM round trip.

## Developer CLI (`custom_cuda_cli`)

A `uv`-style command-line toolkit for evaluating, benchmarking, and auditing these kernels - not for running inference. Nine subcommands, all thin wrappers around the same `baselines/`, `tests/`, `benchmarks/`, and `scripts/` infrastructure documented above, so the CLI's numbers are always the exact numbers the underlying scripts would produce standalone.

```bash
# Check the local environment: CUDA toolkit, nvcc, PyTorch, GPU, Rust extension
custom_cuda_cli doctor

# List all 12 kernels, or read one kernel's full technical spec
custom_cuda_cli list
custom_cuda_cli info fp8_quant

# Run a kernel's correctness suite (or omit the kernel name to run all 12)
custom_cuda_cli verify pairwise_distance

# Run the hardware benchmark harness and print latency/bandwidth/TFLOPS
custom_cuda_cli benchmark viterbi

# Same benchmark data, reshaped into eager-vs-kernel speedup multipliers
custom_cuda_cli compare graph_message_passing

# Regenerate a kernel's performance charts
custom_cuda_cli visualize cosine_topk

# Capture a torch.profiler Chrome trace for bottleneck analysis
custom_cuda_cli profile swiglu --iters 20
```

## Installation and Usage

Requires a CUDA-capable GPU, the CUDA toolkit (11.8+, `nvcc` and `CUDA_PATH`/`CUDA_HOME` set), a Rust toolchain (1.75+), and Python 3.9+ with PyTorch 2.3+.

```bash
git clone git@github.com:BhargavKumarNath/custom_cuda_kernals.git
cd custom_cuda_kernals
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install maturin
maturin develop --release                            # builds the CUDA kernels + Rust bindings
```

Once built, every kernel is a plain Python function that takes and returns ordinary CUDA tensors - no custom tensor subclass, no context manager, no graph capture required:

```python
import torch
from custom_cuda.kernels.rmsnorm_residual import rmsnorm_residual

x = torch.randn(4, 2048, 4096, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

# y = RMSNorm(x + residual) * weight, residual_out = x + residual - one kernel launch
y, residual_out = rmsnorm_residual(x, residual, weight, eps=1e-5)
```

Alternatively, `pip install .` builds and installs the package the same way via the `maturin` build backend declared in `pyproject.toml`.

## Benchmarks and Visualizations

Every kernel's full benchmark sweep is plotted automatically (`scripts/plot_*.py`) into `visualizations/<NN_kernel_name>/` - four charts per kernel: a speedup bar chart, a latency-or-bandwidth-vs-size curve, a bandwidth-or-TFLOPS-vs-peak chart, and a scaling curve across the kernel's own natural dimension (sequence length, `k`, node count, embedding dimension, or similar). A few representative examples:

| Kernel 1: RMSNorm + Residual | Kernel 11: Parallel Viterbi |
|---|---|
| ![RMSNorm speedup](visualizations/01_fused_rmsnorm_residual/01_speedup_bar.png) | ![Viterbi speedup](visualizations/11_parallel_viterbi/01_speedup_vs_seq_len.png) |

| Kernel 9: Pairwise Distance | Kernel 12: FP8 Quantization |
|---|---|
| ![Pairwise distance bandwidth](visualizations/09_block_pairwise_distance/04_bandwidth_vs_peak.png) | ![FP8 quant speedup](visualizations/12_fp8_dynamic_quant/01_speedup_bar.png) |

The full set - all 48 charts across all 12 kernels, in both PNG and SVG - lives under `visualizations/`. Raw benchmark data (CSV) is regenerated on demand via `custom_cuda_cli benchmark <kernel>` and isn't committed to the repository, since it's a function of whatever GPU it's run on.

---

Full technical specification, per-kernel design history, and the complete milestone log for how this library was built are in [`project_plan.md`](./project_plan.md).
