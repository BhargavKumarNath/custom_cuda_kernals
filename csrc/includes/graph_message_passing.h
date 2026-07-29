#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Spatiotemporal Graph Message Passing (Kernel 10). See project_plan.md
// Section 3.10 and baselines/graph_message_passing.py for the reference
// semantics:
//
//   out[dst] = sum over spatial edges (src->dst):  spatial_weight  * x_curr[src]
//            + sum over temporal edges (src->dst): temporal_weight * x_prev[src]
//
// Both edge sets are passed in CSR form, indexed by destination node —
// `*_indptr[node]`/`*_indptr[node+1]` bound that node's incoming-edge
// range in `*_col`/`*_weight` — built by the Python wrapper (sorting
// COO edges by destination is cheap O(E) bookkeeping, delegated to
// PyTorch the same way Kernel 4/7/8/9 delegate index bookkeeping).
// `x_curr`/`x_prev`: `[num_nodes, feature_dim]`, any supported dtype
// (see common.cuh). `*_weight` arrays are always float32 (derived
// quantities, not a feature tensor to quantize — same convention as
// Kernel 9's row norms). `out`: `[num_nodes, feature_dim]`, same dtype
// as `x_curr`/`x_prev`. Launch is a no-op if num_nodes or feature_dim
// <= 0.
extern "C" cudaError_t launch_graph_message_passing_fwd(
    const void* x_curr,
    const void* x_prev,
    const int64_t* spatial_indptr,
    const int64_t* spatial_col,
    const float* spatial_weight,
    const int64_t* temporal_indptr,
    const int64_t* temporal_col,
    const float* temporal_weight,
    void* out,
    int64_t num_nodes,
    int64_t feature_dim,
    int32_t dtype,
    cudaStream_t stream);
