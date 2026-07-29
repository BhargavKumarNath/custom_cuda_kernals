//! Rust/PyO3 binding for Kernel 10 (Spatiotemporal Graph Message
//! Passing). Wraps the `extern "C"` launcher declared in
//! csrc/includes/graph_message_passing.h / defined in
//! csrc/kernels/graph_message_passing.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_graph_message_passing_fwd(
        x_curr: *const c_void,
        x_prev: *const c_void,
        spatial_indptr: *const i64,
        spatial_col: *const i64,
        spatial_weight: *const f32,
        temporal_indptr: *const i64,
        temporal_col: *const i64,
        temporal_weight: *const f32,
        out: *mut c_void,
        num_nodes: i64,
        feature_dim: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

fn validate_weight(name: &'static str, obj: &Bound<'_, PyAny>) -> PyResult<crate::kernels::tensor::CudaTensorView> {
    let view = validate_cuda_tensor(name, obj)?;
    if view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!("{name} must be float32, got {:?}", view.dtype)).into());
    }
    Ok(view)
}

/// `graph_message_passing_fwd(x_curr, x_prev, spatial_indptr,
/// spatial_col, spatial_weight, temporal_indptr, temporal_col,
/// temporal_weight, out)`
///
/// Computes, for every node `dst`:
/// `out[dst] = sum_{(src,dst) in spatial} spatial_weight * x_curr[src]
///           + sum_{(src,dst) in temporal} temporal_weight * x_prev[src]`.
/// `x_curr`/`x_prev`/`out`: `[num_nodes, feature_dim]`, contiguous CUDA
/// tensors sharing one dtype. `*_indptr`: `[num_nodes+1]` int64 CSR
/// offsets (by destination node). `*_col`: `[E]` int64 source-node
/// indices. `*_weight`: `[E]` float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn graph_message_passing_fwd(
    py: Python<'_>,
    x_curr: Bound<'_, PyAny>,
    x_prev: Bound<'_, PyAny>,
    spatial_indptr: Bound<'_, PyAny>,
    spatial_col: Bound<'_, PyAny>,
    spatial_weight: Bound<'_, PyAny>,
    temporal_indptr: Bound<'_, PyAny>,
    temporal_col: Bound<'_, PyAny>,
    temporal_weight: Bound<'_, PyAny>,
    out: Bound<'_, PyAny>,
) -> PyResult<()> {
    let x_curr_view = validate_cuda_tensor("x_curr", &x_curr)?;
    let x_prev_view = validate_cuda_tensor("x_prev", &x_prev)?;
    let out_view = validate_cuda_tensor("out", &out)?;

    for (name, v) in [("x_prev", &x_prev_view), ("out", &out_view)] {
        if v.dtype != x_curr_view.dtype {
            return Err(KernelError::DtypeMismatch(name, v.dtype, x_curr_view.dtype).into());
        }
    }

    if x_curr_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "x_curr must be 2-D [num_nodes, feature_dim]; got {:?}",
            x_curr_view.shape
        ))
        .into());
    }
    let (num_nodes, feature_dim) = (x_curr_view.shape[0], x_curr_view.shape[1]);
    for (name, v) in [("x_prev", &x_prev_view), ("out", &out_view)] {
        if v.shape != [num_nodes, feature_dim] {
            return Err(KernelError::ShapeMismatch(format!(
                "{name} shape {:?} must match x_curr's [{num_nodes}, {feature_dim}]",
                v.shape
            ))
            .into());
        }
    }

    let spatial_indptr_view = validate_int64_cuda_tensor("spatial_indptr", &spatial_indptr)?;
    let spatial_col_view = validate_int64_cuda_tensor("spatial_col", &spatial_col)?;
    let spatial_weight_view = validate_weight("spatial_weight", &spatial_weight)?;
    let temporal_indptr_view = validate_int64_cuda_tensor("temporal_indptr", &temporal_indptr)?;
    let temporal_col_view = validate_int64_cuda_tensor("temporal_col", &temporal_col)?;
    let temporal_weight_view = validate_weight("temporal_weight", &temporal_weight)?;

    for (name, v) in [("spatial_indptr", &spatial_indptr_view), ("temporal_indptr", &temporal_indptr_view)] {
        if v.shape != [num_nodes + 1] {
            return Err(KernelError::ShapeMismatch(format!(
                "{name} shape {:?} must be [num_nodes+1] = [{}]",
                v.shape,
                num_nodes + 1
            ))
            .into());
        }
    }
    if spatial_col_view.shape != spatial_weight_view.shape {
        return Err(KernelError::ShapeMismatch(format!(
            "spatial_col shape {:?} must match spatial_weight shape {:?}",
            spatial_col_view.shape, spatial_weight_view.shape
        ))
        .into());
    }
    if temporal_col_view.shape != temporal_weight_view.shape {
        return Err(KernelError::ShapeMismatch(format!(
            "temporal_col shape {:?} must match temporal_weight shape {:?}",
            temporal_col_view.shape, temporal_weight_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (x_curr_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_graph_message_passing_fwd(
            x_curr_view.ptr as *const c_void,
            x_prev_view.ptr as *const c_void,
            spatial_indptr_view.ptr as *const i64,
            spatial_col_view.ptr as *const i64,
            spatial_weight_view.ptr as *const f32,
            temporal_indptr_view.ptr as *const i64,
            temporal_col_view.ptr as *const i64,
            temporal_weight_view.ptr as *const f32,
            out_view.ptr as *mut c_void,
            num_nodes,
            feature_dim,
            x_curr_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(
            KernelError::Cuda(format!("launch_graph_message_passing_fwd failed, cudaError_t = {err}")).into(),
        );
    }
    Ok(())
}
