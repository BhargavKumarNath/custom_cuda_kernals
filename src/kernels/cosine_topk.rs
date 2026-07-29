//! Rust/PyO3 bindings for Kernel 8 (Fused Cosine Similarity + Top-K).
//! Wraps the two `extern "C"` launchers declared in
//! csrc/includes/cosine_topk.h / defined in csrc/kernels/cosine_topk.cu.
//! See that header's comment for the partition+merge architecture — the
//! Python wrapper (custom_cuda/kernels/cosine_topk.py) chooses
//! `num_partitions` and orchestrates both calls.

use std::ffi::c_void;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_cosine_topk_partial_fwd(
        queries: *const c_void,
        candidates: *const c_void,
        partial_scores: *mut f32,
        partial_indices: *mut i64,
        num_queries: i64,
        num_candidates: i64,
        dim: i64,
        k: i64,
        num_partitions: i64,
        eps: f32,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32;

    fn launch_cosine_topk_merge_fwd(
        partial_scores: *const f32,
        partial_indices: *const i64,
        topk_scores: *mut f32,
        topk_indices: *mut i64,
        num_queries: i64,
        num_partitions: i64,
        k: i64,
        stream: *mut c_void,
    ) -> i32;
}

fn get_stream_ptr(py: Python<'_>, device_index: i64) -> PyResult<u64> {
    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (device_index,))?;
    stream_obj.getattr("cuda_stream")?.extract()
}

/// `cosine_topk_partial_fwd(queries, candidates, partial_scores, partial_indices, eps)`
///
/// Computes a partial top-k (over one candidate-pool partition per warp)
/// for every query, without materializing any full similarity matrix.
/// `queries`: `[Q, D]`, `candidates`: `[N, D]`, contiguous CUDA tensors
/// sharing one dtype. `partial_scores`: `[Q, P, k]` float32.
/// `partial_indices`: `[Q, P, k]` int64 (`P` = num_partitions, inferred
/// from these tensors' shape).
#[pyfunction]
pub fn cosine_topk_partial_fwd(
    py: Python<'_>,
    queries: Bound<'_, PyAny>,
    candidates: Bound<'_, PyAny>,
    partial_scores: Bound<'_, PyAny>,
    partial_indices: Bound<'_, PyAny>,
    eps: f64,
) -> PyResult<()> {
    let queries_view = validate_cuda_tensor("queries", &queries)?;
    let candidates_view = validate_cuda_tensor("candidates", &candidates)?;
    let scores_view = validate_cuda_tensor("partial_scores", &partial_scores)?;
    let indices_view = validate_int64_cuda_tensor("partial_indices", &partial_indices)?;

    if candidates_view.dtype != queries_view.dtype {
        return Err(
            KernelError::DtypeMismatch("candidates", candidates_view.dtype, queries_view.dtype).into(),
        );
    }
    if scores_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "partial_scores must be float32, got {:?}",
            scores_view.dtype
        ))
        .into());
    }

    if queries_view.shape.len() != 2 || candidates_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "queries and candidates must be 2-D; got queries={:?}, candidates={:?}",
            queries_view.shape, candidates_view.shape
        ))
        .into());
    }
    let (num_queries, dim) = (queries_view.shape[0], queries_view.shape[1]);
    let (num_candidates, cand_dim) = (candidates_view.shape[0], candidates_view.shape[1]);
    if cand_dim != dim {
        return Err(KernelError::ShapeMismatch(format!(
            "candidates' dim ({cand_dim}) must match queries' dim ({dim})"
        ))
        .into());
    }

    if scores_view.shape.len() != 3 || scores_view.shape[0] != num_queries {
        return Err(KernelError::ShapeMismatch(format!(
            "partial_scores shape {:?} must be [{num_queries}, num_partitions, k]",
            scores_view.shape
        ))
        .into());
    }
    let (num_partitions, k) = (scores_view.shape[1], scores_view.shape[2]);
    if indices_view.shape != [num_queries, num_partitions, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "partial_indices shape {:?} must match partial_scores' {:?}",
            indices_view.shape, scores_view.shape
        ))
        .into());
    }
    if k > 32 {
        return Err(PyRuntimeError::new_err(format!("k ({k}) exceeds this kernel's supported maximum of 32")));
    }
    if k > num_candidates {
        return Err(PyRuntimeError::new_err(format!(
            "k ({k}) must be <= num_candidates ({num_candidates})"
        )));
    }

    let stream_ptr = get_stream_ptr(py, queries_view.device_index)?;

    let err = unsafe {
        launch_cosine_topk_partial_fwd(
            queries_view.ptr as *const c_void,
            candidates_view.ptr as *const c_void,
            scores_view.ptr as *mut f32,
            indices_view.ptr as *mut i64,
            num_queries,
            num_candidates,
            dim,
            k,
            num_partitions,
            eps as f32,
            queries_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(
            KernelError::Cuda(format!("launch_cosine_topk_partial_fwd failed, cudaError_t = {err}")).into(),
        );
    }
    Ok(())
}

/// `cosine_topk_merge_fwd(partial_scores, partial_indices, topk_scores, topk_indices)`
///
/// Merges `[Q, P, k]` partial top-k results (from `cosine_topk_partial_fwd`)
/// into the final `[Q, k]` top-k. Pure score/index merge — no dot products.
#[pyfunction]
pub fn cosine_topk_merge_fwd(
    py: Python<'_>,
    partial_scores: Bound<'_, PyAny>,
    partial_indices: Bound<'_, PyAny>,
    topk_scores: Bound<'_, PyAny>,
    topk_indices: Bound<'_, PyAny>,
) -> PyResult<()> {
    let partial_scores_view = validate_cuda_tensor("partial_scores", &partial_scores)?;
    let partial_indices_view = validate_int64_cuda_tensor("partial_indices", &partial_indices)?;
    let topk_scores_view = validate_cuda_tensor("topk_scores", &topk_scores)?;
    let topk_indices_view = validate_int64_cuda_tensor("topk_indices", &topk_indices)?;

    if partial_scores_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "partial_scores must be float32, got {:?}",
            partial_scores_view.dtype
        ))
        .into());
    }
    if topk_scores_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "topk_scores must be float32, got {:?}",
            topk_scores_view.dtype
        ))
        .into());
    }

    if partial_scores_view.shape.len() != 3 {
        return Err(KernelError::ShapeMismatch(format!(
            "partial_scores must be 3-D [Q, num_partitions, k], got {:?}",
            partial_scores_view.shape
        ))
        .into());
    }
    let (num_queries, num_partitions, k) = (
        partial_scores_view.shape[0],
        partial_scores_view.shape[1],
        partial_scores_view.shape[2],
    );
    if partial_indices_view.shape != [num_queries, num_partitions, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "partial_indices shape {:?} must match partial_scores' {:?}",
            partial_indices_view.shape, partial_scores_view.shape
        ))
        .into());
    }
    if topk_scores_view.shape != [num_queries, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "topk_scores shape {:?} must be [{num_queries}, {k}]",
            topk_scores_view.shape
        ))
        .into());
    }
    if topk_indices_view.shape != [num_queries, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "topk_indices shape {:?} must be [{num_queries}, {k}]",
            topk_indices_view.shape
        ))
        .into());
    }

    let stream_ptr = get_stream_ptr(py, partial_scores_view.device_index)?;

    let err = unsafe {
        launch_cosine_topk_merge_fwd(
            partial_scores_view.ptr as *const f32,
            partial_indices_view.ptr as *const i64,
            topk_scores_view.ptr as *mut f32,
            topk_indices_view.ptr as *mut i64,
            num_queries,
            num_partitions,
            k,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(
            KernelError::Cuda(format!("launch_cosine_topk_merge_fwd failed, cudaError_t = {err}")).into(),
        );
    }
    Ok(())
}
