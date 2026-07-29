//! Rust/PyO3 binding for Kernel 4's chunk-update kernel (Fused Linear
//! Cross Entropy Loss). Wraps the `extern "C"` launcher declared in
//! csrc/includes/linear_cross_entropy.h / defined in
//! csrc/kernels/linear_cross_entropy.cu. The vocab-chunking loop and the
//! per-chunk `hidden @ weight_chunk.T` matmul live in the Python wrapper
//! (custom_cuda/kernels/linear_cross_entropy.py) — see that module's
//! docstring for the full architecture.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_linear_ce_chunk_update(
        logits_chunk: *const c_void,
        targets: *const c_void,
        v_start: i64,
        n: i64,
        chunk_v: i64,
        running_max: *mut c_void,
        running_sum: *mut c_void,
        target_logit: *mut c_void,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `linear_ce_chunk_update(logits_chunk, targets, v_start, running_max, running_sum, target_logit)`
///
/// Fuses the online-softmax running-max/running-sum update and the
/// target-logit gather for one `[N, chunk_v]` logits chunk into a single
/// kernel launch. `logits_chunk` must be float32 (the Python wrapper
/// upcasts before calling); `targets` must be int64; `running_max`,
/// `running_sum`, `target_logit` are float32 `[N]` accumulators updated
/// in place. See csrc/includes/linear_cross_entropy.h for the calling
/// convention (accumulators must be initialized by the caller and threaded
/// through chunks in increasing `v_start` order).
#[pyfunction]
pub fn linear_ce_chunk_update(
    py: Python<'_>,
    logits_chunk: Bound<'_, PyAny>,
    targets: Bound<'_, PyAny>,
    v_start: i64,
    running_max: Bound<'_, PyAny>,
    running_sum: Bound<'_, PyAny>,
    target_logit: Bound<'_, PyAny>,
) -> PyResult<()> {
    let logits_view = validate_cuda_tensor("logits_chunk", &logits_chunk)?;
    let targets_view = validate_int64_cuda_tensor("targets", &targets)?;
    let max_view = validate_cuda_tensor("running_max", &running_max)?;
    let sum_view = validate_cuda_tensor("running_sum", &running_sum)?;
    let target_logit_view = validate_cuda_tensor("target_logit", &target_logit)?;

    // Accumulators (running_max/running_sum/target_logit) are always
    // float32 regardless of logits_chunk's dtype — same convention as
    // Kernels 1-3's internal reduction math.
    for (name, v) in [
        ("running_max", &max_view),
        ("running_sum", &sum_view),
        ("target_logit", &target_logit_view),
    ] {
        if v.dtype != KernelDType::F32 {
            return Err(KernelError::UnsupportedDtype(format!(
                "{name} must be float32, got {:?}",
                v.dtype
            ))
            .into());
        }
    }

    if logits_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "logits_chunk must be 2-D [N, chunk_v], got {:?}",
            logits_view.shape
        ))
        .into());
    }
    let (n, chunk_v) = (logits_view.shape[0], logits_view.shape[1]);

    if targets_view.shape != [n] {
        return Err(KernelError::ShapeMismatch(format!(
            "targets shape {:?} must be [{n}] (logits_chunk's row count)",
            targets_view.shape
        ))
        .into());
    }
    for (name, v) in [
        ("running_max", &max_view),
        ("running_sum", &sum_view),
        ("target_logit", &target_logit_view),
    ] {
        if v.shape != [n] {
            return Err(KernelError::ShapeMismatch(format!(
                "{name} shape {:?} must be [{n}]",
                v.shape
            ))
            .into());
        }
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (logits_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_linear_ce_chunk_update(
            logits_view.ptr as *const c_void,
            targets_view.ptr as *const c_void,
            v_start,
            n,
            chunk_v,
            max_view.ptr as *mut c_void,
            sum_view.ptr as *mut c_void,
            target_logit_view.ptr as *mut c_void,
            logits_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(
            KernelError::Cuda(format!("launch_linear_ce_chunk_update failed, cudaError_t = {err}")).into(),
        );
    }
    Ok(())
}
