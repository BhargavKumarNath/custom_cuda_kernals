//! Rust/PyO3 binding for Kernel 6 (MoE Top-K Router). Wraps the
//! `extern "C"` launcher declared in csrc/includes/moe_router.h / defined
//! in csrc/kernels/moe_router.cu.

use std::ffi::c_void;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_moe_router_fwd(
        logits: *const c_void,
        topk_weights: *mut f32,
        topk_indices: *mut i64,
        num_tokens: i64,
        num_experts: i64,
        k: i64,
        dtype: i32,
        renormalize: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `moe_router_fwd(logits, topk_weights, topk_indices, renormalize)`
///
/// Computes `softmax(logits, dim=-1)` and selects the top-k experts per
/// token (k inferred from `topk_weights`/`topk_indices`'s shape) in place
/// into the caller-provided output tensors. `logits`: `[T, E]`, contiguous
/// CUDA tensor (float32/float16/bfloat16). `topk_weights`: `[T, k]`,
/// float32. `topk_indices`: `[T, k]`, int64. `E` must be <= 256.
#[pyfunction]
pub fn moe_router_fwd(
    py: Python<'_>,
    logits: Bound<'_, PyAny>,
    topk_weights: Bound<'_, PyAny>,
    topk_indices: Bound<'_, PyAny>,
    renormalize: bool,
) -> PyResult<()> {
    let logits_view = validate_cuda_tensor("logits", &logits)?;
    let weights_view = validate_cuda_tensor("topk_weights", &topk_weights)?;
    let indices_view = validate_int64_cuda_tensor("topk_indices", &topk_indices)?;

    if weights_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "topk_weights must be float32, got {:?}",
            weights_view.dtype
        ))
        .into());
    }

    if logits_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "logits must be 2-D [num_tokens, num_experts], got {:?}",
            logits_view.shape
        ))
        .into());
    }
    let (num_tokens, num_experts) = (logits_view.shape[0], logits_view.shape[1]);

    if weights_view.shape.len() != 2 || weights_view.shape[0] != num_tokens {
        return Err(KernelError::ShapeMismatch(format!(
            "topk_weights shape {:?} must be [{num_tokens}, k]",
            weights_view.shape
        ))
        .into());
    }
    let k = weights_view.shape[1];
    if indices_view.shape != [num_tokens, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "topk_indices shape {:?} must match topk_weights' [{num_tokens}, {k}]",
            indices_view.shape
        ))
        .into());
    }
    if k > num_experts {
        return Err(PyRuntimeError::new_err(format!(
            "k ({k}) must be <= num_experts ({num_experts})"
        )));
    }
    if num_experts > 256 {
        return Err(KernelError::ShapeMismatch(format!(
            "num_experts ({num_experts}) exceeds this kernel's supported maximum of 256"
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (logits_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_moe_router_fwd(
            logits_view.ptr as *const c_void,
            weights_view.ptr as *mut f32,
            indices_view.ptr as *mut i64,
            num_tokens,
            num_experts,
            k,
            logits_view.dtype as i32,
            renormalize as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_moe_router_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
