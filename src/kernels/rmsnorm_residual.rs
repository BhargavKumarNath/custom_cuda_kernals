//! Rust/PyO3 binding for Kernel 1 (Fused RMSNorm + Residual Addition).
//! Wraps the `extern "C"` launcher declared in
//! csrc/includes/rmsnorm_residual.h / defined in
//! csrc/kernels/rmsnorm_residual.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, CudaTensorView};

extern "C" {
    fn launch_rmsnorm_residual_fwd(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        y: *mut c_void,
        residual_out: *mut c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

fn check_dtype_matches(name: &'static str, view: &CudaTensorView, expected: &CudaTensorView) -> PyResult<()> {
    if view.dtype != expected.dtype {
        return Err(KernelError::DtypeMismatch(name, view.dtype, expected.dtype).into());
    }
    Ok(())
}

/// `rmsnorm_residual_fwd(x, residual, weight, y, residual_out, eps)`
///
/// Computes `residual_out = x + residual` and
/// `y = residual_out * rsqrt(mean(residual_out**2, -1) + eps) * weight` in
/// place into the caller-provided `y` / `residual_out` output tensors.
/// `x`, `residual`, `y`, `residual_out` must share shape `[..., cols]`;
/// `weight` must have shape `[cols]`. All five tensors must be contiguous
/// CUDA tensors of the same dtype (float32, float16, or bfloat16).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_fwd(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    residual: Bound<'_, PyAny>,
    weight: Bound<'_, PyAny>,
    y: Bound<'_, PyAny>,
    residual_out: Bound<'_, PyAny>,
    eps: f64,
) -> PyResult<()> {
    let x_view = validate_cuda_tensor("x", &x)?;
    let r_view = validate_cuda_tensor("residual", &residual)?;
    let w_view = validate_cuda_tensor("weight", &weight)?;
    let y_view = validate_cuda_tensor("y", &y)?;
    let ro_view = validate_cuda_tensor("residual_out", &residual_out)?;

    check_dtype_matches("residual", &r_view, &x_view)?;
    check_dtype_matches("weight", &w_view, &x_view)?;
    check_dtype_matches("y", &y_view, &x_view)?;
    check_dtype_matches("residual_out", &ro_view, &x_view)?;

    if r_view.shape != x_view.shape || y_view.shape != x_view.shape || ro_view.shape != x_view.shape {
        return Err(KernelError::ShapeMismatch(format!(
            "x/residual/y/residual_out must share the same shape; got x={:?}, residual={:?}, y={:?}, residual_out={:?}",
            x_view.shape, r_view.shape, y_view.shape, ro_view.shape
        ))
        .into());
    }

    let cols = *x_view
        .shape
        .last()
        .ok_or_else(|| KernelError::ShapeMismatch("x must have at least 1 dimension".to_string()))?;
    if w_view.shape != [cols] {
        return Err(KernelError::ShapeMismatch(format!(
            "weight shape {:?} must be [{cols}] (x's last dimension)",
            w_view.shape
        ))
        .into());
    }
    let rows: i64 = x_view.shape[..x_view.shape.len() - 1].iter().product();

    // Launch on PyTorch's current stream for this device so kernel
    // execution stays correctly ordered relative to surrounding ops without
    // an explicit synchronize() at the Python/Rust boundary.
    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (x_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_rmsnorm_residual_fwd(
            x_view.ptr as *const c_void,
            r_view.ptr as *const c_void,
            w_view.ptr as *const c_void,
            y_view.ptr as *mut c_void,
            ro_view.ptr as *mut c_void,
            rows,
            cols,
            eps as f32,
            x_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_rmsnorm_residual_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
