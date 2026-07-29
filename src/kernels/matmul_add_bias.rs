//! Rust/PyO3 binding for Kernel 5 (Fused MatMul + Add Bias). Wraps the
//! `extern "C"` launcher declared in csrc/includes/matmul_add_bias.h /
//! defined in csrc/kernels/matmul_add_bias.cu.

use std::ffi::c_void;
use std::ptr;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::error::KernelError;
use crate::kernels::tensor::validate_cuda_tensor;

extern "C" {
    fn launch_matmul_add_bias_fwd(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        y: *mut c_void,
        m: i64,
        k: i64,
        n: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `matmul_add_bias_fwd(x, weight, bias, y)`
///
/// Computes `y = x @ weight.T + bias` (bias optional — pass `None` to
/// omit) in place into the caller-provided `y` output tensor. `x`:
/// `[M, K]`, `weight`: `[N, K]`, `bias`: `[N]` or `None`, `y`: `[M, N]`.
/// All must be contiguous CUDA tensors sharing one dtype (float32,
/// float16, or bfloat16).
#[pyfunction]
pub fn matmul_add_bias_fwd(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    weight: Bound<'_, PyAny>,
    bias: Option<Bound<'_, PyAny>>,
    y: Bound<'_, PyAny>,
) -> PyResult<()> {
    let x_view = validate_cuda_tensor("x", &x)?;
    let weight_view = validate_cuda_tensor("weight", &weight)?;
    let y_view = validate_cuda_tensor("y", &y)?;
    let bias_view = bias.as_ref().map(|b| validate_cuda_tensor("bias", b)).transpose()?;

    for (name, v) in [("weight", &weight_view), ("y", &y_view)] {
        if v.dtype != x_view.dtype {
            return Err(KernelError::DtypeMismatch(name, v.dtype, x_view.dtype).into());
        }
    }
    if let Some(bv) = &bias_view {
        if bv.dtype != x_view.dtype {
            return Err(KernelError::DtypeMismatch("bias", bv.dtype, x_view.dtype).into());
        }
    }

    if x_view.shape.len() != 2 || weight_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "x and weight must be 2-D; got x={:?}, weight={:?}",
            x_view.shape, weight_view.shape
        ))
        .into());
    }
    let (m, k) = (x_view.shape[0], x_view.shape[1]);
    let (n, weight_k) = (weight_view.shape[0], weight_view.shape[1]);
    if weight_k != k {
        return Err(KernelError::ShapeMismatch(format!(
            "weight's in_features ({weight_k}) must match x's last dim ({k})"
        ))
        .into());
    }
    if y_view.shape != [m, n] {
        return Err(KernelError::ShapeMismatch(format!(
            "y shape {:?} must be [{m}, {n}]",
            y_view.shape
        ))
        .into());
    }
    if let Some(bv) = &bias_view {
        if bv.shape != [n] {
            return Err(KernelError::ShapeMismatch(format!(
                "bias shape {:?} must be [{n}]",
                bv.shape
            ))
            .into());
        }
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (x_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let bias_ptr: *const c_void = match &bias_view {
        Some(bv) => bv.ptr as *const c_void,
        None => ptr::null(),
    };

    let err = unsafe {
        launch_matmul_add_bias_fwd(
            x_view.ptr as *const c_void,
            weight_view.ptr as *const c_void,
            bias_ptr,
            y_view.ptr as *mut c_void,
            m,
            k,
            n,
            x_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(
            KernelError::Cuda(format!("launch_matmul_add_bias_fwd failed, cudaError_t = {err}")).into(),
        );
    }
    Ok(())
}
