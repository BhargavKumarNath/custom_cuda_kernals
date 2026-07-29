//! Rust/PyO3 binding for Kernel 9 (Block Pairwise Distance Matrix). Wraps
//! the `extern "C"` launcher declared in
//! csrc/includes/pairwise_distance.h / defined in
//! csrc/kernels/pairwise_distance.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::validate_cuda_tensor;

extern "C" {
    fn launch_pairwise_distance_fwd(
        a: *const c_void,
        b: *const c_void,
        a_norm_sq: *const f32,
        b_norm_sq: *const f32,
        dist_sq: *mut f32,
        m: i64,
        n: i64,
        dim: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `pairwise_distance_fwd(a, b, a_norm_sq, b_norm_sq, dist_sq)`
///
/// Computes `dist_sq[i,j] = max(||a[i]||^2 + ||b[j]||^2 - 2*dot(a[i],
/// b[j]), 0)` in place into the caller-provided `dist_sq` output tensor.
/// `a`: `[M, Dim]`, `b`: `[N, Dim]`, contiguous CUDA tensors sharing one
/// dtype. `a_norm_sq`: `[M]`, `b_norm_sq`: `[N]`, both float32
/// (precomputed by the Python wrapper). `dist_sq`: `[M, N]` float32.
#[pyfunction]
pub fn pairwise_distance_fwd(
    py: Python<'_>,
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    a_norm_sq: Bound<'_, PyAny>,
    b_norm_sq: Bound<'_, PyAny>,
    dist_sq: Bound<'_, PyAny>,
) -> PyResult<()> {
    let a_view = validate_cuda_tensor("a", &a)?;
    let b_view = validate_cuda_tensor("b", &b)?;
    let a_norm_view = validate_cuda_tensor("a_norm_sq", &a_norm_sq)?;
    let b_norm_view = validate_cuda_tensor("b_norm_sq", &b_norm_sq)?;
    let dist_view = validate_cuda_tensor("dist_sq", &dist_sq)?;

    if b_view.dtype != a_view.dtype {
        return Err(KernelError::DtypeMismatch("b", b_view.dtype, a_view.dtype).into());
    }
    for (name, v) in [("a_norm_sq", &a_norm_view), ("b_norm_sq", &b_norm_view), ("dist_sq", &dist_view)] {
        if v.dtype != KernelDType::F32 {
            return Err(KernelError::UnsupportedDtype(format!("{name} must be float32, got {:?}", v.dtype)).into());
        }
    }

    if a_view.shape.len() != 2 || b_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "a and b must be 2-D; got a={:?}, b={:?}",
            a_view.shape, b_view.shape
        ))
        .into());
    }
    let (m, dim) = (a_view.shape[0], a_view.shape[1]);
    let (n, b_dim) = (b_view.shape[0], b_view.shape[1]);
    if b_dim != dim {
        return Err(KernelError::ShapeMismatch(format!(
            "b's dim ({b_dim}) must match a's dim ({dim})"
        ))
        .into());
    }
    if a_norm_view.shape != [m] {
        return Err(KernelError::ShapeMismatch(format!(
            "a_norm_sq shape {:?} must be [{m}]",
            a_norm_view.shape
        ))
        .into());
    }
    if b_norm_view.shape != [n] {
        return Err(KernelError::ShapeMismatch(format!(
            "b_norm_sq shape {:?} must be [{n}]",
            b_norm_view.shape
        ))
        .into());
    }
    if dist_view.shape != [m, n] {
        return Err(KernelError::ShapeMismatch(format!(
            "dist_sq shape {:?} must be [{m}, {n}]",
            dist_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (a_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_pairwise_distance_fwd(
            a_view.ptr as *const c_void,
            b_view.ptr as *const c_void,
            a_norm_view.ptr as *const f32,
            b_norm_view.ptr as *const f32,
            dist_view.ptr as *mut f32,
            m,
            n,
            dim,
            a_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_pairwise_distance_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
