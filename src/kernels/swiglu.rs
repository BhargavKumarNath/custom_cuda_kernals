//! Rust/PyO3 binding for Kernel 2 (Fused SwiGLU Gated Activation). Wraps
//! the `extern "C"` launcher declared in csrc/includes/swiglu.h / defined
//! in csrc/kernels/swiglu.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::error::KernelError;
use crate::kernels::tensor::validate_cuda_tensor;

extern "C" {
    fn launch_swiglu_fwd(
        gate: *const c_void,
        up: *const c_void,
        y: *mut c_void,
        n: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `swiglu_fwd(gate, up, y)`
///
/// Computes `y = SiLU(gate) * up` elementwise in place into the
/// caller-provided `y` output tensor. `gate`, `up`, `y` must share shape,
/// be contiguous CUDA tensors of the same dtype (float32, float16, or
/// bfloat16).
#[pyfunction]
pub fn swiglu_fwd(
    py: Python<'_>,
    gate: Bound<'_, PyAny>,
    up: Bound<'_, PyAny>,
    y: Bound<'_, PyAny>,
) -> PyResult<()> {
    let gate_view = validate_cuda_tensor("gate", &gate)?;
    let up_view = validate_cuda_tensor("up", &up)?;
    let y_view = validate_cuda_tensor("y", &y)?;

    if up_view.dtype != gate_view.dtype {
        return Err(KernelError::DtypeMismatch("up", up_view.dtype, gate_view.dtype).into());
    }
    if y_view.dtype != gate_view.dtype {
        return Err(KernelError::DtypeMismatch("y", y_view.dtype, gate_view.dtype).into());
    }

    if up_view.shape != gate_view.shape || y_view.shape != gate_view.shape {
        return Err(KernelError::ShapeMismatch(format!(
            "gate/up/y must share the same shape; got gate={:?}, up={:?}, y={:?}",
            gate_view.shape, up_view.shape, y_view.shape
        ))
        .into());
    }

    let n: i64 = gate_view.shape.iter().product();

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (gate_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_swiglu_fwd(
            gate_view.ptr as *const c_void,
            up_view.ptr as *const c_void,
            y_view.ptr as *mut c_void,
            n,
            gate_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_swiglu_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
