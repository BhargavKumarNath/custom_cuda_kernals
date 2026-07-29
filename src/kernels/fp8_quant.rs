//! Rust/PyO3 bindings for Kernel 12 (FP8 Dynamic Quantization &
//! Casting). Wraps the `extern "C"` launchers declared in
//! csrc/includes/fp8_quant.h / defined in csrc/kernels/fp8_quant.cu.

use std::ffi::c_void;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int32_cuda_tensor};

extern "C" {
    fn launch_fp8_quant_block_fwd(
        x: *const c_void,
        x_fp8: *mut u8,
        scale: *mut f32,
        m: i64,
        n: i64,
        dtype: i32,
        fp8_format: i32,
        stream: *mut c_void,
    ) -> i32;

    fn launch_fp8_quant_tensor_fwd(
        x: *const c_void,
        x_fp8: *mut u8,
        scale: *mut f32,
        amax_scratch: *mut i32,
        m: i64,
        n: i64,
        dtype: i32,
        fp8_format: i32,
        stream: *mut c_void,
    ) -> i32;
}

fn parse_fp8_format(fp8_format: &str) -> PyResult<KernelDType> {
    match fp8_format {
        "e4m3" => Ok(KernelDType::F8E4M3),
        "e5m2" => Ok(KernelDType::F8E5M2),
        other => Err(PyValueError::new_err(format!(
            "fp8_format must be 'e4m3' or 'e5m2', got {other:?}"
        ))),
    }
}

fn div_ceil(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// `fp8_quant_block_fwd(x, x_fp8, scale, fp8_format)`
///
/// Block-wise (128x128 tile) dynamic FP8 quantization: one scale per
/// tile. `x`: `[M, N]` contiguous CUDA tensor (float32/float16/
/// bfloat16). `x_fp8`: `[M, N]`, dtype matching `fp8_format` ("e4m3" ->
/// `torch.float8_e4m3fn`, "e5m2" -> `torch.float8_e5m2`). `scale`:
/// `[ceil(M/128), ceil(N/128)]` float32.
#[pyfunction]
pub fn fp8_quant_block_fwd(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    x_fp8: Bound<'_, PyAny>,
    scale: Bound<'_, PyAny>,
    fp8_format: &str,
) -> PyResult<()> {
    let fp8_dtype = parse_fp8_format(fp8_format)?;
    let x_view = validate_cuda_tensor("x", &x)?;
    let x_fp8_view = validate_cuda_tensor("x_fp8", &x_fp8)?;
    let scale_view = validate_cuda_tensor("scale", &scale)?;

    if x_fp8_view.dtype != fp8_dtype {
        return Err(KernelError::UnsupportedDtype(format!(
            "x_fp8 dtype {:?} does not match requested fp8_format {fp8_format:?}",
            x_fp8_view.dtype
        ))
        .into());
    }
    if scale_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "scale must be float32, got {:?}",
            scale_view.dtype
        ))
        .into());
    }
    if x_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!("x must be 2-D [M, N]; got {:?}", x_view.shape)).into());
    }
    let (m, n) = (x_view.shape[0], x_view.shape[1]);
    if x_fp8_view.shape != [m, n] {
        return Err(KernelError::ShapeMismatch(format!(
            "x_fp8 shape {:?} must be [{m}, {n}]",
            x_fp8_view.shape
        ))
        .into());
    }
    let (num_row_blocks, num_col_blocks) = (div_ceil(m, 128), div_ceil(n, 128));
    if scale_view.shape != [num_row_blocks, num_col_blocks] {
        return Err(KernelError::ShapeMismatch(format!(
            "scale shape {:?} must be [{num_row_blocks}, {num_col_blocks}] (ceil(M/128), ceil(N/128))",
            scale_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (x_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_fp8_quant_block_fwd(
            x_view.ptr as *const c_void,
            x_fp8_view.ptr as *mut u8,
            scale_view.ptr as *mut f32,
            m,
            n,
            x_view.dtype as i32,
            fp8_dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_fp8_quant_block_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}

/// `fp8_quant_tensor_fwd(x, x_fp8, scale, amax_scratch, fp8_format)`
///
/// Tensor-wide dynamic FP8 quantization: one scale for the whole
/// matrix. `amax_scratch`: `[1]` int32, caller-zeroed. `scale`: `[1]`
/// (or `[]`) float32. Other arguments as `fp8_quant_block_fwd`.
#[pyfunction]
pub fn fp8_quant_tensor_fwd(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    x_fp8: Bound<'_, PyAny>,
    scale: Bound<'_, PyAny>,
    amax_scratch: Bound<'_, PyAny>,
    fp8_format: &str,
) -> PyResult<()> {
    let fp8_dtype = parse_fp8_format(fp8_format)?;
    let x_view = validate_cuda_tensor("x", &x)?;
    let x_fp8_view = validate_cuda_tensor("x_fp8", &x_fp8)?;
    let scale_view = validate_cuda_tensor("scale", &scale)?;
    let amax_view = validate_int32_cuda_tensor("amax_scratch", &amax_scratch)?;

    if x_fp8_view.dtype != fp8_dtype {
        return Err(KernelError::UnsupportedDtype(format!(
            "x_fp8 dtype {:?} does not match requested fp8_format {fp8_format:?}",
            x_fp8_view.dtype
        ))
        .into());
    }
    if scale_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "scale must be float32, got {:?}",
            scale_view.dtype
        ))
        .into());
    }
    if x_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!("x must be 2-D [M, N]; got {:?}", x_view.shape)).into());
    }
    let (m, n) = (x_view.shape[0], x_view.shape[1]);
    if x_fp8_view.shape != [m, n] {
        return Err(KernelError::ShapeMismatch(format!(
            "x_fp8 shape {:?} must be [{m}, {n}]",
            x_fp8_view.shape
        ))
        .into());
    }
    if amax_view.shape.iter().product::<i64>() != 1 {
        return Err(KernelError::ShapeMismatch(format!(
            "amax_scratch must have exactly 1 element; got shape {:?}",
            amax_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (x_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_fp8_quant_tensor_fwd(
            x_view.ptr as *const c_void,
            x_fp8_view.ptr as *mut u8,
            scale_view.ptr as *mut f32,
            amax_view.ptr as *mut i32,
            m,
            n,
            x_view.dtype as i32,
            fp8_dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_fp8_quant_tensor_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
