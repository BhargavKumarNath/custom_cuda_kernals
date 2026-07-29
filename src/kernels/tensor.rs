//! Extracts and validates a raw device-pointer view of a `torch.Tensor`
//! passed in from Python, without copying any tensor data (project_plan.md
//! Section 1.4: raw `data_ptr()` pass-through path).

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;

pub struct CudaTensorView {
    pub ptr: u64,
    pub dtype: KernelDType,
    pub device_index: i64,
    pub shape: Vec<i64>,
}

/// A CUDA tensor view without a `KernelDType` (float32/float16/bfloat16) —
/// for index tensors like Kernel 4's `targets` (int64), which aren't a
/// compute dtype. `device_index` is kept for symmetry with
/// `CudaTensorView` and future callers that key the CUDA stream off an
/// index tensor rather than a compute tensor; current callers derive the
/// stream from a compute tensor instead.
#[allow(dead_code)]
pub struct IndexTensorView {
    pub ptr: u64,
    pub device_index: i64,
    pub shape: Vec<i64>,
}

struct RawTensorView {
    ptr: u64,
    device_index: i64,
    shape: Vec<i64>,
    dtype_str: String,
}

fn validate_and_extract(name: &'static str, obj: &Bound<'_, PyAny>) -> PyResult<RawTensorView> {
    let device = obj.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;
    if device_type != "cuda" {
        return Err(KernelError::NotCuda(name, device_type).into());
    }

    let is_contiguous: bool = obj.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(KernelError::NotContiguous(name).into());
    }

    let dtype_str: String = obj.getattr("dtype")?.str()?.extract()?;
    let ptr: u64 = obj.call_method0("data_ptr")?.extract()?;
    let device_index: i64 = device.getattr("index")?.extract()?;
    let shape: Vec<i64> = obj.getattr("shape")?.extract()?;

    Ok(RawTensorView {
        ptr,
        device_index,
        shape,
        dtype_str,
    })
}

pub fn validate_cuda_tensor(
    name: &'static str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<CudaTensorView> {
    let raw = validate_and_extract(name, obj)?;
    let dtype = KernelDType::from_torch_dtype_str(&raw.dtype_str)
        .ok_or(KernelError::UnsupportedDtype(raw.dtype_str))?;

    Ok(CudaTensorView {
        ptr: raw.ptr,
        dtype,
        device_index: raw.device_index,
        shape: raw.shape,
    })
}

pub fn validate_int64_cuda_tensor(
    name: &'static str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<IndexTensorView> {
    let raw = validate_and_extract(name, obj)?;
    if raw.dtype_str != "torch.int64" {
        return Err(KernelError::UnsupportedDtype(format!(
            "{name} must be int64, got {}",
            raw.dtype_str
        ))
        .into());
    }

    Ok(IndexTensorView {
        ptr: raw.ptr,
        device_index: raw.device_index,
        shape: raw.shape,
    })
}

/// For Kernel 12's `amax_scratch` — a small int32 CUDA scratch buffer,
/// not an index tensor, but validated the same way (raw dtype-string
/// check rather than a `KernelDType` tag, since int32 isn't a compute
/// dtype any kernel's `dispatch<T>()` templates on).
pub fn validate_int32_cuda_tensor(
    name: &'static str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<IndexTensorView> {
    let raw = validate_and_extract(name, obj)?;
    if raw.dtype_str != "torch.int32" {
        return Err(KernelError::UnsupportedDtype(format!(
            "{name} must be int32, got {}",
            raw.dtype_str
        ))
        .into());
    }

    Ok(IndexTensorView {
        ptr: raw.ptr,
        device_index: raw.device_index,
        shape: raw.shape,
    })
}
