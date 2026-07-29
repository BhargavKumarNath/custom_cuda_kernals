//! Rust/PyO3 bindings for Kernel 7 (Token Scatter/Gather, Permute-
//! Unpermute). Wraps the `extern "C"` launchers declared in
//! csrc/includes/token_permute.h / defined in csrc/kernels/token_permute.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_token_gather_fwd(
        src: *const c_void,
        indices: *const i64,
        dst: *mut c_void,
        n_dst_rows: i64,
        hidden_dim: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32;

    fn launch_token_combine_fwd(
        expert_output: *const c_void,
        unpermute_index: *const i64,
        weights: *const f32,
        combined: *mut c_void,
        n_tokens: i64,
        k: i64,
        hidden_dim: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// `token_gather_fwd(src, indices, dst)`
///
/// Computes `dst[i] = src[indices[i]]` (a row gather). `src`: `[S, H]`,
/// `indices`: `[N]` int64 (values in `[0, S)`), `dst`: `[N, H]`. `src`/
/// `dst` must be contiguous CUDA tensors sharing one dtype (float32,
/// float16, or bfloat16).
#[pyfunction]
pub fn token_gather_fwd(
    py: Python<'_>,
    src: Bound<'_, PyAny>,
    indices: Bound<'_, PyAny>,
    dst: Bound<'_, PyAny>,
) -> PyResult<()> {
    let src_view = validate_cuda_tensor("src", &src)?;
    let indices_view = validate_int64_cuda_tensor("indices", &indices)?;
    let dst_view = validate_cuda_tensor("dst", &dst)?;

    if dst_view.dtype != src_view.dtype {
        return Err(KernelError::DtypeMismatch("dst", dst_view.dtype, src_view.dtype).into());
    }
    if src_view.shape.len() != 2 || dst_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "src and dst must be 2-D; got src={:?}, dst={:?}",
            src_view.shape, dst_view.shape
        ))
        .into());
    }
    let hidden_dim = src_view.shape[1];
    if dst_view.shape[1] != hidden_dim {
        return Err(KernelError::ShapeMismatch(format!(
            "dst's hidden dim ({}) must match src's ({hidden_dim})",
            dst_view.shape[1]
        ))
        .into());
    }
    if indices_view.shape.len() != 1 || indices_view.shape[0] != dst_view.shape[0] {
        return Err(KernelError::ShapeMismatch(format!(
            "indices shape {:?} must be [{}] (dst's row count)",
            indices_view.shape, dst_view.shape[0]
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (src_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_token_gather_fwd(
            src_view.ptr as *const c_void,
            indices_view.ptr as *const i64,
            dst_view.ptr as *mut c_void,
            dst_view.shape[0],
            hidden_dim,
            src_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_token_gather_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}

/// `token_combine_fwd(expert_output, unpermute_index, weights, combined)`
///
/// Computes `combined[t] = sum_j weights[t,j] *
/// expert_output[unpermute_index[t,j]]`. `expert_output`: `[N, H]`,
/// `unpermute_index`: `[T, k]` int64 (values in `[0, N)`), `weights`:
/// `[T, k]` float32, `combined`: `[T, H]`. `expert_output`/`combined`
/// must be contiguous CUDA tensors sharing one dtype.
#[pyfunction]
pub fn token_combine_fwd(
    py: Python<'_>,
    expert_output: Bound<'_, PyAny>,
    unpermute_index: Bound<'_, PyAny>,
    weights: Bound<'_, PyAny>,
    combined: Bound<'_, PyAny>,
) -> PyResult<()> {
    let expert_view = validate_cuda_tensor("expert_output", &expert_output)?;
    let index_view = validate_int64_cuda_tensor("unpermute_index", &unpermute_index)?;
    let weights_view = validate_cuda_tensor("weights", &weights)?;
    let combined_view = validate_cuda_tensor("combined", &combined)?;

    if combined_view.dtype != expert_view.dtype {
        return Err(KernelError::DtypeMismatch("combined", combined_view.dtype, expert_view.dtype).into());
    }
    if weights_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "weights must be float32, got {:?}",
            weights_view.dtype
        ))
        .into());
    }

    if expert_view.shape.len() != 2 || combined_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "expert_output and combined must be 2-D; got expert_output={:?}, combined={:?}",
            expert_view.shape, combined_view.shape
        ))
        .into());
    }
    let hidden_dim = expert_view.shape[1];
    if combined_view.shape[1] != hidden_dim {
        return Err(KernelError::ShapeMismatch(format!(
            "combined's hidden dim ({}) must match expert_output's ({hidden_dim})",
            combined_view.shape[1]
        ))
        .into());
    }

    if index_view.shape.len() != 2 || weights_view.shape.len() != 2 {
        return Err(KernelError::ShapeMismatch(format!(
            "unpermute_index and weights must be 2-D [T, k]; got unpermute_index={:?}, weights={:?}",
            index_view.shape, weights_view.shape
        ))
        .into());
    }
    let (n_tokens, k) = (index_view.shape[0], index_view.shape[1]);
    if weights_view.shape != [n_tokens, k] {
        return Err(KernelError::ShapeMismatch(format!(
            "weights shape {:?} must match unpermute_index's [{n_tokens}, {k}]",
            weights_view.shape
        ))
        .into());
    }
    if combined_view.shape[0] != n_tokens {
        return Err(KernelError::ShapeMismatch(format!(
            "combined's row count ({}) must match unpermute_index's [{n_tokens}, ...]",
            combined_view.shape[0]
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (expert_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_token_combine_fwd(
            expert_view.ptr as *const c_void,
            index_view.ptr as *const i64,
            weights_view.ptr as *const f32,
            combined_view.ptr as *mut c_void,
            n_tokens,
            k,
            hidden_dim,
            expert_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_token_combine_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
