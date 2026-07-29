//! Rust/PyO3 binding for Kernel 3 (Fused RoPE, half-split convention).
//! Wraps the `extern "C"` launcher declared in csrc/includes/rope.h /
//! defined in csrc/kernels/rope.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::validate_cuda_tensor;

extern "C" {
    fn launch_rope_fwd(
        q: *const c_void,
        k: *const c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        batch: i64,
        seq_len: i64,
        n_q_heads: i64,
        n_kv_heads: i64,
        head_dim: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `rope_fwd(q, k, cos, sin, q_out, k_out)`
///
/// Rotates `q: [B, S, Hq, D]` and `k: [B, S, Hkv, D]` in place into the
/// caller-provided `q_out`/`k_out` using the half-split RoPE convention,
/// given precomputed `cos`/`sin: [S, D/2]` tables (always float32). q, k,
/// q_out, k_out must be contiguous CUDA tensors sharing one dtype
/// (float32, float16, or bfloat16); cos/sin must be contiguous float32
/// CUDA tensors. D must be even.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rope_fwd(
    py: Python<'_>,
    q: Bound<'_, PyAny>,
    k: Bound<'_, PyAny>,
    cos: Bound<'_, PyAny>,
    sin: Bound<'_, PyAny>,
    q_out: Bound<'_, PyAny>,
    k_out: Bound<'_, PyAny>,
) -> PyResult<()> {
    let q_view = validate_cuda_tensor("q", &q)?;
    let k_view = validate_cuda_tensor("k", &k)?;
    let cos_view = validate_cuda_tensor("cos", &cos)?;
    let sin_view = validate_cuda_tensor("sin", &sin)?;
    let q_out_view = validate_cuda_tensor("q_out", &q_out)?;
    let k_out_view = validate_cuda_tensor("k_out", &k_out)?;

    for (name, v) in [("k", &k_view), ("q_out", &q_out_view)] {
        if v.dtype != q_view.dtype {
            return Err(KernelError::DtypeMismatch(name, v.dtype, q_view.dtype).into());
        }
    }
    if k_out_view.dtype != k_view.dtype {
        return Err(KernelError::DtypeMismatch("k_out", k_out_view.dtype, k_view.dtype).into());
    }
    if cos_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "cos must be float32 (RoPE tables are always fp32), got {:?}",
            cos_view.dtype
        ))
        .into());
    }
    if sin_view.dtype != KernelDType::F32 {
        return Err(KernelError::UnsupportedDtype(format!(
            "sin must be float32 (RoPE tables are always fp32), got {:?}",
            sin_view.dtype
        ))
        .into());
    }

    if q_view.shape.len() != 4 {
        return Err(KernelError::ShapeMismatch(format!(
            "q must be 4-D [batch, seq_len, n_q_heads, head_dim], got {:?}",
            q_view.shape
        ))
        .into());
    }
    if k_view.shape.len() != 4 {
        return Err(KernelError::ShapeMismatch(format!(
            "k must be 4-D [batch, seq_len, n_kv_heads, head_dim], got {:?}",
            k_view.shape
        ))
        .into());
    }
    if q_out_view.shape != q_view.shape {
        return Err(KernelError::ShapeMismatch("q_out must match q's shape".to_string()).into());
    }
    if k_out_view.shape != k_view.shape {
        return Err(KernelError::ShapeMismatch("k_out must match k's shape".to_string()).into());
    }

    let (batch, seq_len, n_q_heads, head_dim) =
        (q_view.shape[0], q_view.shape[1], q_view.shape[2], q_view.shape[3]);
    let (k_batch, k_seq_len, n_kv_heads, k_head_dim) =
        (k_view.shape[0], k_view.shape[1], k_view.shape[2], k_view.shape[3]);
    if (batch, seq_len, head_dim) != (k_batch, k_seq_len, k_head_dim) {
        return Err(KernelError::ShapeMismatch(format!(
            "q and k must share batch/seq_len/head_dim; got q={:?}, k={:?}",
            q_view.shape, k_view.shape
        ))
        .into());
    }
    if head_dim % 2 != 0 {
        return Err(KernelError::ShapeMismatch(format!("head_dim must be even, got {head_dim}")).into());
    }
    let half_dim = head_dim / 2;
    if cos_view.shape != [seq_len, half_dim] || sin_view.shape != [seq_len, half_dim] {
        return Err(KernelError::ShapeMismatch(format!(
            "cos/sin must have shape [{seq_len}, {half_dim}]; got cos={:?}, sin={:?}",
            cos_view.shape, sin_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (q_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_rope_fwd(
            q_view.ptr as *const c_void,
            k_view.ptr as *const c_void,
            cos_view.ptr as *const c_void,
            sin_view.ptr as *const c_void,
            q_out_view.ptr as *mut c_void,
            k_out_view.ptr as *mut c_void,
            batch,
            seq_len,
            n_q_heads,
            n_kv_heads,
            head_dim,
            q_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_rope_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
