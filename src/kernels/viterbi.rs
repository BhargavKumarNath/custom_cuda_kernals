//! Rust/PyO3 binding for Kernel 11 (Parallel Viterbi Algorithm). Wraps
//! the `extern "C"` launcher declared in csrc/includes/viterbi.h /
//! defined in csrc/kernels/viterbi.cu.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernels::dtype::KernelDType;
use crate::kernels::error::KernelError;
use crate::kernels::tensor::{validate_cuda_tensor, validate_int64_cuda_tensor};

extern "C" {
    fn launch_viterbi_fwd(
        log_emission: *const c_void,
        log_trans: *const f32,
        log_pi: *const f32,
        psi: *mut i64,
        best_path: *mut i64,
        best_score: *mut f32,
        batch: i64,
        seq_len: i64,
        num_states: i64,
        dtype: i32,
        stream: *mut c_void,
    ) -> i32; // cudaError_t; 0 == cudaSuccess
}

/// `viterbi_fwd(log_emission, log_trans, log_pi, psi, best_path, best_score)`
///
/// Batched Viterbi decoding of a single HMM shared across the batch.
/// `log_emission`: `[B, T, S]`, contiguous CUDA tensor (float32/float16/
/// bfloat16). `log_trans`: `[S, S]` float32. `log_pi`: `[S]` float32.
/// `psi`: `[B, T, S]` int64 scratch (caller-allocated). `best_path`:
/// `[B, T]` int64. `best_score`: `[B]` float32.
#[pyfunction]
pub fn viterbi_fwd(
    py: Python<'_>,
    log_emission: Bound<'_, PyAny>,
    log_trans: Bound<'_, PyAny>,
    log_pi: Bound<'_, PyAny>,
    psi: Bound<'_, PyAny>,
    best_path: Bound<'_, PyAny>,
    best_score: Bound<'_, PyAny>,
) -> PyResult<()> {
    let emission_view = validate_cuda_tensor("log_emission", &log_emission)?;
    let trans_view = validate_cuda_tensor("log_trans", &log_trans)?;
    let pi_view = validate_cuda_tensor("log_pi", &log_pi)?;
    let score_view = validate_cuda_tensor("best_score", &best_score)?;
    let psi_view = validate_int64_cuda_tensor("psi", &psi)?;
    let path_view = validate_int64_cuda_tensor("best_path", &best_path)?;

    for (name, v) in [("log_trans", &trans_view), ("log_pi", &pi_view), ("best_score", &score_view)] {
        if v.dtype != KernelDType::F32 {
            return Err(KernelError::UnsupportedDtype(format!("{name} must be float32, got {:?}", v.dtype)).into());
        }
    }

    if emission_view.shape.len() != 3 {
        return Err(KernelError::ShapeMismatch(format!(
            "log_emission must be 3-D [B, T, S]; got {:?}",
            emission_view.shape
        ))
        .into());
    }
    let (batch, seq_len, num_states) = (emission_view.shape[0], emission_view.shape[1], emission_view.shape[2]);

    if trans_view.shape != [num_states, num_states] {
        return Err(KernelError::ShapeMismatch(format!(
            "log_trans shape {:?} must be [{num_states}, {num_states}]",
            trans_view.shape
        ))
        .into());
    }
    if pi_view.shape != [num_states] {
        return Err(KernelError::ShapeMismatch(format!(
            "log_pi shape {:?} must be [{num_states}]",
            pi_view.shape
        ))
        .into());
    }
    if psi_view.shape != [batch, seq_len, num_states] {
        return Err(KernelError::ShapeMismatch(format!(
            "psi shape {:?} must be [{batch}, {seq_len}, {num_states}]",
            psi_view.shape
        ))
        .into());
    }
    if path_view.shape != [batch, seq_len] {
        return Err(KernelError::ShapeMismatch(format!(
            "best_path shape {:?} must be [{batch}, {seq_len}]",
            path_view.shape
        ))
        .into());
    }
    if score_view.shape != [batch] {
        return Err(KernelError::ShapeMismatch(format!(
            "best_score shape {:?} must be [{batch}]",
            score_view.shape
        ))
        .into());
    }

    let torch = py.import("torch")?;
    let stream_obj = torch
        .getattr("cuda")?
        .call_method1("current_stream", (emission_view.device_index,))?;
    let stream_ptr: u64 = stream_obj.getattr("cuda_stream")?.extract()?;

    let err = unsafe {
        launch_viterbi_fwd(
            emission_view.ptr as *const c_void,
            trans_view.ptr as *const f32,
            pi_view.ptr as *const f32,
            psi_view.ptr as *mut i64,
            path_view.ptr as *mut i64,
            score_view.ptr as *mut f32,
            batch,
            seq_len,
            num_states,
            emission_view.dtype as i32,
            stream_ptr as *mut c_void,
        )
    };

    if err != 0 {
        return Err(KernelError::Cuda(format!("launch_viterbi_fwd failed, cudaError_t = {err}")).into());
    }
    Ok(())
}
