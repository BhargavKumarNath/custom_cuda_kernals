//! CFFI-boundary error types (project_plan.md Section 1.2): validation
//! failures and CUDA runtime errors are surfaced to Python as `PyErr`
//! rather than propagating a raw device pointer into undefined behavior.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("argument '{0}' must be a CUDA tensor, got device type '{1}'")]
    NotCuda(&'static str, String),
    #[error("argument '{0}' must be contiguous")]
    NotContiguous(&'static str),
    #[error("unsupported dtype '{0}'; expected float32, float16, or bfloat16")]
    UnsupportedDtype(String),
    #[error("dtype mismatch: '{0}' is {1:?} but expected {2:?} (must match 'x')")]
    DtypeMismatch(&'static str, KernelDtypeDebug, KernelDtypeDebug),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("CUDA error: {0}")]
    Cuda(String),
}

// Thin wrapper so KernelError doesn't need a direct dependency on
// crate::kernels::dtype from this module's #[error(...)] format strings.
pub type KernelDtypeDebug = crate::kernels::dtype::KernelDType;

impl From<KernelError> for PyErr {
    fn from(e: KernelError) -> PyErr {
        match &e {
            KernelError::Cuda(_) => PyRuntimeError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        }
    }
}
