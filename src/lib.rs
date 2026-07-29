//! `custom_cuda._native` — PyO3 extension module root.
//!
//! Each kernel's Rust wrapper lives in its own module under `src/kernels/`
//! (see project_plan.md Section 1 and Section 7) and is mounted onto this
//! module as it is implemented.

use pyo3::prelude::*;

mod kernels;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernels::rmsnorm_residual::rmsnorm_residual_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::swiglu::swiglu_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::rope::rope_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::linear_cross_entropy::linear_ce_chunk_update, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::matmul_add_bias::matmul_add_bias_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::moe_router::moe_router_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::token_permute::token_gather_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::token_permute::token_combine_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::cosine_topk::cosine_topk_partial_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::cosine_topk::cosine_topk_merge_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::pairwise_distance::pairwise_distance_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::graph_message_passing::graph_message_passing_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::viterbi::viterbi_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::fp8_quant::fp8_quant_block_fwd, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::fp8_quant::fp8_quant_tensor_fwd, m)?)?;
    Ok(())
}
