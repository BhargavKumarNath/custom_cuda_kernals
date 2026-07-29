"""Correctness tests for Kernel 11's CUDA implementation
(csrc/kernels/viterbi.cu, via custom_cuda.kernels.viterbi) against the
PyTorch eager baseline. Mirrors tests/test_graph_message_passing_kernel.py.
"""

from __future__ import annotations

import pytest
import torch
from baselines.viterbi import ALL_CASES, ViterbiCase, eager_viterbi, make_inputs, path_log_prob
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Viterbi kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.viterbi import viterbi_decode  # noqa: E402

TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-4, atol=1e-3),
    torch.float16: dict(rtol=1e-2, atol=5e-2),
    torch.bfloat16: dict(rtol=1e-2, atol=1e-1),
}
OUTLIER_FRACTION: dict[torch.dtype, float] = {
    torch.float32: 0.0,
    torch.float16: 0.0,
    torch.bfloat16: DEFAULT_BF16_OUTLIER_FRACTION,
}


def _case_id(case: ViterbiCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager_score(case: ViterbiCase):
    log_emission, log_trans, log_pi = make_inputs(case)
    path_eager, score_eager = eager_viterbi(log_emission, log_trans, log_pi)
    path_kernel, score_kernel = viterbi_decode(log_emission, log_trans, log_pi)

    assert path_kernel.shape == (case.batch, case.seq_len)
    assert score_kernel.shape == (case.batch,)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(score_kernel, score_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_path_is_valid_maximizer(case: ViterbiCase):
    log_emission, log_trans, log_pi = make_inputs(case)
    path_kernel, score_kernel = viterbi_decode(log_emission, log_trans, log_pi)
    recomputed = path_log_prob(path_kernel, log_emission, log_trans, log_pi)
    torch.testing.assert_close(recomputed, score_kernel, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_kernel_peaked_case_exact_path_match(dtype: torch.dtype):
    case = ViterbiCase("peaked_check", batch=4, seq_len=50, num_states=8, dtype=dtype, peaked=True)
    log_emission, log_trans, log_pi = make_inputs(case)
    path, _score = viterbi_decode(log_emission, log_trans, log_pi)

    expected = torch.arange(case.seq_len, device=path.device) % case.num_states
    expected = expected.unsqueeze(0).expand(case.batch, -1)
    torch.testing.assert_close(path, expected)


def test_kernel_t_eq_1_matches_direct_argmax():
    case = ViterbiCase("t1_check", batch=16, seq_len=1, num_states=8, dtype=torch.float32)
    log_emission, log_trans, log_pi = make_inputs(case)
    path, score = viterbi_decode(log_emission, log_trans, log_pi)

    expected_delta = log_pi.unsqueeze(0) + log_emission[:, 0, :]
    expected_score, expected_state = expected_delta.max(dim=1)
    torch.testing.assert_close(path[:, 0], expected_state)
    torch.testing.assert_close(score, expected_score, rtol=1e-5, atol=1e-5)


def test_kernel_rejects_cpu_tensor():
    log_emission = torch.randn(4, 8, 8)
    log_trans = torch.randn(8, 8, device="cuda")
    log_pi = torch.randn(8, device="cuda")
    psi = torch.zeros(4, 8, 8, dtype=torch.long, device="cuda")
    best_path = torch.zeros(4, 8, dtype=torch.long, device="cuda")
    best_score = torch.empty(4, device="cuda")
    from custom_cuda import _native

    with pytest.raises(ValueError):
        _native.viterbi_fwd(log_emission, log_trans, log_pi, psi, best_path, best_score)


def test_kernel_rejects_num_states_over_1024():
    device = "cuda"
    batch, seq_len, num_states = 1, 4, 1025
    log_emission = torch.randn(batch, seq_len, num_states, device=device)
    log_trans = torch.randn(num_states, num_states, device=device)
    log_pi = torch.randn(num_states, device=device)
    with pytest.raises(RuntimeError):
        viterbi_decode(log_emission, log_trans, log_pi)
