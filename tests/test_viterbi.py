"""Correctness tests for Kernel 11 (Parallel Viterbi Algorithm) baselines.
Mirrors tests/test_graph_message_passing.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.viterbi import (
    ALL_CASES,
    STANDARD_CASES,
    ViterbiCase,
    compiled_viterbi,
    eager_viterbi,
    make_inputs,
    path_log_prob,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Viterbi baseline tests", allow_module_level=True)

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
def test_eager_matches_fp64_reference_score(case: ViterbiCase):
    log_emission, log_trans, log_pi = make_inputs(case)
    path, score = eager_viterbi(log_emission, log_trans, log_pi)
    _, score_ref = reference_fp64(log_emission, log_trans, log_pi)

    assert path.shape == (case.batch, case.seq_len)
    assert score.shape == (case.batch,)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(score, score_ref.to(torch.float32), max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_path_is_valid_maximizer(case: ViterbiCase):
    """The decoded path, rescored under the same inputs, must reproduce
    the claimed best_score exactly (up to fp roundoff) — true regardless
    of which specific path a tie-break convention picks.
    """
    log_emission, log_trans, log_pi = make_inputs(case)
    path, score = eager_viterbi(log_emission, log_trans, log_pi)
    recomputed = path_log_prob(path, log_emission, log_trans, log_pi)
    torch.testing.assert_close(recomputed, score, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_eager_peaked_case_exact_path_match(dtype: torch.dtype):
    """With a near-deterministic transition matrix (state -> (state+1)%S
    overwhelmingly favored) and a peaked initial distribution (state 0),
    the best path has no meaningful ties and must exactly equal the
    forced cyclic sequence.
    """
    case = ViterbiCase("peaked_check", batch=4, seq_len=50, num_states=8, dtype=dtype, peaked=True)
    log_emission, log_trans, log_pi = make_inputs(case)
    path, _score = eager_viterbi(log_emission, log_trans, log_pi)

    expected = torch.arange(case.seq_len, device=path.device) % case.num_states
    expected = expected.unsqueeze(0).expand(case.batch, -1)
    torch.testing.assert_close(path, expected)


def test_eager_t_eq_1_matches_direct_argmax():
    case = ViterbiCase("t1_check", batch=16, seq_len=1, num_states=8, dtype=torch.float32)
    log_emission, log_trans, log_pi = make_inputs(case)
    path, score = eager_viterbi(log_emission, log_trans, log_pi)

    expected_delta = log_pi.unsqueeze(0) + log_emission[:, 0, :]
    expected_score, expected_state = expected_delta.max(dim=1)
    torch.testing.assert_close(path[:, 0], expected_state)
    torch.testing.assert_close(score, expected_score, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: ViterbiCase):
    log_emission, log_trans, log_pi = make_inputs(case)
    _path_eager, score_eager = eager_viterbi(log_emission, log_trans, log_pi)
    _path_compiled, score_compiled = compiled_viterbi(log_emission, log_trans, log_pi)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(score_compiled, score_eager, max_outlier_fraction=outliers, **tol)
