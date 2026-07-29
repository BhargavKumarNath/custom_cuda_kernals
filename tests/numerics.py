"""Shared numerical-comparison helpers for kernel correctness tests.

Low-precision dtypes (bf16 has only 8 significand bits) mean that, at
million-element scale, a handful of values will sit exactly on a rounding
boundary between two adjacent representable values. Two equally valid
computation orders — e.g. an fp64 ground truth rounded directly to bf16 vs.
an fp32 computation then rounded to bf16 (double rounding) — can tip such a
boundary element to the adjacent representable bin. That is expected
floating-point behavior, not a correctness bug (see project_plan.md
Section 4.1), so comparisons for these dtypes tolerate a tiny fraction of
such outliers instead of inflating rtol/atol to cover the single worst
element project-wide.

fp32 has 23 mantissa bits and is not expected to hit this in practice; its
tests should stay strict (`max_outlier_fraction=0`, the default).
"""

from __future__ import annotations

import torch

# Elements allowed to exceed (atol + rtol * |expected|) before a mismatch is
# treated as a real failure rather than rounding-boundary noise. 1e-5 permits
# ~1-2 outliers per 100K-1M elements, well above what boundary rounding
# produces in practice, while still failing hard on any systematic error.
DEFAULT_BF16_OUTLIER_FRACTION = 1e-5


def assert_close_with_outliers(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_outlier_fraction: float = 0.0,
) -> None:
    """`torch.testing.assert_close` that tolerates up to
    `max_outlier_fraction` of elements exceeding the rtol/atol bound.

    Falls through to a plain `torch.testing.assert_close` call (for a
    correctly-formatted failure message) whenever the outlier budget is
    exceeded, so a real correctness bug still fails loudly.
    """
    if max_outlier_fraction <= 0:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return

    a = actual.double()
    e = expected.double()
    diff = (a - e).abs()
    bound = atol + rtol * e.abs()
    violations = diff > bound
    fraction = violations.float().mean().item()

    if fraction == 0.0:
        return
    if fraction > max_outlier_fraction:
        # Re-run through assert_close purely to get its structured diff
        # output in the failure message.
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
