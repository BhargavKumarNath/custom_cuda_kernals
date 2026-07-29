"""Python entrypoint for Kernel 4 (Fused Linear Cross Entropy Loss).

Orchestrates the vocab-dimension chunking loop: each chunk's logits
(`hidden @ weight_chunk.T`) are computed via PyTorch/cuBLAS — never more
than one `[N, chunk_v]` chunk materialized at a time — and
`custom_cuda._native.linear_ce_chunk_update` fuses that chunk's
online-softmax update into the running per-token accumulators. See
`baselines/linear_cross_entropy.py` for the reference semantics and the
documented forward-only scope.

**No gradients**: `_native.linear_ce_chunk_update` is a raw PyO3 call, not
a `torch.autograd.Function` — it is opaque to autograd, so calling
`.backward()` on the returned loss will not produce correct (or any)
gradients for `hidden`/`weight`. This function is for inference / eval-time
loss or perplexity computation, not a training-loop drop-in replacement for
`F.cross_entropy`. A real training-capable version would need a fused
backward kernel computing `d_hidden`/`d_weight` in the same chunked
structure — out of scope here (see the baselines module docstring).
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["linear_cross_entropy", "DEFAULT_CHUNK_SIZE"]

DEFAULT_CHUNK_SIZE = 4096
IGNORE_INDEX = -100


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """`loss = cross_entropy(hidden @ weight.T, targets)`, computed without
    ever materializing the full `[N, V]` logits tensor.

    hidden: `[N, H]`, weight: `[V, H]` (no bias), targets: `[N]` int64.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")

    n, _ = hidden.shape
    vocab_size, _ = weight.shape
    device = hidden.device

    running_max = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    running_sum = torch.zeros(n, device=device, dtype=torch.float32)
    target_logit = torch.zeros(n, device=device, dtype=torch.float32)

    for v_start in range(0, vocab_size, chunk_size):
        v_end = min(v_start + chunk_size, vocab_size)
        weight_chunk = weight[v_start:v_end]
        # Native dtype (fp32/fp16/bf16) — no upcast copy. The kernel reads
        # elements directly at this dtype (see csrc/kernels/
        # linear_cross_entropy.cu's design-history comment: an earlier
        # version required a `.float()` cast here, which benchmarking
        # showed cost a fixed, chunk-size-independent overhead).
        logits_chunk = (hidden @ weight_chunk.T).contiguous()
        _native.linear_ce_chunk_update(
            logits_chunk, targets, v_start, running_max, running_sum, target_logit
        )

    loss_per_token = running_max + torch.log(running_sum) - target_logit
    valid_mask = targets != ignore_index
    loss_per_token = torch.where(valid_mask, loss_per_token, torch.zeros_like(loss_per_token))

    if reduction == "none":
        return loss_per_token
    if reduction == "sum":
        return loss_per_token.sum()
    # "mean": intentionally left as an unguarded 0/0 division when no token
    # is valid, to match F.cross_entropy's NaN behavior in that case rather
    # than silently returning 0 (see tests/test_linear_cross_entropy.py::
    # test_eager_all_ignored_mean_is_nan).
    count = valid_mask.sum().float()
    return loss_per_token.sum() / count
