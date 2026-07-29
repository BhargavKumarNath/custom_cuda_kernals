"""Python entrypoint for Kernel 11 (Parallel Viterbi Algorithm).

Allocates the `psi` backpointer scratch buffer and output tensors, then
calls `custom_cuda._native.viterbi_fwd` — a single kernel launch that
internally loops over the entire sequence length (see
csrc/includes/viterbi.h's docstring). See
`baselines/viterbi.py::eager_viterbi` for the reference semantics this
must match.
"""

from __future__ import annotations

import torch

from custom_cuda import _native

__all__ = ["viterbi_decode"]


def viterbi_decode(
    log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Viterbi decoding of a single HMM shared across the batch.

    `log_emission`: `[B, T, S]`, contiguous CUDA tensor (float32/float16/
    bfloat16). `log_trans`: `[S, S]` float32. `log_pi`: `[S]` float32.
    Returns `(best_path, best_score)`: `best_path: [B, T]` int64,
    `best_score: [B]` float32.
    """
    batch, seq_len, num_states = log_emission.shape
    device = log_emission.device

    psi = torch.zeros(batch, seq_len, num_states, dtype=torch.long, device=device)
    best_path = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    best_score = torch.empty(batch, dtype=torch.float32, device=device)

    _native.viterbi_fwd(log_emission, log_trans, log_pi, psi, best_path, best_score)
    return best_path, best_score
