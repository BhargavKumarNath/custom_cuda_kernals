"""Baseline references for the Parallel Viterbi Algorithm (Kernel 11).

Scope: batched Viterbi decoding for a single (shared-across-the-batch)
Hidden Markov Model — matches project_plan.md Section 3.11's spec
exactly:

    delta[b, 0, s]   = log_pi[s] + log_emission[b, 0, s]
    delta[b, t, s]   = max_k(delta[b, t-1, k] + log_trans[k, s]) + log_emission[b, t, s]
    psi[b, t, s]     = argmax_k(delta[b, t-1, k] + log_trans[k, s])
    best_score[b]    = max_s delta[b, T-1, s]
    best_path[b, T-1] = argmax_s delta[b, T-1, s]
    best_path[b, t]   = psi[b, t+1, best_path[b, t+1]]   (backtrack, t = T-2..0)

`log_trans: [S, S]` and `log_pi: [S]` are shared by every sequence in the
batch (one trained HMM decoding many observation sequences — the common
batched-inference case, and the reason project_plan.md Section 3.11
calls for a single resident-in-shared-memory transition matrix, singular).
`log_emission: [B, T, S]` is the per-(batch, timestep, state) emission
log-probability, precomputed externally (e.g. by a Gaussian/neural
emission model) — decoding, not emission scoring, is this kernel's job,
the same scope boundary `hmmlearn`'s own `_do_viterbi_pass` draws
(it also takes precomputed `framelogprob`).

The recursion accumulates *additively* over up to thousands of
timesteps, so `delta` is always accumulated in fp32 regardless of
`log_emission`'s storage dtype (round-tripping each input value through
fp16/bf16 once is fine; compounding fp16 rounding error across a
thousand additions is not) — the same "accumulate in float regardless of
storage dtype" convention as every other kernel in this project.
`log_trans`/`log_pi` are always fp32 (tiny `[S,S]`/`[S]` arrays, not
worth quantizing — the same convention as Kernel 9's row norms and
Kernel 10's edge weights).

project_plan.md Section 3.11 names `hmmlearn` as the reference
implementation; it's a heavy, scikit-learn-based optional dependency
(not installed here — see project_build_environment notes), so rather
than requiring it, `reference_fp64` below implements the identical
textbook Viterbi DP directly (the same algorithm `hmmlearn` runs
internally, just in torch instead of Cython) — the same "build the
actual algorithm/primitive natively" choice Kernel 9 made with
`torch.cdist` and Kernel 10 made with `index_add_`.

Ties (two previous states giving the exact same `delta[t-1,k] +
log_trans[k,s]`) are a real possibility with structured/repeated
transition matrices; `torch.max`'s "first occurrence wins" tie-break is
used as the reference convention (see Kernel 6/8's tie-breaking lesson),
but correctness tests primarily check that the decoded path is a valid
maximizer (its own recomputed score equals `best_score`) rather than
requiring an exact backpointer match, since that's robust to legitimate
tie-breaking or fp16/bf16-rounding-induced differences between
implementations.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "ViterbiCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_viterbi",
    "compiled_viterbi",
    "reference_fp64",
    "path_log_prob",
]


@dataclasses.dataclass(frozen=True)
class ViterbiCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    batch: int
    seq_len: int
    num_states: int
    dtype: torch.dtype
    peaked: bool = False

    @property
    def emission_shape(self) -> tuple[int, int, int]:
        return (self.batch, self.seq_len, self.num_states)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, **kwargs) -> list[ViterbiCase]:
    return [ViterbiCase(f"{name}_{dt}".replace("torch.", ""), dtype=dt, **kwargs) for dt in _DTYPES]


# Representative speech/OCR/bioinformatics-decoding shapes: modest state
# counts, long sequences (Section 3.11's stated ">=512" success-criteria
# length, plus well beyond it).
STANDARD_CASES: list[ViterbiCase] = [
    *_cases_for("short_seq", batch=64, seq_len=64, num_states=16),
    *_cases_for("target_len", batch=64, seq_len=512, num_states=16),
    *_cases_for("long_seq", batch=32, seq_len=2048, num_states=32),
]

# Section 4.3-style edge-case battery, plus this kernel's own: varying
# sequence lengths (including the degenerate T=1 case), varying state
# counts (a single warp's worth exactly, a non-power-of-two count, and a
# count exceeding one warp), single-item batches, and a "peaked" (near-
# deterministic) transition matrix giving an unambiguous best path so
# exact-path-match tests are meaningful despite tie-breaking concerns
# elsewhere.
EDGE_CASES: list[ViterbiCase] = [
    *_cases_for("t_eq_1", batch=32, seq_len=1, num_states=8),
    *_cases_for("t_eq_2", batch=32, seq_len=2, num_states=8),
    *_cases_for("npot_seq_len", batch=16, seq_len=513, num_states=8),
    *_cases_for("single_batch", batch=1, seq_len=256, num_states=8),
    *_cases_for("s_eq_2", batch=16, seq_len=128, num_states=2),
    *_cases_for("s_eq_warp", batch=16, seq_len=128, num_states=32),
    *_cases_for("npot_states", batch=16, seq_len=128, num_states=100),
    *_cases_for("peaked_deterministic", batch=8, seq_len=256, num_states=8, peaked=True),
]

ALL_CASES: list[ViterbiCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: ViterbiCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(log_emission, log_trans, log_pi)`.

    `peaked=True` builds a near-deterministic transition matrix (heavily
    favoring `state -> (state+1) % S`) and a matching sharply-peaked
    initial distribution, so the best path is unambiguous (no
    meaningful ties) and can be exactly reconstructed by construction —
    used for exact-path correctness tests.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    b, t, s = case.batch, case.seq_len, case.num_states

    log_emission = (
        torch.log_softmax(torch.randn(case.emission_shape, dtype=torch.float32, device=device, generator=gen), dim=-1)
        .to(case.dtype)
    )

    if case.peaked:
        trans = torch.full((s, s), -20.0, device=device)
        for i in range(s):
            trans[i, (i + 1) % s] = 0.0
        log_trans = torch.log_softmax(trans, dim=-1)
        pi = torch.full((s,), -20.0, device=device)
        pi[0] = 0.0
        log_pi = torch.log_softmax(pi, dim=-1)
    else:
        log_trans = torch.log_softmax(
            torch.randn(s, s, dtype=torch.float32, device=device, generator=gen), dim=-1
        )
        log_pi = torch.log_softmax(torch.randn(s, dtype=torch.float32, device=device, generator=gen), dim=-1)

    return log_emission, log_trans, log_pi


def _viterbi_impl(
    log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor, acc_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, s = log_emission.shape
    emission = log_emission.to(acc_dtype)
    trans = log_trans.to(acc_dtype)
    pi = log_pi.to(acc_dtype)

    delta = pi.unsqueeze(0) + emission[:, 0, :]  # [B, S]
    if t == 1:
        best_score, last_state = delta.max(dim=1)
        return last_state.unsqueeze(1), best_score

    psi = torch.zeros(b, t, s, dtype=torch.long, device=log_emission.device)
    for step in range(1, t):
        scores = delta.unsqueeze(2) + trans.unsqueeze(0)  # [B, S_prev, S_cur]
        best_prev_scores, best_prev = scores.max(dim=1)  # [B, S_cur]
        delta = best_prev_scores + emission[:, step, :]
        psi[:, step, :] = best_prev

    best_score, last_state = delta.max(dim=1)  # [B]
    path = torch.zeros(b, t, dtype=torch.long, device=log_emission.device)
    path[:, t - 1] = last_state
    for step in range(t - 2, -1, -1):
        path[:, step] = psi[:, step + 1, :].gather(1, path[:, step + 1 : step + 2]).squeeze(1)
    return path, best_score


def eager_viterbi(
    log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager reference: a Python loop over timesteps, each step
    a batched tensor op — deliberately the "per-timestep launch" pattern
    project_plan.md Section 3.11 identifies as catastrophically
    launch-latency-bound for long sequences, and the actual comparison
    point its "≥5x speedup" target names.
    """
    path, score = _viterbi_impl(log_emission, log_trans, log_pi, torch.float32)
    return path, score


_compiled_cache: dict[str, Callable] = {}


def compiled_viterbi(
    log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_viterbi, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(log_emission, log_trans, log_pi)


def reference_fp64(
    log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth for correctness tests only."""
    return _viterbi_impl(log_emission, log_trans, log_pi, torch.float64)


def path_log_prob(
    path: torch.Tensor, log_emission: torch.Tensor, log_trans: torch.Tensor, log_pi: torch.Tensor
) -> torch.Tensor:
    """Recomputes the total log-probability of a given `[B, T]` decoded
    path under `(log_emission, log_trans, log_pi)` — used to verify a
    kernel's decoded path is a valid maximizer (its score equals the
    kernel's own reported `best_score`) without assuming any particular
    tie-break convention.
    """
    b, t, _s = log_emission.shape
    emission = log_emission.to(torch.float32)
    trans = log_trans.to(torch.float32)
    pi = log_pi.to(torch.float32)

    batch_idx = torch.arange(b, device=path.device)
    score = pi[path[:, 0]] + emission[batch_idx, 0, path[:, 0]]
    for step in range(1, t):
        score = score + trans[path[:, step - 1], path[:, step]] + emission[batch_idx, step, path[:, step]]
    return score
