"""Python entrypoints for individual kernels, one module per kernel.

Each submodule wraps the corresponding `custom_cuda._native` PyO3 binding
with a torch.autograd.Function / nn.Module-friendly Python API. See
project_plan.md Section 3 for the per-kernel specification and Section 7
for implementation order.
"""
