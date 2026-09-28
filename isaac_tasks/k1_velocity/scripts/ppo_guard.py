"""Keep one bad PPO update from permanently poisoning the policy weights.

rsl-rl bounds updates by clipping the total gradient norm, but a non-finite
total norm makes torch's clip coefficient NaN, so a single update with an inf
value loss writes NaN into every weight.  The next rollout then dies with
``normal expects all elements of std >= 0.0`` because the Gaussian std is NaN,
and the run cannot be recovered from its own checkpoint.

Sanitising the gradients *before* clipping turns that fatal update into a no-op:
the step is skipped, the weights stay finite, and training continues.
"""
from __future__ import annotations

import torch
import torch.nn.utils as _nn_utils

_ORIGINAL_CLIP = _nn_utils.clip_grad_norm_
_INSTALLED = False
SKIPPED_UPDATES = 0


def _clip_grad_norm_with_finite_guard(parameters, max_norm, *args, **kwargs):
    global SKIPPED_UPDATES
    params = [p for p in parameters]
    for p in params:
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    norms = [p.grad.detach().float().norm(2) for p in params if p.grad is not None]
    if norms:
        total = torch.linalg.vector_norm(torch.stack(norms))
        if not torch.isfinite(total):
            # Overflowing norm even after per-element cleanup: drop the update.
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            SKIPPED_UPDATES += 1
            return torch.zeros(())
    return _ORIGINAL_CLIP(params, max_norm, *args, **kwargs)


def install() -> bool:
    """Patch gradient clipping once per process.  Returns True if applied."""
    global _INSTALLED
    if _INSTALLED:
        return False
    _nn_utils.clip_grad_norm_ = _clip_grad_norm_with_finite_guard
    _INSTALLED = True
    return True


def skipped() -> int:
    return SKIPPED_UPDATES
