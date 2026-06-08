from typing import Optional

import torch

from ._ops import ops


def fused_swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    """Compute SiLU(x @ w_gate.T) * (x @ w_up.T).

    Args:
        x: Input tensor of shape [M, K].
        w_gate: Gate projection weight of shape [N, K].
        w_up: Up projection weight of shape [N, K].
    Returns:
        Output tensor of shape [M, N].
    """
    return ops.fused_swiglu_mlp(x, w_gate, w_up)
