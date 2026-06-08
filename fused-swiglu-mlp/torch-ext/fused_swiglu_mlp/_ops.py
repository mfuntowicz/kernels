import torch
from . import _fused_swiglu_mlp_cuda_6cfvnwjfilxus
ops = torch.ops._fused_swiglu_mlp_cuda_6cfvnwjfilxus

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_fused_swiglu_mlp_cuda_6cfvnwjfilxus::{op_name}"
