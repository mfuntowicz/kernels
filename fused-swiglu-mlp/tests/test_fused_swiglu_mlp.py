import platform

import torch
import torch.nn.functional as F

import fused_swiglu_mlp


def _get_device():
    if platform.system() == "Darwin":
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.version.cuda is not None and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_fused_swiglu_mlp_bf16():
    device = _get_device()
    if device.type != "cuda":
        return

    M, N, K = 128, 64, 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_gate = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_up = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    gate = x @ w_gate.t()
    up = x @ w_up.t()
    expected = F.silu(gate) * up

    result = fused_swiglu_mlp.fused_swiglu_mlp(x, w_gate, w_up)
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


def test_fused_swiglu_mlp_fp16():
    device = _get_device()
    if device.type != "cuda":
        return

    M, N, K = 128, 64, 32
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w_gate = torch.randn(N, K, dtype=torch.float16, device=device)
    w_up = torch.randn(N, K, dtype=torch.float16, device=device)

    gate = x @ w_gate.t()
    up = x @ w_up.t()
    expected = F.silu(gate) * up

    result = fused_swiglu_mlp.fused_swiglu_mlp(x, w_gate, w_up)
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


def test_fused_swiglu_mlp_fp32():
    device = _get_device()

    M, N, K = 128, 64, 32
    x = torch.randn(M, K, dtype=torch.float32, device=device)
    w_gate = torch.randn(N, K, dtype=torch.float32, device=device)
    w_up = torch.randn(N, K, dtype=torch.float32, device=device)

    gate = x @ w_gate.t()
    up = x @ w_up.t()
    expected = F.silu(gate) * up

    result = fused_swiglu_mlp.fused_swiglu_mlp(x, w_gate, w_up)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
