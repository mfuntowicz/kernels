import time
from pathlib import Path

import torch
import torch.nn.functional as F
import kernels

device = torch.device("cuda")
cc = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name()}, cc={cc[0]}.{cc[1]}")

fused = kernels.get_local_kernel(Path(__file__).parent / "torch-ext", "cuda")

M, N, K = 4096, 4096, 4096
x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
w_gate = torch.randn(N, K, dtype=torch.bfloat16, device=device)
w_up = torch.randn(N, K, dtype=torch.bfloat16, device=device)

num_warmup = 10
num_iters = 100


def bench(fn, label):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / num_iters
    print(f"  {label:45s}: {ms:.3f} ms")
    return ms


print(f"\n--- Fair benchmark [{M}, {N}, {K}] bf16 ---")

# Our fused kernel (DualGemm on Sm89)
t_kernel = bench(lambda: fused.fused_swiglu_mlp(x, w_gate, w_up), "Fused (DualGemm)")

# Fair torch baseline: bf16 throughout
def torch_bf16():
    gate = x @ w_gate.t()
    up = x @ w_up.t()
    return F.silu(gate) * up

t_torch_bf16 = bench(torch_bf16, "Torch (2 cuBLAS bf16 + bf16 silu*up)")

# Torch with float32 silu (what example.py uses — unfair)
def torch_f32_silu():
    gate = x @ w_gate.t()
    up = x @ w_up.t()
    return F.silu(gate.float()) * up.float()

t_torch_f32 = bench(torch_f32_silu, "Torch (2 cuBLAS + f32 silu*up)")

# Just the 2 GEMMs (no elementwise) — to measure GEMM-only time
def torch_gemm_only():
    gate = x @ w_gate.t()
    up = x @ w_up.t()
    return gate, up

t_2gemm = bench(torch_gemm_only, "2 cuBLAS GEMMs only (no epilogue)")

# Single GEMM
def torch_1gemm():
    return x @ w_gate.t()

t_1gemm = bench(torch_1gemm, "1 cuBLAS GEMM only")

print(f"\n--- Summary ---")
print(f"  Fused vs Torch bf16:   {t_torch_bf16 / t_kernel:.2f}x")
print(f"  Fused vs Torch f32:    {t_torch_f32 / t_kernel:.2f}x")
print(f"  Fused vs 2 GEMMs only: {t_2gemm / t_kernel:.2f}x (lower bound)")
