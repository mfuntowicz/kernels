# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch",
# ]
# ///

import platform
from pathlib import Path

import kernels
import torch
import torch.nn.functional as F

kernel = kernels.get_local_kernel(Path("torch-ext"), "cuda")

if platform.system() == "Darwin":
    device = torch.device("mps")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.version.cuda is not None and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    cc = torch.cuda.get_device_capability()
    print(f"GPU compute capability: {cc[0]}.{cc[1]}")
    if cc[0] >= 9:
        print("CUTLASS fused SwiGLU path will be used")
    else:
        print("Elementwise fallback path will be used (fused path requires SM90+)")

SIZES = [(64, 32, 16), (1024, 1024, 1024), (4096, 4096, 4096)]

for M, N, K in SIZES:
    print(f"\n--- Shape: [{M}, {N}, {K}], dtype: bfloat16 ---")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_gate = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_up = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    result = kernel.fused_swiglu_mlp(x, w_gate, w_up)
    gate = x @ w_gate.t()
    up = x @ w_up.t()
    expected = F.silu(gate) * up

    diff = torch.abs(result.float() - expected.float())
    rel_diff = diff / (torch.abs(expected.float()) + 1e-8)
    print(
        f"  absdiff  sum: {diff.sum():.2f}  max: {diff.max():.4f}  mean: {diff.mean():.6f}"
    )
    print(f"  reldiff  max: {rel_diff.max():.6f}  mean: {rel_diff.mean():.6f}")
    assert torch.allclose(result.float(), expected.float(), atol=1e-2, rtol=1e-2)

num_warmup = 5
num_iters = 50

M, N, K = 4096, 4096, 4096
x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
w_gate = torch.randn(N, K, dtype=torch.bfloat16, device=device)
w_up = torch.randn(N, K, dtype=torch.bfloat16, device=device)

for _ in range(num_warmup):
    _ = kernel.fused_swiglu_mlp(x, w_gate, w_up)
    _ = F.silu(x @ w_gate.t()) * (x @ w_up.t())
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(num_iters):
    result = kernel.fused_swiglu_mlp(x, w_gate, w_up)
end.record()
torch.cuda.synchronize()
kernel_ms = start.elapsed_time(end) / num_iters

start.record()
for _ in range(num_iters):
    gate = x @ w_gate.t()
    up = x @ w_up.t()
    expected = F.silu(gate) * up
end.record()
torch.cuda.synchronize()
torch_ms = start.elapsed_time(end) / num_iters

print(f"\n--- Timing [{M}, {N}, {K}] ---")
print(f"Kernel : {kernel_ms:.3f} ms")
print(f"Torch  : {torch_ms:.3f} ms")
print(f"Speedup: {torch_ms / kernel_ms:.2f}x")
