import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class FusedSwigluMlpBenchmark(Benchmark):
    def setup(self):
        self.M = 4096
        self.N = 4096
        self.K = 4096
        self.x = torch.randn(self.M, self.K, device=self.device, dtype=torch.bfloat16)
        self.w_gate = torch.randn(
            self.N, self.K, device=self.device, dtype=torch.bfloat16
        )
        self.w_up = torch.randn(
            self.N, self.K, device=self.device, dtype=torch.bfloat16
        )

    def benchmark_fused(self):
        self.kernel.fused_swiglu_mlp(self.x, self.w_gate, self.w_up)

    def benchmark_baseline(self):
        gate = self.x @ self.w_gate.t()
        up = self.x @ self.w_up.t()
        F.silu(gate) * up

    def verify_fused(self) -> torch.Tensor:
        gate = self.x @ self.w_gate.t()
        up = self.x @ self.w_up.t()
        return F.silu(gate) * up
