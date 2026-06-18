#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <ATen/ops/silu.h>

#include "torch_binding.h"

namespace {

torch::Tensor reference(torch::Tensor const& x, torch::Tensor const& w_gate, torch::Tensor const& w_up) {
    auto x_f32 = x.to(torch::kFloat32);
    auto wg_f32 = w_gate.t().to(torch::kFloat32);
    auto wu_f32 = w_up.t().to(torch::kFloat32);
    return at::silu(x_f32.matmul(wg_f32)) * x_f32.matmul(wu_f32);
}

int failures = 0;
int tests = 0;

void check(bool ok, const char* name) {
    tests++;
    if (ok) {
        printf("  [PASS] %s\n", name);
    } else {
        printf("  [FAIL] %s\n", name);
        failures++;
    }
}

void test_correctness(torch::ScalarType dtype, int64_t M, int64_t N, int64_t K,
                      double atol, double rtol) {
    char name[256];
    snprintf(name, sizeof(name), "correctness %s [%ldx%ldx%ld]",
             c10::toString(dtype), M, N, K);

    auto x = torch::randn({M, K}, torch::dtype(dtype).device(torch::kCUDA));
    auto w_gate = torch::randn({N, K}, torch::dtype(dtype).device(torch::kCUDA));
    auto w_up = torch::randn({N, K}, torch::dtype(dtype).device(torch::kCUDA));

    auto result = fused_swiglu_mlp(x, w_gate, w_up);
    auto expected = reference(x, w_gate, w_up);

    check(torch::allclose(result.to(torch::kFloat32), expected, atol, rtol), name);
}

void test_shape_variants() {
    printf("Testing shape variants (bf16):\n");
    test_correctness(torch::kBFloat16, 64, 32, 16, 0.5, 5e-2);
    test_correctness(torch::kBFloat16, 128, 64, 32, 0.5, 5e-2);
    test_correctness(torch::kBFloat16, 1024, 1024, 1024, 0.5, 5e-2);
    test_correctness(torch::kBFloat16, 4096, 4096, 4096, 0.5, 5e-2);
    test_correctness(torch::kBFloat16, 256, 128, 512, 0.5, 5e-2);
    test_correctness(torch::kBFloat16, 1, 128, 256, 0.5, 5e-2);
}

void test_dtypes() {
    printf("Testing dtypes [128x64x32]:\n");
    test_correctness(torch::kBFloat16, 128, 64, 32, 0.5, 5e-2);
    test_correctness(torch::kFloat16, 128, 64, 32, 0.5, 5e-2);
    test_correctness(torch::kFloat32, 128, 64, 32, 1e-5, 1e-5);
}

void test_input_validation() {
    printf("Testing input validation:\n");

    auto x = torch::randn({128, 64}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto w_gate = torch::randn({64, 64}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto w_up = torch::randn({64, 64}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    // Non-contiguous x
    {
        auto x_nc = x.transpose(0, 1).contiguous().transpose(0, 1);
        bool threw = false;
        try {
            fused_swiglu_mlp(x_nc, w_gate, w_up);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "non-contiguous x rejected");
    }

    // Mismatched dtypes
    {
        auto w_gate_f32 = torch::randn({64, 64}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        bool threw = false;
        try {
            fused_swiglu_mlp(x, w_gate_f32, w_up);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "mismatched dtypes rejected");
    }

    // Mismatched K dims
    {
        auto w_gate_bad = torch::randn({64, 32}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        bool threw = false;
        try {
            fused_swiglu_mlp(x, w_gate_bad, w_up);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "mismatched K dims rejected");
    }

    // Mismatched w_gate/w_up shapes
    {
        auto w_up_bad = torch::randn({32, 64}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        bool threw = false;
        try {
            fused_swiglu_mlp(x, w_gate, w_up_bad);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "mismatched w_gate/w_up shapes rejected");
    }

    // CPU tensor (should fail)
    {
        auto x_cpu = torch::randn({128, 64}, torch::dtype(torch::kBFloat16));
        auto wg_cpu = torch::randn({64, 64}, torch::dtype(torch::kBFloat16));
        auto wu_cpu = torch::randn({64, 64}, torch::dtype(torch::kBFloat16));
        bool threw = false;
        try {
            fused_swiglu_mlp(x_cpu, wg_cpu, wu_cpu);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "CPU tensors rejected");
    }

    // 1D tensor (should fail)
    {
        auto x_1d = torch::randn({64}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        bool threw = false;
        try {
            fused_swiglu_mlp(x_1d, w_gate, w_up);
        } catch (const c10::Error&) {
            threw = true;
        }
        check(threw, "1D tensor rejected");
    }
}

} // namespace

int main() {
    if (!torch::cuda::is_available()) {
        printf("CUDA not available, skipping tests.\n");
        return 0;
    }

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s (cc=%d.%d)\n",
           prop.name, prop.major, prop.minor);

    test_dtypes();
    test_shape_variants();
    test_input_validation();

    printf("\n%d/%d tests passed\n", tests - failures, tests);
    return failures == 0 ? 0 : 1;
}
