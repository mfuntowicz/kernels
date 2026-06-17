#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <torch/torch.h>

extern "C" {
bool cutlass_fused_swiglu_bf16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_f16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_fp8_bf16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_fp8_f16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_sm80_bf16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_sm80_f16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_sm80_dual_bf16(
    const void* ptr_A, const void* ptr_B0, const void* ptr_B1,
    void* ptr_D2,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

bool cutlass_fused_swiglu_sm80_dual_f16(
    const void* ptr_A, const void* ptr_B0, const void* ptr_B1,
    void* ptr_D2,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream);

void swiglu_elementwise_bf16(
    void* output, const void* gate, const void* up,
    int64_t M, int64_t N, cudaStream_t stream);

void swiglu_elementwise_f16(
    void* output, const void* gate, const void* up,
    int64_t M, int64_t N, cudaStream_t stream);
}

template <typename T, typename... Ts>
constexpr bool all_equal(const T& first, const Ts&... rest) {
    return ((first == rest) && ...);
}

constexpr bool isSupportedFloatingType(const torch::ScalarType type) {
    return type == c10::kFloat || type == c10::kBFloat16 || type == c10::kHalf;
}

torch::Tensor fused_swiglu_mlp(torch::Tensor const& x, torch::Tensor const& w_gate, torch::Tensor const& w_up) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_gate.device().is_cuda(), "w_gate must be a CUDA tensor");
    TORCH_CHECK(w_up.device().is_cuda(), "w_up must be a CUDA tensor");
    TORCH_CHECK(all_equal(x.device(), w_gate.device(), w_up.device()), "Tensors must be on the same device");
    TORCH_CHECK(all_equal(x.scalar_type(), w_gate.scalar_type(), w_up.scalar_type()), "Tensors must have the same dtype");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_gate.is_contiguous(), "w_gate must be contiguous");
    TORCH_CHECK(w_up.is_contiguous(), "w_up must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(w_gate.dim() == 2, "w_gate must be 2D [N, K]");
    TORCH_CHECK(w_up.dim() == 2, "w_up must be 2D [N, K]");
    TORCH_CHECK(x.size(1) == w_gate.size(1), "x K dim must match w_gate K dim");
    TORCH_CHECK(w_gate.sizes() == w_up.sizes(), "w_gate and w_up must have the same shape");
    TORCH_CHECK(isSupportedFloatingType(x.scalar_type()), "Only float32, bfloat16, float16 supported");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto M = x.size(0);
    const auto N = w_gate.size(0);
    const auto K = x.size(1);

    torch::Tensor out = torch::empty({M, N}, x.options());

    const auto* props = at::cuda::getCurrentDeviceProperties();
    const auto cc = props->major * 10 + props->minor;
    const auto device_id = at::cuda::current_device();
    const auto sm_count = static_cast<int>(props->multiProcessorCount);

    if ((cc >= 90 && cc < 120) && (x.scalar_type() != c10::kFloat)) {
        const torch::Tensor up = at::matmul(x, w_up.transpose(0, 1));

        bool ok = false;
        if (x.scalar_type() == c10::kBFloat16) {
            ok = cutlass_fused_swiglu_bf16(
                x.data_ptr(), w_gate.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        } else if (x.scalar_type() == c10::kHalf) {
            ok = cutlass_fused_swiglu_f16(
                x.data_ptr(), w_gate.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        }

        if (ok)
            return out;

        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        if (x.scalar_type() == c10::kBFloat16) {
            swiglu_elementwise_bf16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        } else {
            swiglu_elementwise_f16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        }
    } else if (cc >= 80 && cc < 90 && (x.scalar_type() == c10::kBFloat16 || x.scalar_type() == c10::kHalf)) {
        // Sm80/Sm89 (Ampere/Ada): try DualGemm first (fuses both GEMMs sharing X,
        // computes SiLU(gate)*up in the dual epilogue, no intermediate stores).
        bool ok = false;
        if (x.scalar_type() == c10::kBFloat16) {
            ok = cutlass_fused_swiglu_sm80_dual_bf16(
                x.data_ptr(), w_gate.data_ptr(), w_up.data_ptr(),
                out.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        } else if (x.scalar_type() == c10::kHalf) {
            ok = cutlass_fused_swiglu_sm80_dual_f16(
                x.data_ptr(), w_gate.data_ptr(), w_up.data_ptr(),
                out.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        }

        if (ok)
            return out;

        // Fallback 1: EVT path (fuses SiLU*multiply into gate GEMM epilogue,
        // but requires pre-computing up via cuBLAS).
        const torch::Tensor up = at::matmul(x, w_up.transpose(0, 1));
        if (x.scalar_type() == c10::kBFloat16) {
            ok = cutlass_fused_swiglu_sm80_bf16(
                x.data_ptr(), w_gate.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        } else if (x.scalar_type() == c10::kHalf) {
            ok = cutlass_fused_swiglu_sm80_f16(
                x.data_ptr(), w_gate.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        }

        if (ok)
            return out;

        // Fallback 2: two bf16 GEMMs + fused elementwise
        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        if (x.scalar_type() == c10::kBFloat16) {
            swiglu_elementwise_bf16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        } else {
            swiglu_elementwise_f16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        }
    } else if (cc >= 120 && (x.scalar_type() == c10::kBFloat16 || x.scalar_type() == c10::kHalf)) {
        // Sm120 (consumer Blackwell): try FP8 fused GEMM + SwiGLU EVT path.
        // CUTLASS has no Sm120 bf16 UMMA, so we convert to FP8 E4M3 for the
        // fused second GEMM. The first GEMM (up) stays bf16 via cuBLAS.
        const torch::Tensor up = at::matmul(x, w_up.transpose(0, 1));

        const auto x_fp8 = x.to(at::kFloat8_e4m3fn);
        const auto w_gate_fp8 = w_gate.to(at::kFloat8_e4m3fn);

        bool ok = false;
        if (x.scalar_type() == c10::kBFloat16) {
            ok = cutlass_fused_swiglu_fp8_bf16(
                x_fp8.data_ptr(), w_gate_fp8.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        } else if (x.scalar_type() == c10::kHalf) {
            ok = cutlass_fused_swiglu_fp8_f16(
                x_fp8.data_ptr(), w_gate_fp8.data_ptr(),
                out.data_ptr(), up.data_ptr(),
                M, N, K, cc, device_id, sm_count, stream);
        }

        if (ok)
            return out;

        // Fallback: two bf16 GEMMs + fused elementwise
        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        if (x.scalar_type() == c10::kBFloat16) {
            swiglu_elementwise_bf16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        } else {
            swiglu_elementwise_f16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        }
    } else {
        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        const auto up = at::matmul(x, w_up.transpose(0, 1));
        if (x.scalar_type() == c10::kBFloat16) {
            swiglu_elementwise_bf16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        } else if (x.scalar_type() == c10::kHalf) {
            swiglu_elementwise_f16(out.data_ptr(), gate.data_ptr(), up.data_ptr(), M, N, stream);
        } else {
            at::silu_out(out, gate);
            out.mul_(up);
        }
    }

    return out;
}
