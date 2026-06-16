#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename Element>
__global__ void swiglu_elementwise_kernel(
    Element* __restrict__ output,
    Element const* __restrict__ gate,
    Element const* __restrict__ up,
    const int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const auto g = static_cast<float>(gate[idx]);
        const auto u = static_cast<float>(up[idx]);
        const float silu = g / (1.0f + expf(-g));
        output[idx] = static_cast<Element>(silu * u);
    }
}

namespace detail {

template <typename Element>
void run_swiglu_elementwise(
    Element* output, Element const* gate, Element const* up,
    const int64_t M, const int64_t N, cudaStream_t stream
) {
    constexpr auto block_size = 256l;
    const auto total = M * N;
    const auto grid_size = (total + block_size - 1) / block_size;
    swiglu_elementwise_kernel<Element><<<grid_size, block_size, 0, stream>>>(output, gate, up, total);
}

} // namespace detail

extern "C" {

void swiglu_elementwise_bf16(
    void* output, const void* gate, const void* up,
    const int64_t M, const int64_t N,
    cudaStream_t stream
) {
    detail::run_swiglu_elementwise<__nv_bfloat16>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<__nv_bfloat16 const*>(gate),
        static_cast<__nv_bfloat16 const*>(up),
        M, N, stream
    );
}

void swiglu_elementwise_f16(
    void* output, const void* gate, const void* up,
    const int64_t M, const int64_t N,
    cudaStream_t stream
) {
    detail::run_swiglu_elementwise<__half>(
        static_cast<__half*>(output),
        static_cast<__half const*>(gate),
        static_cast<__half const*>(up),
        M, N, stream
    );
}

} // extern "C"