#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cutlass/arch/arch.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/half.h>
#include <cutlass/util/packed_stride.hpp>

#include "swiglu.h"

template <typename T, typename... Ts>
constexpr bool all_equal(const T& first, const Ts&... rest) {
    return ((first == rest) && ...);
}

constexpr bool isSupportedFloatingType(const torch::ScalarType type) {
    return type == c10::kFloat || type == c10::kBFloat16 || type == c10::kHalf;
}

namespace detail {

template <typename ElementAB, typename ElementOut, typename ArchTag, typename EpilogueSchedule>
struct FusedSwigluGemm;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <typename ElementAB, typename ElementOut>
struct FusedSwigluGemm<ElementAB, ElementOut, cutlass::arch::Sm90, cutlass::epilogue::TmaWarpSpecializedCooperative> {
    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementC = void;
    using ElementD = ElementOut;
    using ElementAux = ElementOut;
    using ElementAccum = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 1;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using TileShapeMNK = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using ClusterShapeMNK = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using EVT = SwigluEVT<ElementD, ElementAux>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShapeMNK,
        ClusterShapeMNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum,
        ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        EVT
    >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccum,
        TileShapeMNK,
        ClusterShapeMNK,
        StageCount,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int64_t, int64_t, int64_t, int64_t>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
};

#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename ElementAB, typename ElementOut>
struct FusedSwigluGemm<ElementAB, ElementOut, cutlass::arch::Sm100, cutlass::epilogue::TmaWarpSpecialized2Sm> {
    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementC = void;
    using ElementD = ElementOut;
    using ElementAux = ElementOut;
    using ElementAccum = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 1;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using TileShapeMNK = cute::Shape<cute::_256, cute::_128, cute::_64>;
    using ClusterShapeMNK = cute::Shape<cute::_2, cute::_2, cute::_1>;

    using EVT = SwigluEVT<ElementD, ElementAux>;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShapeMNK,
        ClusterShapeMNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum,
        ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        EVT
    >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccum,
        TileShapeMNK,
        ClusterShapeMNK,
        StageCount,
        cutlass::gemm::KernelTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int64_t, int64_t, int64_t, int64_t>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
};

#endif

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

template <typename GemmDevice>
cutlass::Status launch_fused_swiglu_gemm(
    typename GemmDevice::GemmKernel::CollectiveMainloop::ElementA const* ptr_A,
    typename GemmDevice::GemmKernel::CollectiveMainloop::ElementB const* ptr_B,
    typename GemmDevice::GemmKernel::CollectiveEpilogue::ElementD* ptr_D,
    typename GemmDevice::GemmKernel::CollectiveEpilogue::ElementD const* ptr_aux,
    const int64_t M, const int64_t N, const int64_t K,
    cudaStream_t stream
) {
    using GemmKernel = typename GemmDevice::GemmKernel;
    using ProblemShape = typename GemmKernel::ProblemShape;

    ProblemShape problem_shape = ProblemShape{M, N, K, static_cast<int64_t>(1)};

    const auto m = static_cast<int32_t>(M);
    const auto n = static_cast<int32_t>(N);
    const auto k = static_cast<int32_t>(K);

    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, cute::make_shape(m, n, 1));

    auto stride_aux = cutlass::make_cute_packed_stride(
        cute::Stride<int64_t, cute::Int<1>, int64_t>{}, cute::make_shape(m, n, 1)
    );

    typename GemmKernel::CollectiveMainloop::Arguments mainloop_args = { ptr_A, stride_A, ptr_B, stride_B };

    using EVT = typename GemmKernel::CollectiveEpilogue::FusionCallbacks;
    typename EVT::Arguments evt_args = {
        {},
        {ptr_aux, typename GemmDevice::GemmKernel::CollectiveEpilogue::ElementD(0), stride_aux},
        {}
    };

    typename GemmKernel::CollectiveEpilogue::Arguments epilogue_args = {
        evt_args,
        nullptr, {},
        ptr_D, stride_D
    };

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = static_cast<unsigned char>(at::cuda::current_device());
    hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    typename GemmDevice::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        mainloop_args,
        epilogue_args,
        hw_info,
        {}
    };

    GemmDevice gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    auto* allocator = c10::cuda::CUDACachingAllocator::get();
    const auto workspace_size = GemmDevice::get_workspace_size(args);
    const auto workspace = allocator->allocate(workspace_size);

    status = gemm_op.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) return status;

    return gemm_op.run(stream);
}

} // namespace detail

template <typename ElementAB, typename ElementOut, typename ArchTag, typename EpilogueSchedule>
cutlass::Status run_swiglu_gemm(
    ElementAB const* ptr_A,
    ElementAB const* ptr_B,
    ElementOut* ptr_D,
    ElementOut const* ptr_aux,
    const int64_t M, const int64_t N, const int64_t K,
    cudaStream_t stream
) {
    using Gemm = typename detail::FusedSwigluGemm<ElementAB, ElementOut, ArchTag, EpilogueSchedule>::GemmDevice;
    return detail::launch_fused_swiglu_gemm<Gemm>(
        ptr_A, ptr_B, ptr_D, ptr_aux,
        M, N, K, stream
    );
}

template <typename Element>
void run_swiglu_elementwise(
    Element* output, Element const* gate, Element const* up,
    const int64_t M, const int64_t N, cudaStream_t stream
) {
    constexpr auto block_size = 256l;
    const auto total = M * N;
    const auto grid_size = (total + block_size - 1) / block_size;
    detail::swiglu_elementwise_kernel<Element><<<grid_size, block_size, 0, stream>>>(output, gate, up, total);
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

    const auto cc_major = at::cuda::getCurrentDeviceProperties()->major;
    const auto cc_minor = at::cuda::getCurrentDeviceProperties()->minor;
    const auto cc = cc_major * 10 + cc_minor;

    if (bool use_cutlass_fusion = (cc >= 90) && (x.scalar_type() != c10::kFloat)) {
        const torch::Tensor up = at::matmul(x, w_up.transpose(0, 1));

        auto try_cutlass = [&]() -> bool {
            auto status = cutlass::Status::kErrorInternal;

            if (x.scalar_type() == c10::kBFloat16) {
                using ElementAB = cutlass::bfloat16_t;
                using ElementOut = cutlass::bfloat16_t;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
                if (cc >= 100) {
                    status = run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm100, cutlass::epilogue::TmaWarpSpecialized2Sm>(
                        static_cast<ElementAB const*>(x.data_ptr()),
                        static_cast<ElementAB const*>(w_gate.data_ptr()),
                        static_cast<ElementOut*>(out.data_ptr()),
                        static_cast<ElementOut const*>(up.data_ptr()),
                        M, N, K, stream
                    );
                } else
#endif
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
                {
                    status = run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm90, cutlass::epilogue::TmaWarpSpecializedCooperative>(
                        static_cast<ElementAB const*>(x.data_ptr()),
                        static_cast<ElementAB const*>(w_gate.data_ptr()),
                        static_cast<ElementOut*>(out.data_ptr()),
                        static_cast<ElementOut const*>(up.data_ptr()),
                        M, N, K, stream
                    );
                }
#else
                { return false; }
#endif
            } else if (x.scalar_type() == c10::kHalf) {
                using ElementAB = cutlass::half_t;
                using ElementOut = cutlass::half_t;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
                if (cc >= 100) {
                    status = run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm100, cutlass::epilogue::TmaWarpSpecialized2Sm>(
                        static_cast<ElementAB const*>(x.data_ptr()),
                        static_cast<ElementAB const*>(w_gate.data_ptr()),
                        static_cast<ElementOut*>(out.data_ptr()),
                        static_cast<ElementOut const*>(up.data_ptr()),
                        M, N, K, stream
                    );
                } else
#endif
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
                {
                    status = run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm90, cutlass::epilogue::TmaWarpSpecializedCooperative>(
                        static_cast<ElementAB const*>(x.data_ptr()),
                        static_cast<ElementAB const*>(w_gate.data_ptr()),
                        static_cast<ElementOut*>(out.data_ptr()),
                        static_cast<ElementOut const*>(up.data_ptr()),
                        M, N, K, stream
                    );
                }
#else
                { return false; }
#endif
            }

            return status == cutlass::Status::kSuccess;
        };

        if (try_cutlass())
            return out;

        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        at::silu_out(out, gate);
        out.mul_(up);
    } else {
        const auto gate = at::matmul(x, w_gate.transpose(0, 1));
        const auto up = at::matmul(x, w_up.transpose(0, 1));

        at::silu_out(out, gate);
        out.mul_(up);
    }

    return out;
}