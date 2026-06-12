#include <cuda_runtime.h>

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

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
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

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
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
    const int device_id, const int sm_count,
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
    hw_info.device_id = static_cast<unsigned char>(device_id);
    hw_info.sm_count = sm_count;

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

    const auto workspace_size = GemmDevice::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        auto cuda_status = cudaMalloc(&workspace, workspace_size);
        if (cuda_status != cudaSuccess) return cutlass::Status::kErrorInternal;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return status;
    }

    status = gemm_op.run(stream);
    if (workspace) cudaFree(workspace);
    return status;
}

template <typename ElementAB, typename ElementOut, typename ArchTag, typename EpilogueSchedule>
cutlass::Status run_swiglu_gemm(
    ElementAB const* ptr_A,
    ElementAB const* ptr_B,
    ElementOut* ptr_D,
    ElementOut const* ptr_aux,
    const int64_t M, const int64_t N, const int64_t K,
    const int device_id, const int sm_count,
    cudaStream_t stream
) {
    using Gemm = typename FusedSwigluGemm<ElementAB, ElementOut, ArchTag, EpilogueSchedule>::GemmDevice;
    return launch_fused_swiglu_gemm<Gemm>(
        ptr_A, ptr_B, ptr_D, ptr_aux,
        M, N, K, device_id, sm_count, stream
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
    swiglu_elementwise_kernel<Element><<<grid_size, block_size, 0, stream>>>(output, gate, up, total);
}

} // namespace detail

// ---------------------------------------------------------------------------
// C-linkage interface — called from the PyTorch host .cpp file.
// This avoids including any PyTorch headers in the .cu compilation unit,
// avoiding GCC 15 + C++17 template-body errors in ATen headers and
// C++20/C++17 ABI mismatches with libtorch.
// ---------------------------------------------------------------------------

extern "C" {

// Returns true if the CUTLASS fused SwiGLU GEMM succeeded.
// ptr_A: x data [M, K], ptr_B: w_gate data [N, K], ptr_D: output [M, N]
// ptr_aux: up projection result [M, N]
bool cutlass_fused_swiglu_bf16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    const int64_t M, const int64_t N, const int64_t K,
    const int cc, const int device_id, const int sm_count,
    cudaStream_t stream
) {
    using ElementAB = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;
    auto status = cutlass::Status::kErrorInternal;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (cc >= 100) {
        status = detail::run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm100, cutlass::epilogue::TmaWarpSpecialized2Sm>(
            static_cast<ElementAB const*>(ptr_A),
            static_cast<ElementAB const*>(ptr_B),
            static_cast<ElementOut*>(ptr_D),
            static_cast<ElementOut const*>(ptr_aux),
            M, N, K, device_id, sm_count, stream
        );
    } else
#endif
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        status = detail::run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm90, cutlass::epilogue::TmaWarpSpecializedCooperative>(
            static_cast<ElementAB const*>(ptr_A),
            static_cast<ElementAB const*>(ptr_B),
            static_cast<ElementOut*>(ptr_D),
            static_cast<ElementOut const*>(ptr_aux),
            M, N, K, device_id, sm_count, stream
        );
    }
#else
    { return false; }
#endif

    return status == cutlass::Status::kSuccess;
}

bool cutlass_fused_swiglu_f16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    const int64_t M, const int64_t N, const int64_t K,
    const int cc, const int device_id, const int sm_count,
    cudaStream_t stream
) {
    using ElementAB = cutlass::half_t;
    using ElementOut = cutlass::half_t;
    auto status = cutlass::Status::kErrorInternal;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (cc >= 100) {
        status = detail::run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm100, cutlass::epilogue::TmaWarpSpecialized2Sm>(
            static_cast<ElementAB const*>(ptr_A),
            static_cast<ElementAB const*>(ptr_B),
            static_cast<ElementOut*>(ptr_D),
            static_cast<ElementOut const*>(ptr_aux),
            M, N, K, device_id, sm_count, stream
        );
    } else
#endif
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        status = detail::run_swiglu_gemm<ElementAB, ElementOut, cutlass::arch::Sm90, cutlass::epilogue::TmaWarpSpecializedCooperative>(
            static_cast<ElementAB const*>(ptr_A),
            static_cast<ElementAB const*>(ptr_B),
            static_cast<ElementOut*>(ptr_D),
            static_cast<ElementOut const*>(ptr_aux),
            M, N, K, device_id, sm_count, stream
        );
    }
#else
    { return false; }
#endif

    return status == cutlass::Status::kSuccess;
}

void swiglu_elementwise_bf16(
    void* output, const void* gate, const void* up,
    const int64_t M, const int64_t N,
    cudaStream_t stream
) {
    detail::run_swiglu_elementwise<cutlass::bfloat16_t>(
        static_cast<cutlass::bfloat16_t*>(output),
        static_cast<cutlass::bfloat16_t const*>(gate),
        static_cast<cutlass::bfloat16_t const*>(up),
        M, N, stream
    );
}

void swiglu_elementwise_f16(
    void* output, const void* gate, const void* up,
    const int64_t M, const int64_t N,
    cudaStream_t stream
) {
    detail::run_swiglu_elementwise<cutlass::half_t>(
        static_cast<cutlass::half_t*>(output),
        static_cast<cutlass::half_t const*>(gate),
        static_cast<cutlass::half_t const*>(up),
        M, N, stream
    );
}

} // extern "C"