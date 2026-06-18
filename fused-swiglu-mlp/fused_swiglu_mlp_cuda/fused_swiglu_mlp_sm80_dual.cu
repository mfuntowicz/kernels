#include <cstdio>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>

#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace detail {

template <typename ElementAB, typename ElementOut>
struct FusedSwigluDualGemmSm80 {
    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementC = ElementOut;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB0 = cutlass::layout::ColumnMajor;
    using LayoutB1 = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int kAlign = 128 / cutlass::sizeof_bits<ElementOut>::value;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int kStages = 3;
    static constexpr bool kStoreD0 = false;
    static constexpr bool kStoreD1 = false;
    static constexpr bool kSplitKSerial = false;

    using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
        ElementOut, kAlign, ElementAccumulator, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Nothing
    >;
    using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
        ElementOut, kAlign, ElementAccumulator, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Nothing
    >;
    using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
        ElementOut, kAlign, ElementOut, ElementCompute
    >;

    using DualGemm = cutlass::gemm::device::DualGemm<
        ElementA, LayoutA,
        ElementB, LayoutB0, LayoutB1,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp0, EpilogueOutputOp1, EpilogueOutputOp2,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        kStages, kStoreD0, kStoreD1, kSplitKSerial
    >;
};

template <typename DualGemmDevice>
bool launch_dual_gemm(
    typename DualGemmDevice::ElementA const* ptr_A,
    typename DualGemmDevice::ElementB const* ptr_B0,
    typename DualGemmDevice::ElementB const* ptr_B1,
    typename DualGemmDevice::ElementC* ptr_D2,
    const int64_t M, const int64_t N, const int64_t K,
    cudaStream_t stream
) {
    auto problem_size = cutlass::gemm::GemmCoord(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    typename DualGemmDevice::Arguments args(
        cutlass::gemm::DualGemmMode::kGemm,
        problem_size,
        {ptr_A,  static_cast<typename DualGemmDevice::LayoutA::LongIndex>(K)},
        {ptr_B0, static_cast<typename DualGemmDevice::LayoutB0::LongIndex>(K)},
        {},
        {},
        {ptr_B1, static_cast<typename DualGemmDevice::LayoutB1::LongIndex>(K)},
        {},
        {},
        {ptr_D2, static_cast<typename DualGemmDevice::LayoutC::LongIndex>(N)},
        {typename DualGemmDevice::EpilogueOutputOp0::ElementCompute(1),
         typename DualGemmDevice::EpilogueOutputOp0::ElementCompute(0)},
        {typename DualGemmDevice::EpilogueOutputOp1::ElementCompute(1),
         typename DualGemmDevice::EpilogueOutputOp1::ElementCompute(0)},
        {},
        1,
        1,
        0, 0, 0, 0, 0
    );

    DualGemmDevice gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return false;

    const auto workspace_size = DualGemmDevice::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        auto cuda_status = cudaMalloc(&workspace, workspace_size);
        if (cuda_status != cudaSuccess) return false;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return false;
    }

    status = gemm_op.run(stream);
    if (workspace) cudaFree(workspace);
    return status == cutlass::Status::kSuccess;
}

} // namespace detail

extern "C" {

bool cutlass_fused_swiglu_sm80_dual_bf16(
    const void* ptr_A, const void* ptr_B0, const void* ptr_B1,
    void* ptr_D2,
    const int64_t M, const int64_t N, const int64_t K,
    const int cc, const int device_id, const int sm_count,
    cudaStream_t stream
) {
    (void)cc; (void)device_id; (void)sm_count;
    using Gemm = detail::FusedSwigluDualGemmSm80<cutlass::bfloat16_t, cutlass::bfloat16_t>;
    return detail::launch_dual_gemm<typename Gemm::DualGemm>(
        static_cast<cutlass::bfloat16_t const*>(ptr_A),
        static_cast<cutlass::bfloat16_t const*>(ptr_B0),
        static_cast<cutlass::bfloat16_t const*>(ptr_B1),
        static_cast<cutlass::bfloat16_t*>(ptr_D2),
        M, N, K, stream
    );
}

bool cutlass_fused_swiglu_sm80_dual_f16(
    const void* ptr_A, const void* ptr_B0, const void* ptr_B1,
    void* ptr_D2,
    const int64_t M, const int64_t N, const int64_t K,
    const int cc, const int device_id, const int sm_count,
    cudaStream_t stream
) {
    (void)cc; (void)device_id; (void)sm_count;
    using Gemm = detail::FusedSwigluDualGemmSm80<cutlass::half_t, cutlass::half_t>;
    return detail::launch_dual_gemm<typename Gemm::DualGemm>(
        static_cast<cutlass::half_t const*>(ptr_A),
        static_cast<cutlass::half_t const*>(ptr_B0),
        static_cast<cutlass::half_t const*>(ptr_B1),
        static_cast<cutlass::half_t*>(ptr_D2),
        M, N, K, stream
    );
}

} // extern "C"
