#include <cstdio>
#include <cuda_runtime.h>

#include <cutlass/arch/memory.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

namespace cutlass::epilogue::thread {

template <typename T>
struct SwigluOp {
    static constexpr int Arguments = 2;

    CUTLASS_HOST_DEVICE
    T operator()(T const& gate, T const& up) const {
        float g = static_cast<float>(gate);
        float u = static_cast<float>(up);
        float silu = g / (1.0f + expf(-g));
        return static_cast<T>(silu * u);
    }
};

template <typename T, int N>
struct SwigluOp<Array<T, N>> {
    static constexpr int Arguments = 2;

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const& gate, Array<T, N> const& up) const {
        Array<T, N> result;
        SwigluOp<T> op;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = op(gate[i], up[i]);
        }
        return result;
    }
};

} // namespace cutlass::epilogue::thread

namespace detail {

template <typename ElementAB, typename ElementOut>
struct FusedSwigluGemmSm80 {
    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementC = ElementOut;
    using ElementD = ElementOut;
    using ElementAux = ElementOut;
    using ElementAccum = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int kStages = 3;
    static constexpr int kEVTEpilogueStages = 1;

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape, WarpShape, ElementOut, AlignmentC, kEVTEpilogueStages
    >;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    using UpLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
        OutputTileThreadMap, ElementAux,
        cute::Stride<int64_t, cute::_1, int64_t>
    >;

    using SwigluCompute = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::epilogue::thread::SwigluOp,
        ElementOut,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest
    >;

    using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<
        SwigluCompute,
        Accum,
        UpLoad
    >;

    using DStore = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementOut,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>
    >;

    using EVT = cutlass::epilogue::threadblock::Sm80EVT<
        DStore,
        EVTCompute
    >;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC,
        ElementAccum, ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape,
        EVT,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages,
        cutlass::arch::OpMultiplyAdd,
        kEVTEpilogueStages
    >::GemmKernel;

    using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename GemmDevice>
cutlass::Status launch_sm80_evt(
    typename GemmDevice::ElementA const* ptr_A,
    typename GemmDevice::ElementB const* ptr_B,
    typename GemmDevice::ElementC* ptr_D,
    typename GemmDevice::ElementC const* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int sm_count,
    cudaStream_t stream
) {
    using GemmKernel = typename GemmDevice::GemmKernel;
    using EVT = typename GemmKernel::Epilogue::FusionCallbacks;

    auto problem_size = cutlass::gemm::GemmCoord(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    typename EVT::Arguments callback_args{
        {
            {},
            {const_cast<typename GemmDevice::ElementC*>(ptr_aux), typename GemmDevice::ElementC(0),
             cute::Stride<int64_t, cute::_1, int64_t>{N, cute::_1{}, M * N}},
            {}
        },
        {ptr_D, cute::Stride<int64_t, cute::_1, int64_t>{N, cute::_1{}, M * N}}
    };

    typename GemmDevice::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        callback_args,
        ptr_A,
        ptr_B,
        nullptr,
        nullptr,
        static_cast<int64_t>(M * K),
        static_cast<int64_t>(N * K),
        0,
        static_cast<int64_t>(M * N),
        static_cast<int64_t>(K),
        static_cast<int64_t>(K),
        static_cast<int64_t>(N),
        static_cast<int64_t>(N)
    );

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

} // namespace detail

extern "C" {

bool cutlass_fused_swiglu_sm80_bf16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream
) {
    (void)cc; (void)device_id;
    using Gemm = detail::FusedSwigluGemmSm80<cutlass::bfloat16_t, cutlass::bfloat16_t>;
    auto status = detail::launch_sm80_evt<typename Gemm::GemmDevice>(
        reinterpret_cast<cutlass::bfloat16_t const*>(ptr_A),
        reinterpret_cast<cutlass::bfloat16_t const*>(ptr_B),
        reinterpret_cast<cutlass::bfloat16_t*>(ptr_D),
        reinterpret_cast<cutlass::bfloat16_t const*>(ptr_aux),
        M, N, K, sm_count, stream
    );
    return status == cutlass::Status::kSuccess;
}

bool cutlass_fused_swiglu_sm80_f16(
    const void* ptr_A, const void* ptr_B,
    void* ptr_D, const void* ptr_aux,
    int64_t M, int64_t N, int64_t K,
    int cc, int device_id, int sm_count,
    cudaStream_t stream
) {
    (void)cc; (void)device_id;
    using Gemm = detail::FusedSwigluGemmSm80<cutlass::half_t, cutlass::half_t>;
    auto status = detail::launch_sm80_evt<typename Gemm::GemmDevice>(
        reinterpret_cast<cutlass::half_t const*>(ptr_A),
        reinterpret_cast<cutlass::half_t const*>(ptr_B),
        reinterpret_cast<cutlass::half_t*>(ptr_D),
        reinterpret_cast<cutlass::half_t const*>(ptr_aux),
        M, N, K, sm_count, stream
    );
    return status == cutlass::Status::kSuccess;
}

} // extern "C"
