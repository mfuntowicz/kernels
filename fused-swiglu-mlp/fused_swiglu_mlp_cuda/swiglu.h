#ifndef FUSED_SWIGLU_OP
#define FUSED_SWIGLU_OP

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

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

template <
    typename ElementOutput,
    typename ElementAux = ElementOutput,
    typename EpilogueTile = cutlass::gemm::GemmShape<16, 8>
>
using SwigluEVT = cutlass::epilogue::fusion::Sm90TreeVisitor<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::SwigluOp,
        ElementOutput,
        float,
        cutlass::FloatRoundStyle::round_to_nearest
    >,
    cutlass::epilogue::fusion::Sm90AccFetch,
    cutlass::epilogue::fusion::Sm90AuxLoad<
        0,
        EpilogueTile,
        ElementAux,
        cutlass::layout::RowMajor,
        void,
        void
    >
>;

#endif