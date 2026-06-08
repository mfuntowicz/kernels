#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fused_swiglu_mlp(Tensor x, Tensor w_gate, Tensor w_up) -> Tensor");

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("fused_swiglu_mlp", torch::kCUDA, &fused_swiglu_mlp);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)