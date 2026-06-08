#pragma once

#include <torch/torch.h>

torch::Tensor fused_swiglu_mlp(
    torch::Tensor const& x,
    torch::Tensor const& w_gate,
    torch::Tensor const& w_up
);