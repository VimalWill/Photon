#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace Photon {

    __host__ void LinearAttn(
        const float* Q,
        const float* K,
        const float* V,
        float*       Out,
        int          b,
        int          n,
        int          d
    );

}

torch::Tensor AccelLinearAttention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value
);
