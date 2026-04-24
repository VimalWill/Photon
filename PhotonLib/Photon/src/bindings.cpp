#include <torch/extension.h>
#include "AccelLinearAttention.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_attention",
        &AccelLinearAttention,
        "Linear attention (CUDA)",
        py::arg("query"), py::arg("key"), py::arg("value")
    );
}
