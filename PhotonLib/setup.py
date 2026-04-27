import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;9.0")

from setuptools import setup
import torch.utils.cpp_extension as cpp_ext
cpp_ext._check_cuda_version = lambda *_: None  # system CUDA 12.2 vs torch cu130
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="photon",
    ext_modules=[
        CUDAExtension(
            name="photon",
            sources=[
                "Photon/src/AccelLinearAttention.cu",
                "Photon/src/bindings.cpp",
            ],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
