# AccelLinearAttn — Photon

A CUDA-accelerated linear attention kernel exposed as a PyTorch extension.  
The kernel implements normalized linear attention with a first-order Taylor feature map (`φ(x) = 1 + x`) using a chunked recurrent formulation to exploit sequence-level parallelism.

---

## How it works

Given query **Q**, key **K**, value **V** tensors of shape `(B, N, D)`:

```
S      = Σ_t φ(K_t) ⊗ V_t          # D×D state matrix
z      = Σ_t φ(K_t)                 # D normalizer
Out_t  = (φ(Q_t) · S) / (φ(Q_t) · z + ε)
```

The sequence is split into chunks to parallelize the state accumulation across the N dimension. Tile size T is chosen at runtime based on head dimension and SM compute capability:

| Head dim `d` | Tile `T` |
|---|---|
| ≤ 64 | 8 |
| ≤ 128 | 16 |
| > 128, SM ≥ 9 | 32 |
| > 128, SM < 9 | 16 |

---

## Requirements

| Dependency | Version |
|---|---|
| CUDA toolkit | 11.8 – 12.x |
| Python | 3.8+ |
| PyTorch (GPU build) | 2.0+ |
| `setuptools` | any recent |

> The `devcontainer.json` at the repo root configures a ready-to-use CUDA 12.2 Docker environment (see [Container setup](#container-setup)).

---

## Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd AccelLinearAttn
```

### 2. Install the extension

```bash
cd PhotonLib
pip install .
```

This compiles `AccelLinearAttention.cu` and `bindings.cpp` with `nvcc -O3 --use_fast_math` and installs the `photon` Python package.

> **Architecture targeting:** The build defaults to `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"` (Turing, Ampere, Hopper). Override before installing if your GPU is not in this list:
> ```bash
> TORCH_CUDA_ARCH_LIST="8.9" pip install .   # e.g. Ada Lovelace
> ```

---

## Usage

```python
import torch
import photon

# Tensors must be: CUDA, float32, contiguous, same shape (B, N, D)
B, N, D = 8, 2048, 64
q = torch.randn(B, N, D, device="cuda")
k = torch.randn(B, N, D, device="cuda")
v = torch.randn(B, N, D, device="cuda")

out = photon.linear_attention(q, k, v)  # shape (B, N, D)
```

**Constraints checked at runtime:**
- All three tensors must be on CUDA.
- All three tensors must be `float32`.
- All three tensors must be contiguous (call `.contiguous()` if unsure).
- `Q`, `K`, `V` must have identical shapes.

---

## Profiling

`profile.py` at the repo root runs a correctness check and a full performance sweep across batch sizes, sequence lengths, and head dimensions:

```bash
# from repo root
python3 profile.py
```

**Default sweep parameters** (edit at the top of `profile.py` to change):

```python
BATCH_SIZES = [8, 16, 32, 64]
SEQ_LENS    = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
HEAD_DIMS   = [64, 128, 256]
```

Sample output:

```
Device : NVIDIA A100-SXM4-80GB
Memory : 80 GB
Flash path covers d <= 128 (sm80)

==================================================================
  d=64  [flash]              b=8       b=16      b=32      b=64
  n                          ms    GB/s  TFLOPS     ms ...
------------------------------------------------------------------
    2048    0.2ms   45.3GB/s  0.03T  ...
  131072    8.4ms  312.1GB/s  1.24T  ...
==================================================================
```

---

## Container setup

A Dev Container configuration is provided for environments without a local CUDA install:

```bash
# requires Docker with NVIDIA Container Toolkit
# open in VS Code → "Reopen in Container"
```

Or manually:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  bash -c "apt update && apt install -y python3 python3-pip && \
           pip3 install torch --index-url https://download.pytorch.org/whl/cu122 && \
           cd PhotonLib && pip3 install . && cd .. && python3 profile.py"
```

---

## Repository layout

```
AccelLinearAttn/
├── PhotonLib/
│   ├── setup.py                          # build & install script
│   └── Photon/
│       ├── Photon.h                      # public C++ header
│       └── src/
│           ├── AccelLinearAttention.cuh  # kernel declarations
│           ├── AccelLinearAttention.cu   # kernel implementations
│           └── bindings.cpp             # pybind11 module entry point
├── profile.py                            # correctness + benchmark harness
├── libtorch/                             # optional local libtorch (CPU build)
└── .devcontainer/devcontainer.json       # CUDA 12.2 dev container
```
