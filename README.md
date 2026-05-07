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

## HPC / SLURM deployment

### University of Arizona HPC (Puma)

UA HPC uses the OHPC module system. The default GCC (`gnu8/8.3.0`) is too old for PyTorch — use the system GCC 13 which is on `$PATH` by default.

**Puma GPU:** Tesla V100S-PCIE-32GB → `sm_70`

**Interactive build session:**

```bash
# Request a GPU node
interactive -a <your-group> -p gpu_standard -t 02:00:00 --gres=gpu:1

# Load CUDA (check available versions with: module avail cuda12)
module load cuda12/12.4.1

# Verify GCC is >= 9 (system GCC 13 should be on PATH by default)
gcc --version

# Set architecture list — include sm_70 for V100, extend as needed
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# Build and install in editable mode
cd /path/to/Photon/PhotonLib
pip3 install --no-build-isolation -e .
```

> **Why `--no-build-isolation`?** The HPC PyTorch install lives in `~/.local` rather than a standard virtualenv. Without this flag, pip creates an isolated build environment that can't find the existing PyTorch headers and fails.

> **Why set `TORCH_CUDA_ARCH_LIST`?** The default arch list in `setup.py` targets sm_75+ (Turing and newer). V100 GPUs (Puma) are sm_70 and will raise `no kernel image is available for execution on the device` at runtime if sm_70 is omitted.

**Verify the compiled architectures after build:**

```bash
cuobjdump /path/to/PhotonLib/photon.cpython-*.so | grep arch
# should list: sm_70  sm_75  sm_80  sm_86  sm_90
```

---

### Interactive session (2 GPUs)

```bash
salloc --account=<--> --partition=gpu_standard \
       --nodes=1 --ntasks=2 --time=2:00:00 \
       --job-name=multi-gpu --gres=gpu:2
```

Once the shell is granted, load the required modules and install:

```bash
module load cuda12/12.4.1      # adjust to your cluster's module name

cd /path/to/AccelLinearAttn/PhotonLib
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"
pip3 install --no-build-isolation -e .
```

Then run the profiler:

```bash
cd /path/to/AccelLinearAttn
python3 profile.py
```

> `profile.py` always runs on `cuda:0`. With `--gres=gpu:2` both GPUs are
> reserved; the second is available for experiments you add manually.

---

### Batch job script

Save as `submit.sh` at the repo root and submit with `sbatch submit.sh`:

```bash
#!/bin/bash
#SBATCH --account=<-->
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --job-name=photon-profile
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda12/12.4.1      # adjust to your cluster's module name

# Install if not already present (set arch list to match your GPU)
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"
pip3 install --no-build-isolation --quiet /path/to/AccelLinearAttn/PhotonLib

cd /path/to/AccelLinearAttn
python3 profile.py
```

```bash
mkdir -p logs
sbatch submit.sh
```

Monitor with:

```bash
squeue --me
tail -f logs/<job-id>.out
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
