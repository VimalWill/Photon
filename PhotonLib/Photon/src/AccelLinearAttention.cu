#include "AccelLinearAttention.cuh"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace Photon {

    __device__ __forceinline__ float taylorD1Phi(float x) {
        return 1.0f + x;
    }

    // One block per batch element; smem holds S[d*d], z[d], and two d-wide scratch
    // buffers.  kBuf doubles as phi(Q[t]) in pass 2; vBuf[0] holds the denom scalar.
    __global__ void flashLinearAttn(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*                    Out,
        int n, int d
    ) {
        int b = blockIdx.x;

        extern __shared__ float smem[];
        float* S    = smem;
        float* z    = S + d * d;
        float* kBuf = z + d;
        float* vBuf = kBuf + d;

        const float* Qb = Q   + (ptrdiff_t)b * n * d;
        const float* Kb = K   + (ptrdiff_t)b * n * d;
        const float* Vb = V   + (ptrdiff_t)b * n * d;
        float*       Ob = Out + (ptrdiff_t)b * n * d;

        for (int i = threadIdx.x; i < d * d; i += blockDim.x) S[i] = 0.f;
        for (int i = threadIdx.x; i < d;     i += blockDim.x) z[i] = 0.f;
        __syncthreads();

        for (int t = 0; t < n; t++) {
            for (int i = threadIdx.x; i < d; i += blockDim.x) {
                kBuf[i] = taylorD1Phi(Kb[t * d + i]);
                vBuf[i] = Vb[t * d + i];
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < d * d; idx += blockDim.x)
                S[idx] += kBuf[idx / d] * vBuf[idx % d];
            for (int i = threadIdx.x; i < d; i += blockDim.x)
                z[i] += kBuf[i];
            __syncthreads();
        }

        for (int t = 0; t < n; t++) {
            for (int i = threadIdx.x; i < d; i += blockDim.x)
                kBuf[i] = taylorD1Phi(Qb[t * d + i]);
            __syncthreads();

            // denom is the same for every output column; compute it once.
            if (threadIdx.x == 0) {
                float den = 0.f;
                for (int k = 0; k < d; k++) den += kBuf[k] * z[k];
                vBuf[0] = den;
            }
            __syncthreads();

            float denom = vBuf[0] + 1e-6f;
            for (int col = threadIdx.x; col < d; col += blockDim.x) {
                float num = 0.f;
                for (int k = 0; k < d; k++)
                    num += kBuf[k] * S[k * d + col];
                Ob[t * d + col] = num / denom;
            }
            __syncthreads();
        }
    }

    __global__ void computeZ(
        const float* __restrict__ K,
        float*                    z,
        int n, int d
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (i >= d) return;

        float acc = 0.f;
        const float* Kb = K + (ptrdiff_t)b * n * d;
        for (int t = 0; t < n; t++)
            acc += taylorD1Phi(Kb[(ptrdiff_t)t * d + i]);
        z[(ptrdiff_t)b * d + i] = acc;
    }

    template <int T>
    __global__ void fusedRecurrentState(
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*                    S,
        int n, int d
    ) {
        int row = blockIdx.y * T + threadIdx.y;
        int col = blockIdx.x * T + threadIdx.x;
        int b   = blockIdx.z;
        if (row >= d || col >= d) return;

        ptrdiff_t kv_base = (ptrdiff_t)b * n * d;
        ptrdiff_t s_base  = (ptrdiff_t)b * d * d;

        __shared__ float Stile[T][T];
        Stile[threadIdx.y][threadIdx.x] = S[s_base + row * d + col];

        for (int t = 0; t < n; t++) {
            float pk = taylorD1Phi(K[kv_base + t * d + row]);
            float vj = V[kv_base + t * d + col];
            Stile[threadIdx.y][threadIdx.x] += pk * vj;
        }

        S[s_base + row * d + col] = Stile[threadIdx.y][threadIdx.x];
    }

    template <int T>
    __global__ void normalize(
        const float* __restrict__ Q,
        const float* __restrict__ S,
        const float* __restrict__ z,
        float*                    Out,
        int n, int d
    ) {
        __shared__ float tileA[T][T];
        __shared__ float tileB[T][T];
        __shared__ float tileZ[T];

        int row = blockIdx.y * T + threadIdx.y;
        int col = blockIdx.x * T + threadIdx.x;
        int b   = blockIdx.z;
        if (row >= n || col >= d) return;

        ptrdiff_t qo_base = (ptrdiff_t)b * n * d;
        ptrdiff_t s_base  = (ptrdiff_t)b * d * d;
        ptrdiff_t z_base  = (ptrdiff_t)b * d;

        float num   = 0.f;
        float denom = 0.f;

        for (int k = 0; k < d; k += T) {
            int tx = k + threadIdx.x;
            int ty = k + threadIdx.y;
            tileA[threadIdx.y][threadIdx.x] = (tx < d) ? taylorD1Phi(Q[qo_base + row * d + tx]) : 0.f;
            tileB[threadIdx.y][threadIdx.x] = (ty < d) ? S[s_base + ty * d + col]               : 0.f;
            if (threadIdx.y == 0)
                tileZ[threadIdx.x] = (tx < d) ? z[z_base + tx] : 0.f;
            __syncthreads();

            #pragma unroll
            for (int t = 0; t < T; t++) {
                num   += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];
                denom += tileA[threadIdx.y][t] * tileZ[t];
            }
            __syncthreads();
        }

        Out[qo_base + row * d + col] = num / (denom + 1e-6f);
    }

    __host__ void LinearAttn(
        const float* Q, const float* K, const float* V,
        float* Out,
        int b, int n, int d,
        torch::DeviceIndex device_index
    ) {
        cudaStream_t s_main = at::cuda::getCurrentCUDAStream(device_index);

        int dev, sm_major;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);

        // sm90 (H100/H200) has 228 KB configurable smem per SM, so S[d*d] fits
        // for d=128 (~67 KB). Earlier hardware caps out at 48 KB → d=64 only.
        const int flashMaxD = (sm_major >= 9) ? 128 : 64;

        if (d <= flashMaxD) {
            size_t smem = ((size_t)d * d + 3 * d) * sizeof(float);
            int threads = ((d + 31) / 32) * 32;
            if (smem > 49152) {
                // Need to opt in to the device's extended shared memory limit.
                // Query the actual maximum rather than hardcoding an arch constant.
                int max_smem = 49152;
                cudaDeviceGetAttribute(&max_smem,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
                TORCH_CHECK((size_t)max_smem >= smem,
                    "flash path needs ", smem, " B smem but device max is ", max_smem, " B");
                TORCH_CHECK(
                    cudaFuncSetAttribute(flashLinearAttn,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        max_smem) == cudaSuccess,
                    "cudaFuncSetAttribute failed");
            }
            flashLinearAttn<<<b, threads, smem, s_main>>>(Q, K, V, Out, n, d);
        } else {
            auto opts = torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(torch::kCUDA, device_index);
            auto S_t = torch::zeros({b, d, d}, opts);
            auto z_t = torch::zeros({b, d},   opts);
            float* S = S_t.data_ptr<float>();
            float* z = z_t.data_ptr<float>();

            static cudaStream_t s_side_per_dev[64] = {};
            if (!s_side_per_dev[dev])
                cudaStreamCreateWithFlags(&s_side_per_dev[dev], cudaStreamNonBlocking);
            cudaStream_t s_side = s_side_per_dev[dev];

            cudaEvent_t ready;
            cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);

            computeZ<<<dim3((d + 255) / 256, b), 256, 0, s_side>>>(K, z, n, d);

            if (sm_major >= 9) {
                fusedRecurrentState<32><<<dim3((d+31)/32,(d+31)/32,b), dim3(32,32), 0, s_main>>>(K, V, S, n, d);
                cudaEventRecord(ready, s_side);
                cudaStreamWaitEvent(s_main, ready);
                normalize<32><<<dim3((d+31)/32,(n+31)/32,b), dim3(32,32), 0, s_main>>>(Q, S, z, Out, n, d);
            } else {
                fusedRecurrentState<16><<<dim3((d+15)/16,(d+15)/16,b), dim3(16,16), 0, s_main>>>(K, V, S, n, d);
                cudaEventRecord(ready, s_side);
                cudaStreamWaitEvent(s_main, ready);
                normalize<16><<<dim3((d+15)/16,(n+15)/16,b), dim3(16,16), 0, s_main>>>(Q, S, z, Out, n, d);
            }

            cudaEventDestroy(ready);
        }

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel error in LinearAttn: ", cudaGetErrorString(err));
    }
}

torch::Tensor AccelLinearAttention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value
) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 &&
                key  .scalar_type() == torch::kFloat32 &&
                value.scalar_type() == torch::kFloat32,
                "only float32 supported");
    TORCH_CHECK(key  .sizes() == query.sizes(), "key must match query shape");
    TORCH_CHECK(value.sizes() == query.sizes(), "value must match query shape");

    const int b = query.size(0);
    const int n = query.size(1);
    const int d = query.size(2);

    auto out = torch::zeros({b, n, d}, query.options());

    Photon::LinearAttn(
        query.data_ptr<float>(),
        key  .data_ptr<float>(),
        value.data_ptr<float>(),
        out  .data_ptr<float>(),
        b, n, d,
        query.device().index()
    );

    return out;
}
