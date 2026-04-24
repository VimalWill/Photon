#include "AccelLinearAttention.cuh"
#include <cuda_runtime.h>

#if   __CUDA_ARCH__ >= 900
  #define TILE 32
#elif __CUDA_ARCH__ >= 800
  #define TILE 32
#else
  #define TILE 16
#endif

namespace Photon {

    __device__ __forceinline__ float taylorD1Phi(float x) {
        return 1.0f + x;
    }

    template <int T>
    __global__ void fusedRecurrentState(
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*                    S,
        float*                    z,
        int n,
        int d
    ) {
        int row = blockIdx.y * T + threadIdx.y;
        int col = blockIdx.x * T + threadIdx.x;
        int b   = blockIdx.z;

        if (row >= d || col >= d) return;

        int kv_base = b * n * d;
        int s_base  = b * d * d;
        int z_base  = b * d;

        __shared__ float Stile[T][T];
        Stile[threadIdx.y][threadIdx.x] = S[s_base + row * d + col];
        __syncthreads();

        for (int t = 0; t < n; t++) {
            float pk = taylorD1Phi(K[kv_base + t * d + row]);
            float vj = V[kv_base + t * d + col];
            Stile[threadIdx.y][threadIdx.x] += pk * vj;
            if (blockIdx.x == 0 && threadIdx.x == 0)
                z[z_base + row] += pk;
        }
        __syncthreads();

        S[s_base + row * d + col] = Stile[threadIdx.y][threadIdx.x];
    }

    template <int T>
    __global__ void normalize(
        const float* __restrict__ Q,
        const float* __restrict__ S,
        const float* __restrict__ z,
        float*                    Out,
        int n,
        int d
    ) {
        __shared__ float tileA[T][T];
        __shared__ float tileB[T][T];

        int row = blockIdx.y * T + threadIdx.y;
        int col = blockIdx.x * T + threadIdx.x;
        int b   = blockIdx.z;

        if (row >= n || col >= d) return;

        int qo_base = b * n * d;
        int s_base  = b * d * d;
        int z_base  = b * d;

        float num   = 0.0f;
        float denom = 0.0f;

        for (int k = 0; k < d; k += T) {
            int tx = k + threadIdx.x;
            int ty = k + threadIdx.y;
            tileA[threadIdx.y][threadIdx.x] = (tx < d) ? taylorD1Phi(Q[qo_base + row * d + tx]) : 0.f;
            tileB[threadIdx.y][threadIdx.x] = (ty < d) ? S[s_base + ty * d + col]               : 0.f;
            __syncthreads();

            #pragma unroll
            for (int t = 0; t < T; t++) {
                num   += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];
                denom += tileA[threadIdx.y][t] * z[z_base + k + t];
            }
            __syncthreads();
        }

        Out[qo_base + row * d + col] = num / (denom + 1e-6f);
    }

    __host__ void LinearAttn(
        const float* Q,
        const float* K,
        const float* V,
        float*       Out,
        int b, int n, int d
    ) {
        float *S, *z;
        cudaMalloc(&S, b * d * d * sizeof(float));
        cudaMalloc(&z, b * d     * sizeof(float));
        cudaMemset(S, 0, b * d * d * sizeof(float));
        cudaMemset(z, 0, b * d     * sizeof(float));

        int dev, sm_major;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);

        if (sm_major >= 8) {
            fusedRecurrentState<32><<<dim3((d+31)/32,(d+31)/32,b), dim3(32,32)>>>(K, V, S, z, n, d);
            normalize<32>        <<<dim3((d+31)/32,(n+31)/32,b), dim3(32,32)>>>(Q, S, z, Out, n, d);
        } else {
            fusedRecurrentState<16><<<dim3((d+15)/16,(d+15)/16,b), dim3(16,16)>>>(K, V, S, z, n, d);
            normalize<16>        <<<dim3((d+15)/16,(n+15)/16,b), dim3(16,16)>>>(Q, S, z, Out, n, d);
        }

        cudaDeviceSynchronize();
        cudaFree(S);
        cudaFree(z);
    }
}

torch::Tensor AccelLinearAttention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value
) {
    TORCH_CHECK(query.is_cuda(),   "query must be a CUDA tensor");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "only float32 supported");

    const int b = query.size(0);
    const int n = query.size(1);
    const int d = query.size(2);

    auto out = torch::zeros({b, n, d}, query.options());

    Photon::LinearAttn(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        out.data_ptr<float>(),
        b, n, d
    );

    return out;
}
