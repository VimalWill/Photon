#include "cuda.h"
#define TILE 16

namespace Photon {

    __device__ __forceinline__ void taylorD1Phi(
        const float* in,
        float*       out,
        int          n,
        int          d
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n * d)
            out[idx] = 1.0f + in[idx];
    }

    __global__ void fusedRecurrentState(
        const float* Kphi,
        const float* V,
        float*       S,
        float*       z,
        int          n,
        int          d
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int b   = blockIdx.z;

        if (col >= d) return;

        int kv_base = b * n * d;
        int s_base  = b * d * d;
        int z_base  = b * d;

        for (int t = 0; t < n; t++) {
            float pk = Kphi[kv_base + t * d + col];

            z[z_base + col] += pk;

            for (int j = 0; j < d; j++)
                S[s_base + col * d + j] += pk * V[kv_base + t * d + j];
        }
    }

    __global__ void normalize(
        const float* Qphi,
        const float* S,
        const float* z,
        float*       Out,
        int          n,
        int          d
    ) {
        __shared__ float tileA[TILE][TILE];
        __shared__ float tileB[TILE][TILE];

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int b   = blockIdx.z;

        if (row >= n || col >= d) return;

        int qo_base = b * n * d;
        int s_base  = b * d * d;
        int z_base  = b * d;

        float num   = 0.0f;
        float denom = 0.0f;

        for (int k = 0; k < d; k += TILE) {
            tileA[threadIdx.y][threadIdx.x] = Qphi[qo_base + row * d + k + threadIdx.x];
            tileB[threadIdx.y][threadIdx.x] = S[s_base + (k + threadIdx.y) * d + col];
            __syncthreads();

            for (int t = 0; t < TILE; t++) {
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
        int          b,
        int          n,
        int          d
    ) {
        float *Qphi, *Kphi, *S, *z;

        cudaMalloc(&Qphi, b * n * d * sizeof(float));
        cudaMalloc(&Kphi, b * n * d * sizeof(float));
        cudaMalloc(&S,    b * d * d * sizeof(float));
        cudaMalloc(&z,    b * d     * sizeof(float));

        cudaMemset(Qphi, 0, b * n * d * sizeof(float));
        cudaMemset(Kphi, 0, b * n * d * sizeof(float));
        cudaMemset(S,    0, b * d * d * sizeof(float));
        cudaMemset(z,    0, b * d     * sizeof(float));

        dim3 phiThreads(TILE * TILE);
        dim3 phiBlocks((b * n * d + TILE * TILE - 1) / (TILE * TILE));

        taylorD1Phi<<<phiBlocks, phiThreads>>>(Q, Qphi, b * n, d);
        taylorD1Phi<<<phiBlocks, phiThreads>>>(K, Kphi, b * n, d);

        dim3 stateThreads(TILE * TILE);
        dim3 stateBlocks(
            (d + TILE * TILE - 1) / (TILE * TILE),
            1,
            b
        );

        fusedRecurrentState<<<stateBlocks, stateThreads>>>(Kphi, V, S, z, n, d);

        dim3 normThreads(TILE, TILE);
        dim3 normBlocks(
            (d + TILE - 1) / TILE,
            (n + TILE - 1) / TILE,
            b
        );

        normalize<<<normBlocks, normThreads>>>(Qphi, S, z, Out, n, d);

        cudaDeviceSynchronize();

        cudaFree(Qphi);
        cudaFree(Kphi);
        cudaFree(S);
        cudaFree(z);
    }
}