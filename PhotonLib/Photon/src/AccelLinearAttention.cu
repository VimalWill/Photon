#include "AccelLinearAttention.cuh"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace Photon {

    __device__ __forceinline__ float taylorD1Phi(float x) {
        return 1.0f + x;
    }

    // Grid: (d_stripes, b*num_chunks, 1)  — each block owns one (stripe, chunk, batch).
    __global__ void computeZChunked(
        const float* __restrict__ K,
        float*                    z,
        int n, int d, int num_chunks
    ) {
        int i     = blockIdx.x * blockDim.x + threadIdx.x;
        int chunk = blockIdx.y % num_chunks;
        int b     = blockIdx.y / num_chunks;
        if (i >= d) return;

        int chunk_size = (n + num_chunks - 1) / num_chunks;
        int t_start    = chunk * chunk_size;
        int t_end      = min(t_start + chunk_size, n);

        float acc = 0.f;
        const float* Kb = K + (ptrdiff_t)b * n * d;
        for (int t = t_start; t < t_end; t++)
            acc += taylorD1Phi(Kb[(ptrdiff_t)t * d + i]);
        atomicAdd(&z[(ptrdiff_t)b * d + i], acc);
    }

    // Grid: (d/T, d/T, b*num_chunks) — each block owns one (row-tile, col-tile, n-chunk, batch).
    // Partial sums are atomically folded into S so no second reduction kernel is needed.
    template <int T>
    __global__ void fusedRecurrentStateChunked(
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*                    S,
        int n, int d, int num_chunks
    ) {
        int chunk = blockIdx.z % num_chunks;
        int b     = blockIdx.z / num_chunks;
        int row   = blockIdx.y * T + threadIdx.y;
        int col   = blockIdx.x * T + threadIdx.x;
        if (row >= d || col >= d) return;

        int chunk_size = (n + num_chunks - 1) / num_chunks;
        int t_start    = chunk * chunk_size;
        int t_end      = min(t_start + chunk_size, n);

        ptrdiff_t kv_base = (ptrdiff_t)b * n * d;
        ptrdiff_t s_base  = (ptrdiff_t)b * d * d;

        __shared__ float Stile[T][T];
        Stile[threadIdx.y][threadIdx.x] = 0.f;

        for (int t = t_start; t < t_end; t++) {
            float pk = taylorD1Phi(K[kv_base + t * d + row]);
            float vj = V[kv_base + t * d + col];
            Stile[threadIdx.y][threadIdx.x] += pk * vj;
        }

        // Each (row,col) has a unique S slot, so different chunks race only on
        // their own element — contention is bounded by num_chunks.
        atomicAdd(&S[s_base + row * d + col], Stile[threadIdx.y][threadIdx.x]);
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

        // ~64 tokens per chunk exposes N-parallelism while keeping per-block work substantial.
        int num_chunks = max(1, min((n + 31) / 32, 128));

        computeZChunked<<<dim3((d + 255) / 256, b * num_chunks), 256, 0, s_side>>>(
            K, z, n, d, num_chunks);

        if (d <= 64) {
            constexpr int T = 8;
            fusedRecurrentStateChunked<T><<<dim3((d+T-1)/T,(d+T-1)/T, b*num_chunks), dim3(T,T), 0, s_main>>>(
                K, V, S, n, d, num_chunks);
            cudaEventRecord(ready, s_side);
            cudaStreamWaitEvent(s_main, ready);
            normalize<T><<<dim3((d+T-1)/T,(n+T-1)/T,b), dim3(T,T), 0, s_main>>>(
                Q, S, z, Out, n, d);
        } else if (d <= 128) {
            constexpr int T = 16;
            fusedRecurrentStateChunked<T><<<dim3((d+T-1)/T,(d+T-1)/T, b*num_chunks), dim3(T,T), 0, s_main>>>(
                K, V, S, n, d, num_chunks);
            cudaEventRecord(ready, s_side);
            cudaStreamWaitEvent(s_main, ready);
            normalize<T><<<dim3((d+T-1)/T,(n+T-1)/T,b), dim3(T,T), 0, s_main>>>(
                Q, S, z, Out, n, d);
        } else if (sm_major >= 9) {
            constexpr int T = 32;
            fusedRecurrentStateChunked<T><<<dim3((d+T-1)/T,(d+T-1)/T, b*num_chunks), dim3(T,T), 0, s_main>>>(
                K, V, S, n, d, num_chunks);
            cudaEventRecord(ready, s_side);
            cudaStreamWaitEvent(s_main, ready);
            normalize<T><<<dim3((d+T-1)/T,(n+T-1)/T,b), dim3(T,T), 0, s_main>>>(
                Q, S, z, Out, n, d);
        } else {
            constexpr int T = 16;
            fusedRecurrentStateChunked<T><<<dim3((d+T-1)/T,(d+T-1)/T, b*num_chunks), dim3(T,T), 0, s_main>>>(
                K, V, S, n, d, num_chunks);
            cudaEventRecord(ready, s_side);
            cudaStreamWaitEvent(s_main, ready);
            normalize<T><<<dim3((d+T-1)/T,(n+T-1)/T,b), dim3(T,T), 0, s_main>>>(
                Q, S, z, Out, n, d);
        }

        cudaEventDestroy(ready);

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
