#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

static constexpr float EPS = 1e-10f;

template <int UNROLL>
__global__ __launch_bounds__ (256, 4)
void kl_divergence_kernel(const float* preds, const float* tgt, float* output, size_t n){
    const size_t base = (size_t)blockIdx.x * blockDim.x * UNROLL + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x * UNROLL;

    size_t idx = base;

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i, idx += blockDim.x){
        if (idx < n){
            float t = __ldg(&tgt[idx]) + EPS;
            float p = __ldg(&preds[idx]) + EPS;

            float diff = __logf(t) - __logf(p);
            output[idx] = __fmaf_rn(t, diff, 0.0f);
        }
    }
}

int main(){
    std::srand((unsigned)std::time(nullptr));

    const size_t n = 400;
    float* h_preds  = (float*)malloc(n * sizeof(float));
    float* h_tgt    = (float*)malloc(n * sizeof(float));
    float* h_output = (float*)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i){
        h_preds[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_tgt[i]   = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *d_preds = nullptr, *d_tgt = nullptr, *d_output = nullptr;
    cudaMalloc(&d_preds,  n * sizeof(float));
    cudaMalloc(&d_tgt,    n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_preds, h_preds, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt,   h_tgt,   n * sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 256;
    constexpr int UNROLL = 4;
    const size_t ele_per_block = (size_t)threads * UNROLL;
    const int blocks = static_cast<int>((n + ele_per_block - 1) / ele_per_block);

    kl_divergence_kernel<UNROLL><<<blocks, threads>>>(d_preds, d_tgt, d_output, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    double kl_sum = 0.0;
    for (size_t i = 0; i < n; ++i) kl_sum += static_cast<double>(h_output[i]);

    printf("Kullback-Leibler Divergence %f\n", kl_sum);

    cudaFree(d_preds);
    cudaFree(d_tgt);
    cudaFree(d_output);
    free(h_preds);
    free(h_tgt);
    free(h_output);

    return 0;
}
