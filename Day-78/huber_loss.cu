#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void huber_loss_kernel(const float* preds, const float* targets, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x; // FIXED

    for (size_t i = idx; i < n; i += stride) {
        float diff = preds[i] - targets[i];
        float abs_diff = fabsf(diff);

        if (abs_diff < 1.0f) {
            output[i] = 0.5f * diff * diff;
        } else {
            output[i] = abs_diff - 0.5f;
        }
    }
}

int main() {
    const size_t N = 10;
    float h_preds[N]   = {1.0f, 2.0f, 3.0f, 4.5f, 5.2f, 6.0f, 7.1f, 8.0f, 9.3f, 10.0f};
    float h_targets[N] = {1.5f, 1.8f, 3.2f, 4.0f, 5.5f, 5.8f, 7.0f, 8.5f, 9.0f, 9.5f};
    float h_output[N];

    float *d_preds, *d_targets, *d_output;

    cudaMalloc(&d_preds,   N * sizeof(float));
    cudaMalloc(&d_targets, N * sizeof(float));
    cudaMalloc(&d_output,  N * sizeof(float));

    cudaMemcpy(d_preds,   h_preds,   N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    huber_loss_kernel<<<blocks, threads>>>(d_preds, d_targets, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Huber loss results:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << "Idx " << i << ": " << h_output[i] << "\n";
    }

    cudaFree(d_preds);
    cudaFree(d_targets);
    cudaFree(d_output);

    return 0;
}
