#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__global__ void update_step(float *input, float *noise, float *result, float a, float b, float abar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float inv_sqrt_a = 1.0f / sqrtf(a);
        float scaled_eps = b / sqrtf(1.0f - abar);
        result[idx] = inv_sqrt_a * (input[idx] - scaled_eps * noise[idx]);
    }
}

int main() {
    int total = 1024 * 1024 * 3;
    float a = 0.9f, b = 0.1f, abar = 0.5f;

    float *host_input = (float*)malloc(total * sizeof(float));
    float *host_noise = (float*)malloc(total * sizeof(float));
    float *host_result = (float*)malloc(total * sizeof(float));

    float *dev_input, *dev_noise, *dev_result;
    cudaMalloc(&dev_input, total * sizeof(float));
    cudaMalloc(&dev_noise, total * sizeof(float));
    cudaMalloc(&dev_result, total * sizeof(float));

    dim3 block(1024);
    dim3 grid((total + block.x - 1) / block.x);

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    int warm = 10;
    srand(time(NULL));
    for (int i = 0; i < warm; i++) {
        for (int j = 0; j < total; j++) {
            host_input[j] = ((float)rand() / RAND_MAX) * 2 - 1;
            host_noise[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        cudaMemcpy(dev_input, host_input, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_noise, host_noise, total * sizeof(float), cudaMemcpyHostToDevice);
        update_step<<<grid, block>>>(dev_input, dev_noise, dev_result, a, b, abar, total);
        cudaDeviceSynchronize();
    }

    int iters = 1000;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < total; j++) {
            host_input[j] = ((float)rand() / RAND_MAX) * 2 - 1;
            host_noise[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        cudaMemcpy(dev_input, host_input, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_noise, host_noise, total * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(t_start);
        update_step<<<grid, block>>>(dev_input, dev_noise, dev_result, a, b, abar, total);
        cudaEventRecord(t_stop);
        cudaEventSynchronize(t_stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, t_start, t_stop);
        total_ms += elapsed;
    }

    float avg_ms = total_ms / iters;
    printf("CUDA Kernel Time: %f ms\n", avg_ms);

    cudaMemcpy(host_result, dev_result, total * sizeof(float), cudaMemcpyDeviceToHost);

    free(host_input); free(host_noise); free(host_result);
    cudaFree(dev_input); cudaFree(dev_noise); cudaFree(dev_result);

    return 0;
}
