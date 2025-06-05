#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <math.h>

__global__ void gelu_kernel(float* data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // erff(x) = (2 / sqrt(pi)) * integrate from t = 0 to x of exp(−t^2) dt
    if (i < size)
        data[i] = 0.5f * data[i] * (1.0f + erff(data[i] / sqrtf(2.0f)));
}

int main(){
    const int N = 1000000;
    float data[N];

    for (int i = 0; i < N; i++)
        data[i] = -1.0f * (float)i / 2;

    for (int i = 0; i < 10; ++i)
        std::cout << "data[" << i << "]: " << data[i] << "\n\n";

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread_per_block(256);
    dim3 block_per_grid((N + thread_per_block.x - 1) / thread_per_block.x);

    gelu_kernel<<<block_per_grid, thread_per_block>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    for (int i = 0; i < 10; ++i)
        std::cout << "data_gelu[" << i << "]: " << data[i] << "\n\n";

    return 0;
}
