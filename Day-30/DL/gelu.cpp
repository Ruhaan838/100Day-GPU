#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>

__global__ void gelu_kernel(float* data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    if (i < size)
        data[i] = 0.5f * data[i] * (1.0f + erff(data[i] / sqrtf(2.0f)));
}

int main(){
    const int N = 1000000;
    float* data = new float[N];

    for (int i = 0; i < N; i++)
        data[i] = -1.0f * (float)i / 2;

    for (int i = 0; i < 10; ++i)
        std::cout << "data[" << i << "]: " << data[i] << "\n";

    float* d_data;
    hipMalloc(&d_data, N * sizeof(float));
    hipMemcpy(d_data, data, N * sizeof(float), hipMemcpyHostToDevice);

    dim3 thread_per_block(256);
    dim3 block_per_grid((N + thread_per_block.x - 1) / thread_per_block.x);

    hipLaunchKernelGGL(gelu_kernel, block_per_grid, thread_per_block, 0, 0, d_data, N);
    hipDeviceSynchronize();

    hipMemcpy(data, d_data, N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_data);

    std::cout << "\nGELU Output:\n";
    for (int i = 0; i < 10; ++i)
        std::cout << "data_gelu[" << i << "]: " << data[i] << "\n";

    delete[] data;
    return 0;
}
