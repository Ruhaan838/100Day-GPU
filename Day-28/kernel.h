#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
__global__ void add_kernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel(float* a, float* b, float* out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}

#endif