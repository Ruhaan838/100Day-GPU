#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float softplus(float x){
    return log1pf(expf(x)); // log (1 + x);
}

__global__ void softplus_kernel(const float* input, float* output, size_t n, size_t m){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_ele = n * m;
    if (idx < total_ele)
        output[idx] = softplus(input[idx]);
}

int main(){
    size_t n = 1024, m = 512;
    size_t total_ele = n * m;
    size_t bytes = total_ele * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    for(size_t i = 0; i < total_ele; i++) h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    size_t threads = 256;
    size_t blocks = (total_ele + threads - 1) / threads;
    softplus_kernel<<<blocks, threads>>>(d_input, d_output, n, m);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < 10; i++){
        printf("%f ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
