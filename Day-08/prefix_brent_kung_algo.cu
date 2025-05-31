#define LOAD_SIZE 32
#include <iostream>
#include <cuda_runtime.h>

__global__ void prefix_sum(float* in, float *out, int n){
    int thread_idx = threadIdx.x;
    int i = 2 * blockDim.x * blockIdx.x + thread_idx;

    __shared__ float shared_memo[LOAD_SIZE];

    if(i < n)
        shared_memo[thread_idx] = in[i];
    if(i + blockDim.x < n)
        shared_memo[thread_idx + blockDim.x] = in[i + blockDim.x];
    
    __syncthreads();

    for(int jump=1; jump <= blockDim.x; jump *= 2){
        int j = jump * 2 * (thread_idx + 1) - 1;
        if (j < LOAD_SIZE)
            shared_memo[j] += shared_memo[j-jump];
    }
    __syncthreads();

    for(int jump = LOAD_SIZE / 4; jump >= 1; jump /= 2){
        int j = jump * 2 * (thread_idx + 1) - 1;
        if (j < LOAD_SIZE - jump)
            shared_memo[j + jump] += shared_memo[j];
    }
    __syncthreads();

    if (i < n)
        out[i] = shared_memo[thread_idx];
    if (i < n - blockDim.x)
        out[i+blockDim.x] = shared_memo[thread_idx + blockDim.x];
    
    __syncthreads();
    
}

void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void print_data(float *vec, int n){
    for(int i = 0; i < n; i++)
        printf("%.2f ", vec[i]);
    printf("\n");
}

int main(){
    int n = 10;
    float in[n], out[n];
    for (int i = 0; i < n; i++)
        in[i] = i + 1.0f;
    
    float *din;
    float *dout;
    size_t size = n * sizeof(float);
    cudaMalloc(&din, size);
    cudaMalloc(&dout, size);

    cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);
    CudaError("Failed to copy input data");

    dim3 dimBlock(32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    prefix_sum<<<dimGrid, dimBlock>>>(din, dout, n);
    CudaError("Failed to execute the kernel");

    cudaDeviceSynchronize();

    cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);
    CudaError("Failed to copy output data");

    cudaFree(din);
    cudaFree(dout);

    printf("in:\n");
    print_data(in, n);

    printf("out:\n");
    print_data(out, n);


}