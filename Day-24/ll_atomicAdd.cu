#include <stdio.h>
#include <cuda_runtime.h>

__device__ long long LLAtomicAdd(long long *data, long long val){
    unsigned long long *udata = (unsigned long long *)data;
    unsigned long long old = *udata, assumed;
    do {
        assumed = old;
        old = atomicCAS(udata, assumed, assumed + val);
    } while (assumed != old);
    return (long long)old;
}

__global__ void LLAtomicAddKernel(long long *data){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    LLAtomicAdd(data, t);
}

int main(){
    long long *d_data;
    long long data = 0;
    cudaMalloc(&d_data, sizeof(long long));
    cudaMemcpy(d_data, &data, sizeof(long long), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size(4);

    LLAtomicAddKernel<<<grid_size, block_size>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(&data, d_data, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Final Value: %lld\n", data);

    cudaFree(d_data);
    return 0;
}