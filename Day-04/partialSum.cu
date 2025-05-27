#include <stdio.h>

__global__ void partialSum(int *in, int *out, int n){
    extern __shared__ int sharedMemory[]; // NEW: __shared__ this keyword is allow share the memory.

    int t = threadIdx.x;
    int block = blockDim.x;
    int index = blockIdx.x * block * 2 + t;

    if (index < n){
        sharedMemory[t] = in[index] + in[index+block];
        __syncthreads(); // NEW: __syncthreads() is stabelze the hang or produce unintended side effects.
        
        // inclusive scan = partial_sum
        for (int st = 1; st < block; st *= 2){
            int temp = 0;
            if (t >= st)
                temp = sharedMemory[t - st];
            __syncthreads();
            sharedMemory[t] += temp;
            __syncthreads();
        }
        out[index] = sharedMemory[t];
    }

}

void print_vec(const int* vec, int n){
    for (int i = 0; i < n; i++)
        printf("%d ", vec[i]);
    printf("\n");
}

int main(){
    const int n = 16;
    const int blocksize = 8;

    int host_in[n];
    int host_out[n];

    for (int i = 0; i < n; i++)
        host_in[i] = i+1;
    
    int *d_in, *d_out;
    size_t size = n * sizeof(int);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, host_in, size, cudaMemcpyHostToDevice);
    
    int gridsize = n / blocksize;
    int shared_mem = blocksize * sizeof(int);

    partialSum<<<gridsize, blocksize, shared_mem>>>(d_in, d_out, n); // learn more about the <<<...>>> what that 3rd args dose? 

    cudaMemcpy(host_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("In(vec):");
    print_vec(host_in, n);

    printf("Out(vec):");
    print_vec(host_out, n);

    cudaFree(d_in);
    cudaFree(d_out);

}