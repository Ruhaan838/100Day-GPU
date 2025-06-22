#include <stdio.h>
#include <hip/hip_runtime.h>

__global__ void partialSum(int *in, int *out, int n){
    extern __shared__ int sharedMemory[];

    int t = threadIdx.x;
    int block = blockDim.x;
    int index = blockIdx.x * block * 2 + t;

    if (index < n){
        sharedMemory[t] = in[index] + in[index + block];
        __syncthreads();
        
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

    int host_in[n], host_out[n];

    for (int i = 0; i < n; i++)
        host_in[i] = i + 1;

    int *d_in, *d_out;
    size_t size = n * sizeof(int);

    hipMalloc(&d_in, size);
    hipMalloc(&d_out, size);

    hipMemcpy(d_in, host_in, size, hipMemcpyHostToDevice);

    int gridsize = n / blocksize;
    int shared_mem = blocksize * sizeof(int);

    hipLaunchKernelGGL(partialSum, dim3(gridsize), dim3(blocksize), shared_mem, 0, d_in, d_out, n);

    hipMemcpy(host_out, d_out, size, hipMemcpyDeviceToHost);

    printf("In(vec): ");
    print_vec(host_in, n);

    printf("Out(vec): ");
    print_vec(host_out, n);

    hipFree(d_in);
    hipFree(d_out);

    return 0;
}
