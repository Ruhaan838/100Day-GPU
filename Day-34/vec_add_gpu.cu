#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define UNROLL_FACTOR 4

__global__ void vector_add(double *a, double *b, double *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = n - (n % UNROLL_FACTOR);
    for (; i < limit; i += blockDim.x * gridDim.x){
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < n; i ++){
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv){
    int n = atoi(argv[1]);
    double *a, *b, *c;
    double *d_a, *d_b, *d_c;

    size_t size = n * sizeof(double);
    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c = (double *)malloc(size);
    
    for (int i = 0; i < n; i++){
        a[i] = i;
        b[i] = i * 2.0;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken (GPU): %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}