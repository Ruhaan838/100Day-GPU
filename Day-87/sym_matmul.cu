#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 16

__global__ void symMatmul_kernel(const float* a, const float* b, float *c, size_t n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for(size_t k = 0; k < n; k++)
        sum += a[row * n + k] * b[k * n + col];
    c[row * n + col] = sum;
}

int main(){
    size_t n = 4; 
    size_t bytes = n * n * sizeof(float);

    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);

    for (size_t i = 0; i < n * n; i++) {
        a[i] = static_cast<float>(i + 1);     
        b[i] = static_cast<float>((i % n) + 1); 
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    
    symMatmul_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    
    printf("C\n");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }

    
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
