//upper triangular matrix multiplication 

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void upper_tri_mm_kernel(const float* a, const float* b, float* c, size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n){
        if (row <= col){
            float sum = 0.0f;

            for(int k = row; k <= col; ++k){
                sum += a[row * n + k] * b[k * n + col];
            }

            c[row * n + col] = sum;
        } else {
            c[row * n + col] = 0.0f;
        }
    }
}

int main(){
    size_t n = 10;
    size_t size = n * n * sizeof(float);
    float input_a[n][n], input_b[n][n], output_c[n][n];


    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            input_a[i][j] = 10 * i + j;
            input_b[i][j] = 10 * i + j;
        }
    }

    float *d_input_a, *d_input_b, *d_output_c;

    cudaMalloc((void**)&d_input_a, size);
    cudaMalloc((void**)&d_input_b, size);
    cudaMalloc((void**)&d_output_c, size);

    cudaMemcpy(d_input_a, input_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, input_b, size, cudaMemcpyHostToDevice);
    

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    upper_tri_mm_kernel<<<gridDim, blockDim>>>(d_input_a, d_input_b, d_output_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output_c, d_output_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%f", output_c[i][j]);
        }
        printf("\n");
    }
}