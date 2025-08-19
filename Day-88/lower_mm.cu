#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

const int BLOCK_SIZE = 16;

__global__ void lower_tri_mm_kernel(const float* a, const float* b, float* c, size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    if (col > row){
        c[row * n + col] = 0.0f;
    } else {
        float sum = 0.0f;
        for(int k = col; k <= row; k++){
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int n = 4;
    size_t size = n * n * sizeof(float);

    float *h_a = new float[n * n];
    float *h_b = new float[n * n];
    float *h_c = new float[n * n];

    std::srand(std::time(0));
    for (int i = 0; i < n * n; i++) {
        h_a[i] = static_cast<float>(std::rand() % 10);
        h_b[i] = static_cast<float>(std::rand() % 10);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1)/BLOCK_SIZE,(n + BLOCK_SIZE - 1)/BLOCK_SIZE);
    lower_tri_mm_kernel<<<grid,block>>>(d_a,d_b,d_c,n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%f ", h_c[i*n+j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}