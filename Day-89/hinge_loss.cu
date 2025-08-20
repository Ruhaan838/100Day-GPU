#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void hinge_kenel(const float* pred, const float* tgt, float* out, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        float p = pred[idx] * tgt[idx];
        out[idx] = fmaxf(0.0f, 1.0f - p);
    }
}

int main() {
    const size_t N = 1 << 16;  
    const size_t SIZE = N * sizeof(float);

    float *h_pred = new float[N];
    float *h_tgt  = new float[N];
    float *h_out  = new float[N];

    std::srand(std::time(nullptr));
    for (size_t i = 0; i < N; i++) {
        h_pred[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f; 
        h_tgt[i]  = (rand() % 2 == 0) ? 1.0f : -1.0f;    
    }

    float *d_pred, *d_tgt, *d_out;
    cudaMalloc(&d_pred, SIZE);
    cudaMalloc(&d_tgt, SIZE);
    cudaMalloc(&d_out, SIZE);

    cudaMemcpy(d_pred, h_pred, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, h_tgt, SIZE, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    hinge_kenel<<<gridSize, blockSize>>>(d_pred, d_tgt, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << "pred=" << h_pred[i]
                  << " tgt=" << h_tgt[i]
                  << " out=" << h_out[i] << std::endl;
    }

    cudaFree(d_pred);
    cudaFree(d_tgt);
    cudaFree(d_out);
    delete[] h_pred;
    delete[] h_tgt;
    delete[] h_out;

    return 0;
}
