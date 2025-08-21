#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

__global__ void conv1d(const float* a, const float* b, float* c, size_t N, size_t K){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = int(K / 2);
    if (i >= N) return;

    float sum = 0.0f;
    for (int j = 0; j < int(K); j++){
        int idx = int(i) + j - radius;
        if (idx >= 0 && idx < int(N)){
            sum += a[idx] * b[j];
        }
    }
    c[i] = sum;
}


int main() {
    size_t N = 16;   
    size_t K = 3;    

    vector<float> h_a(N), h_b(K), h_c(N);

    for (size_t i = 0; i < N; i++) h_a[i] = static_cast<float>(rand() % 10);
    for (size_t j = 0; j < K; j++) h_b[j] = static_cast<float>(rand() % 5);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, K * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), K * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    conv1d<<<gridSize, blockSize>>>(d_a, d_b, d_c, N, K);

    cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Input: ";
    for (auto v : h_a) cout << v << " ";
    cout << "\nKernel: ";
    for (auto v : h_b) cout << v << " ";
    cout << "\nOutput: ";
    for (auto v : h_c) cout << v << " ";
    cout << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
