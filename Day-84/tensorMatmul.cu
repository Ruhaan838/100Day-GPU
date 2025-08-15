#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void tensor_matmul(
    const float* a,
    const float* b,
    float* c,
    size_t B_dim,
    size_t I_dim,
    size_t J_dim,
    size_t L_dim,
    size_t K_dim
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_ele = B_dim * I_dim * J_dim * K_dim;

    if (idx >= total_ele) return;
    int k = idx % K_dim;
    int j = (idx / K_dim) % J_dim;
    int i = (idx / (K_dim * J_dim)) % I_dim;
    int bs = idx / (I_dim * J_dim * K_dim);

    size_t c_idx = ((bs * I_dim + i) * J_dim + j) * K_dim + k;
    size_t a_base = ((bs * I_dim + i) * J_dim + j) * L_dim;
    float sum = 0.0f;
    for(int l = 0; l < L_dim; l++){
        sum += a[a_base + l] * b[l * K_dim + k];
    }
    c[c_idx] = sum;
}

int main() {
    size_t B_dim = 2, I_dim = 3, J_dim = 4, L_dim = 5, K_dim = 6;
    size_t size_a = B_dim * I_dim * J_dim * L_dim;
    size_t size_b = L_dim * K_dim;
    size_t size_c = B_dim * I_dim * J_dim * K_dim;

    std::vector<float> h_a(size_a), h_b(size_b), h_c(size_c);

    for (auto &x : h_a) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto &x : h_b) x = static_cast<float>(rand()) / RAND_MAX;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a * sizeof(float));
    cudaMalloc(&d_b, size_b * sizeof(float));
    cudaMalloc(&d_c, size_c * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int totalThreads = size_c;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    tensor_matmul<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, B_dim, I_dim, J_dim, L_dim, K_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tensor C shape: [" << B_dim << ", " << I_dim << ", " << J_dim << ", " << K_dim << "]\n";

    for (size_t b = 0; b < B_dim; b++) {
        std::cout << "Batch " << b << ":\n";
        for (size_t i = 0; i < I_dim; i++) {
            std::cout << "  I " << i << ":\n";
            for (size_t j = 0; j < J_dim; j++) {
                std::cout << "    [ ";
                for (size_t k = 0; k < K_dim; k++) {
                    size_t idx = ((b * I_dim + i) * J_dim + j) * K_dim + k;
                    std::cout << h_c[idx] << (k + 1 < K_dim ? ", " : "");
                }
                std::cout << " ]\n";
            }
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}