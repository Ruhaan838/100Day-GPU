#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 16;

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockIdx.x + threadIdx.x;

    if (row < M && col < K){
        float sum = 0.0f;
        for(int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void update_kernel(float* X, float* A, float* B, int size, float a, float b, float c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        X[idx] = a * X[idx] + b * A[idx] + c * B[idx];
    }
}

torch::Tensor newton_schulz5_cuda(torch::Tensor G, int steps = 5, float eps = 1e-7){
    int M = G.size(0);
    int N = G.size(1);
    torch::Tensor X = G.clone().to(torch::kCUDA);
    X /= (X.norm() + eps);

    dim3 thread_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    torch::Tensor A = torch::zeros_like(X);
    torch::Tensor B = torch::zeros_like(X);
    float a = 3.4445f, b = -4.7750f, c = 2.0315;

    for(int i = 0; i < steps; i++){
        matmul_kernel<<<grid_size, thread_size>>>(X.data_ptr<float>(), X.data_ptr<float>(), A.data_ptr<float>(), M, N, N);
        matmul_kernel<<<grid_size, thread_size>>>(A.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), M, N, N);
        update_kernel<<<(M * N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(X.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), M * N, a, b, c);
    }
    return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("newton_schulz5_cuda", &newton_schulz5_cuda, "Moun optimizer Newtown-Schulz CUDA");
}