#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
const int BLOCK_SIZE = 256;

__global__ void compute_squared_norm(const float* data, float* result, int size){
    __shared__ float cache[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    float temp = 0.0f;
    
    while (tid < size) {
        temp += data[tid] * data[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[local_tid] = temp;
    __syncthreads();

    for(int i = BLOCK_SIZE / 2; i > 0; i >>=1){
        if(local_tid < i) 
            cache[local_tid] += cache[local_tid + i];
        __syncthreads();
    }

    if (local_tid == 0) 
        atomicAdd(result, cache[0]);
    
}

__global__ void normalize_vector(float* data, float norm, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size)
        data[tid] /= norm;
}

void normalize(float* data, int size){
    float* d_norm;
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemset(d_norm, 0, sizeof(float));

    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_squared_norm<<<blocks, BLOCK_SIZE>>>(data, d_norm, size);
    cudaDeviceSynchronize();

    float norm;
    cudaMemcpy(&norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
    norm = sqrtf(norm);

    normalize_vector<<<blocks, BLOCK_SIZE>>>(data, norm, size);
    cudaDeviceSynchronize();

    cudaFree(d_norm);
}


void matvec_mul(cublasHandle_t handle, const float* matrix, const float* vec, float* result, int rows, int cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, matrix, rows, vec, 1, &beta, result, 1);
}

float power_inter(const float* d_matrix, int rows, int cols, int num_iter){
    float* d_u;
    float* d_v;
    cudaMalloc(&d_v, rows * sizeof(float));
    cudaMalloc(&d_u, cols * sizeof(float));

    vector<float> h_v(cols, 1.0f);
    cudaMemcpy(d_v, h_v.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    for(int i = 0; i < num_iter; ++i){
        matvec_mul(handle, d_matrix, d_v, d_u, rows, cols);
        normalize(d_u, cols);

        matvec_mul(handle, d_matrix, d_u, d_v, cols, rows);
        normalize(d_v, rows);
    }

    float *d_Wv;
    cudaMalloc(&d_Wv, rows * sizeof(float));
    matvec_mul(handle, d_matrix, d_v, d_Wv, rows, cols);

    float sigma;
    cublasSnrm2(handle, rows, d_Wv, 1, &sigma);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_Wv);
    cublasDestroy(handle);
    return sigma;
}

int main(){
    const int rows = 4;
    const int cols = 4;
    const int num_iter = 100;

    vector<float> matirx = {
        0.5f, 0.2, 0.1f,
        0.2f, 0.5f, 0.3f,
        0.1f, 0.8f, 0.6f,
        0.9f, 0.4f, 0.7f
    };

    float* d_matrix;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMemcpy(d_matrix, matirx.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    float sigma = power_inter(d_matrix, rows, cols, num_iter);
    cout << "Spectral Norm: " << sigma << "\n";

    cudaFree(d_matrix);
    return 0;
}