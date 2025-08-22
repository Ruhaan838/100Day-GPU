#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

const float EPS = 1e-5f;
const int BLOCK_SIZE = 256;
using namespace std;

__global__ void compute_rms(const float* x, float* rms, size_t B, size_t N){
    extern __shared__ float sdata[];
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    const float* row_ptr = x + row * N;
    float sum = 0.0f;
    for(size_t i = tid; i < N; i+= blockDim.x){
        float v = row_ptr[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        float mean_sq = sdata[0] / static_cast<float>(N);
        rms[row] = sqrtf(mean_sq + EPS);
    }
}

__global__ void normalize_rms(const float* x, float* y, const float* rms, size_t B, size_t N){
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    float r = rms[row];
    const float *row_in = x + row * N;
    float *row_out = y + row * N;

    for(size_t i = tid; i < N; i += blockDim.x){
        row_out[i] = row_in[i] / r;
    }
}

int main() {
    size_t B = 4;   
    size_t N = 16;  

    vector<float> h_x(B * N);
    vector<float> h_y(B * N, 0.0f);
    vector<float> h_rms(B, 0.0f);

    for (size_t i = 0; i < B * N; i++) {
        h_x[i] = static_cast<float>(rand() % 10 + 1);
    }

    float *d_x, *d_y, *d_rms;
    cudaMalloc(&d_x, B * N * sizeof(float));
    cudaMalloc(&d_y, B * N * sizeof(float));
    cudaMalloc(&d_rms, B * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), B * N * sizeof(float), cudaMemcpyHostToDevice);

    compute_rms<<<B, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_x, d_rms, B, N);
    normalize_rms<<<B, BLOCK_SIZE>>>(d_x, d_y, d_rms, B, N);

    cudaMemcpy(h_y.data(), d_y, B * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rms.data(), d_rms, B * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t row = 0; row < B; row++) {
        cout << "Row " << row << " RMS = " << h_rms[row] << "\n";
        cout << "Normalized values: ";
        for (size_t j = 0; j < N; j++) {
            cout << h_y[row * N + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_rms);

    return 0;
}
