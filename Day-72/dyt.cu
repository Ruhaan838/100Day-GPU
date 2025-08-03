#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

const int BLOCKS_SIZE = 256;
using namespace std;

__global__ void dyt_forward_kernel(const float* x, float* y, const float alpha, const float* gamma, const float* beta, int n_cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cols) return;
    float val = alpha * x[idx];
    y[idx] = gamma[idx] * tanhf(val) + beta[idx];
}

__global__ void dyt_backward_kernel(const float* x, const float* dy, float* dx, float* dalpha, float* dgamma, float* dbeta,
float alpha, const float* gamma, int n_cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cols) return;

    float tanh_ax = tanhf(alpha * x[idx]);
    float sech_2_ax = 1 - tanh_ax * tanh_ax;

    dx[idx] = dy[idx] * gamma[idx] * sech_2_ax * alpha;
    atomicAdd(dalpha, dy[idx] * gamma[idx] * sech_2_ax * x[idx]);
    atomicAdd(&dgamma[idx], dy[idx] * tanh_ax);
    atomicAdd(&dbeta[idx], dy[idx]);
}

void dyt_forward(float* x, float* y, float alpha, float* gamma, float* beta, int n_cols){
    float *dx, *dy, *d_gamma, *d_beta;
    size_t size = n_cols * sizeof(float);
    cudaMalloc(&dx, size);
    cudaMalloc(&dy, size);
    cudaMalloc(&d_gamma, size);
    cudaMalloc(&d_beta, size);

    cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, size, cudaMemcpyHostToDevice);

    dim3 blocks((n_cols + BLOCKS_SIZE - 1) / BLOCKS_SIZE);

    dyt_forward_kernel<<<blocks, BLOCKS_SIZE>>>(dx, dy, alpha, d_gamma, d_beta, n_cols);
    cudaMemcpy(y, dy, size, cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

void dyt_backward(float* x, float* dy, float* dx, float* dalpha, float* dgamma, float* dbeta,
                  float alpha, float* gamma, int n_cols){
    float *d_x, *d_dy, *d_dx, *d_dalpha, *d_gamma, *d_dgamma, *d_dbeta;
    float dalpha_host = 0;
    size_t size = n_cols * sizeof(float);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_dy, size);
    cudaMalloc(&d_dx, size);
    cudaMalloc(&d_dalpha, sizeof(float));
    cudaMalloc(&d_gamma, size);
    cudaMalloc(&d_dgamma, size);
    cudaMalloc(&d_dbeta, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dalpha, &dalpha_host, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, size, cudaMemcpyHostToDevice);

    dim3 blocks((n_cols + BLOCKS_SIZE - 1) / BLOCKS_SIZE);

    dyt_backward_kernel<<<blocks, BLOCKS_SIZE>>>(
        d_x, d_dy, d_dx, d_dalpha, d_dgamma, d_dbeta, alpha, d_gamma, n_cols);

    cudaMemcpy(dx, d_dx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dalpha, d_dalpha, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dgamma, d_dgamma, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dbeta, d_dbeta, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_dy);
    cudaFree(d_dx);
    cudaFree(d_dalpha);
    cudaFree(d_gamma);
    cudaFree(d_dgamma);
    cudaFree(d_dbeta);
}

int main(){
    const int n_cols = 8;
    float x[n_cols], gamma[n_cols], beta[n_cols], dy[n_cols], alpha = 0.8;

    for(int i = 0; i < n_cols; i++){
        x[i] = static_cast<float>((rand() % 4) - 2);
        gamma[i] = 1;
        beta[i] = 0;
        dy[i] = 1;
    }

    float y[n_cols], dx[n_cols], dalpha, dgamma[n_cols], dbeta[n_cols];
    dyt_forward(x, y, alpha, gamma, beta, n_cols);

    cout << "Forward output:";
    for(int i = 0; i < n_cols; i++){
        cout << y[i] << " ";
    }
    cout << '\n';

    dyt_backward(x, dy, dx, &dalpha, dgamma, dbeta, alpha, gamma, n_cols);

    cout << "Backward dx:";
    for(int i = 0; i < n_cols; ++i){
        cout << dx[i] << " ";
    }
    cout << "\nBackward dalpha: " << dalpha << '\n';
    return 0;
}