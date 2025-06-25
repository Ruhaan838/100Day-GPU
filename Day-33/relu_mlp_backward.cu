#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

const int BLOCK_SIZE = 16;
using namespace std;

void check_cuda(cudaError_t err){
    if (err != cudaSuccess){
        cerr << "Cuda error: at" << __LINE__ << "In " << __FILE__ << " " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE); 
    }
}

__global__ void matmul_forward(float *out, const float* X, float* W, int N , int in_dim, int out_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= out_dim) return;
    float val = 0.0f;
    for (int i = 0; i < in_dim; ++i){
        val += X[row * in_dim + i] * W[i * out_dim + col];
    }
    out[row * out_dim + col] = val;
}

__global__ void add_bias_relu(float* out, const float* b, int N, int D){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    int col = idx % D;
    int row = idx / D;
    float z = out[idx] + b[col];
    out[idx] = fmaxf(0.0f, z);
}

__global__ void relu_backward(float* dReLU, const float* out, const float* dout, int N, int D){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    dReLU[idx] = (out[idx] > 0) ? dout[idx] : 0.0f;
}

// dW = x.T @ dReLu
__global__ void compute_dW(float* dW, const float* x, const float* dReLU, int N, int in_dim, int out_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= in_dim || col >= out_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += x[i * in_dim + row] * dReLU[i * out_dim + col];
    dW[row * out_dim + col] = sum;
}

// db = sum(dReLU, axis=0)
__global__ void compute_db(float* db, const float* dReLU, int N, int D){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += dReLU[i * D + col];
    db[col] = sum;
}

// dx = dReLU @ W.T
__global__ void compute_dx(float* dx, const float* dReLU, const float* W, int N, int in_dim, int out_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= in_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < out_dim; ++i)
        sum += dReLU[row * out_dim + i] * W[col * out_dim + i];
    dx[row * in_dim + col] = sum;
}

void init_mat(float* data, int size, float scale = 1.0f){
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX * scale;
}

void quick_check_sync(){
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
}

void print_mat(const float* data, int rows, int cols){
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            cout << data[i * cols + j] << " ";
        }
        cout << '\n';
    }
}

int main(){
    int N = 64;
    int in_dim = 128;
    int out_dim = 16;
    size_t x_size = N * in_dim * sizeof(float);
    size_t w_size = in_dim * out_dim * sizeof(float);
    size_t b_size = out_dim * sizeof(float);
    size_t out_size = N * out_dim * sizeof(float);

    float *X = (float*)malloc(x_size);
    float *W = (float*)malloc(w_size);
    float *b = (float*)malloc(b_size);
    float *out = (float*)malloc(out_size);

    init_mat(X, N * in_dim);
    init_mat(W, in_dim * out_dim);
    init_mat(b, out_dim);
    init_mat(out, N * out_dim, 0.0f);

    float *dx, *dw, *db, *dReLU, *d_out;
    float *dd_x, *dd_w, *dd_b, *dd_ReLU, *dd_out;

    check_cuda(cudaMalloc(&dx, x_size));
    check_cuda(cudaMalloc(&dw, w_size));
    check_cuda(cudaMalloc(&db, b_size));
    check_cuda(cudaMalloc(&dReLU, out_size));
    check_cuda(cudaMalloc(&d_out, out_size));

    check_cuda(cudaMalloc(&dd_x, x_size));
    check_cuda(cudaMalloc(&dd_w, w_size));
    check_cuda(cudaMalloc(&dd_b, b_size));
    check_cuda(cudaMalloc(&dd_ReLU, out_size));
    check_cuda(cudaMalloc(&dd_out, out_size));

    check_cuda(cudaMemcpy(dd_x, X, x_size, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dd_w, W, w_size, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dd_b, b, b_size, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dd_out, out, out_size, cudaMemcpyHostToDevice));

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_forward<<<grid_size, block_size>>>(dd_out, dd_x, dd_w, N, in_dim, out_dim);
    quick_check_sync();
    add_bias_relu<<<(N * out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dd_out, dd_b, N, out_dim);
    quick_check_sync();
    relu_backward<<<(N * out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dd_ReLU, dd_out, dd_out, N, out_dim);
    quick_check_sync();

    compute_dW<<<grid_size, block_size>>>(dd_w, dd_x, dd_ReLU, N, in_dim, out_dim);
    quick_check_sync();
    compute_db<<<(out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dd_b, dd_ReLU, N, out_dim);
    quick_check_sync();
    compute_dx<<<grid_size, block_size>>>(dd_x, dd_ReLU, dd_w, N, in_dim, out_dim);
    quick_check_sync();

    check_cuda(cudaMemcpy(W, dd_w, w_size, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(b, dd_b, b_size, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(X, dd_x, x_size, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(out, dd_out, out_size, cudaMemcpyDeviceToHost));

    cout << "W:\n";
    print_mat(W, in_dim, out_dim);
    cout << "b:\n";
    print_mat(b, 1, out_dim);
    cout << "X:\n";
    print_mat(X, N, in_dim);
    cout << "out:\n";
    print_mat(out, N, out_dim);

    free(X);
    free(W);
    free(b);
    free(out);
    check_cuda(cudaFree(dx));
    check_cuda(cudaFree(dw));
    check_cuda(cudaFree(db));
    check_cuda(cudaFree(dReLU));
    check_cuda(cudaFree(d_out));
    check_cuda(cudaFree(dd_x));
    check_cuda(cudaFree(dd_w));
    check_cuda(cudaFree(dd_b));
    check_cuda(cudaFree(dd_ReLU));
    check_cuda(cudaFree(dd_out));
}