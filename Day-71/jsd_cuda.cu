#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>

const int MAX_FUSED_SIZE = 65536;

int next_power_of_2(int n){
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

__global__ void jsd_kernel(
    const float* X, int x_stride,
    const float* Y, int y_stride,
    float* loss, int loss_stride,
    float* dx, int dx_stride,
    const int* labels,
    float beta,
    int n_non_ignore,
    int ingore_idx,
    int n_cols,
    bool has_label
) {
    int row = blockIdx.x;
    const float* X_row = X + row * x_stride;
    const float* Y_row = Y + row * y_stride;
    float* loss_row = loss + row * loss_stride;
    float* dx_row = dx + row * dx_stride;

    if (has_label){
        int label = labels[row];
        if (label == ingore_idx){
            for(int col = threadIdx.x; col < n_cols; col += blockDim.x){
                dx_row[col] = 0.0f;
            }
            return;
        }
    }

    for(int i = 0; i < n_cols; i+= blockDim.x){
        int idx = i + threadIdx.x;
        float x_val = (idx < n_cols) ? X_row[idx] : -INFINITY;
        float y_val = (idx < n_cols) ? Y_row[idx] : -INFINITY;

        float max_val = -INFINITY;
        extern __shared__ float sdata[];
        if (beta == 0.0f){
            sdata[threadIdx.x] = y_val;
        } else if (beta == 1.0f){
            sdata[threadIdx.x] = x_val;
        } else {
            sdata[threadIdx.x] = fmaxf(x_val, y_val);
        }
        __syncthreads();

        for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
            if (threadIdx.x < offset){
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + offset]);
            }
            __syncthreads();
        }
        max_val = sdata[0];

        float l = 0.0f;
        float dx = 0.0f;
        if (beta == 0.0f){
            float y_shifted = y_val - max_val;
            float y_prob = expf(y_shifted) * expf(max_val);
            l = y_prob * (y_val - x_val);
            dx = -y_prob;
        } else if (beta == 1.0f){
            float x_shifted = x_val - max_val;
            float x_prob = expf(x_shifted) * expf(max_val);
            l = x_prob * (x_val - y_val);
            dx = l + x_prob;
        } else {
            float x_shifted = x_val - max_val;
            float y_shifted = y_val - max_val;
            float exp_max = expf(max_val);
            float Q = expf(x_shifted) * exp_max;
            float P = expf(y_shifted) * exp_max;
            float beta_p = beta * P;
            float one_minus_beta_Q = (1.0f - beta) * Q;
            float M = beta_p + one_minus_beta_Q;
            float log_m = logf(M);
            l = beta_p * y_val + one_minus_beta_Q * x_val - M * log_m;
            dx = one_minus_beta_Q * (x_val - log_m);
        }

        float scale = 1.0f / n_non_ignore;
        l *= scale;
        dx *= scale;

        if (idx < n_cols){
            loss_row[idx] = l;
            dx_row[idx] = dx;
        }

        __syncthreads();
    }
}

void jsd_forward(
    const float* d_in,
    const float* d_tgt,
    const int* d_shift_labels,
    float beta,
    int ignore_idx,
    bool has_label,
    int BT,
    int V,
    float* d_loss,
    float* d_dx,
    int n_non_ignore
) {
    int blockSize = next_power_of_2(V);
    if (blockSize > MAX_FUSED_SIZE){
        blockSize = MAX_FUSED_SIZE;
    }
    size_t smem_bytes = blockSize * sizeof(float);
    dim3 grid(BT);
    dim3 block(blockSize);

    jsd_kernel<<<grid, block, smem_bytes>>>(d_in, V, d_tgt, V, d_loss, V, d_dx, V, d_shift_labels, beta, n_non_ignore, ignore_idx, V, has_label);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("jsd_kernel_failed %s\n", cudaGetErrorString(err));
}

__global__ void jsd_backward_kernel(const float* dx_in, float* dx_out, float grad_out, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dx_out[idx] = (grad_out == 1.0f) ? dx_in[idx] : grad_out * dx_in[idx];
}

void jsd_backward(
    const float* d_dx,
    float* d_dx_out,
    float grad_out,
    int total_ele
){
    int threads = 256;
    int blocks = (total_ele + threads - 1) / threads;
    jsd_backward_kernel<<<blocks, threads>>>(d_dx, d_dx_out, grad_out, total_ele);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("just_kernel_failed_backward %s\n", cudaGetErrorString(err));
}

__global__ void reduce_sum_kernel(const float* d_in, float *d_out, int N){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if (idx < N){
        sum = d_in[idx];
        if (idx + blockDim.x < N)
            sum += d_in[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

float reduce_loss(float* d_in, int N){
    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    int current_N = N;
    float sum = 0.0f;
    float* d_crr = d_in;
    while (true){
        size_t smem_size = threads * sizeof(float);
        reduce_sum_kernel<<<blocks, threads, smem_size>>>(d_crr, d_partial, current_N);
        cudaDeviceSynchronize();
        
        if (blocks == 1){
            cudaMemcpy(&sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        current_N = blocks;
        blocks = (current_N + threads * 2 - 1) / (threads * 2);
        d_crr = d_partial;
    }
    cudaFree(d_partial);
    return sum;
}

int main() {
    const int BT = 2;        
    const int V = 8;         
    const int ignore_idx = -1;
    const int total_ele = BT * V;
    const float beta = 0.5f;
    const bool has_label = true;
    const int n_non_ignore = BT;

    size_t data_size = total_ele * sizeof(float);
    size_t label_size = BT * sizeof(int);

    float h_in[total_ele], h_tgt[total_ele];
    int h_labels[BT];

    for (int i = 0; i < total_ele; ++i) {
        h_in[i] = static_cast<float>(rand() % 10) / 10.0f;
        h_tgt[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < BT; ++i) {
        h_labels[i] = rand() % V;
    }

    float *d_in, *d_tgt, *d_loss, *d_dx, *d_dx_out;
    int *d_labels;
    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_tgt, data_size);
    cudaMalloc(&d_loss, data_size);
    cudaMalloc(&d_dx, data_size);
    cudaMalloc(&d_dx_out, data_size);
    cudaMalloc(&d_labels, label_size);

    cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, h_tgt, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, label_size, cudaMemcpyHostToDevice);

    jsd_forward(d_in, d_tgt, d_labels, beta, ignore_idx, has_label, BT, V, d_loss, d_dx, n_non_ignore);

    float total_loss = reduce_loss(d_loss, total_ele);
    printf("Total JSD Loss: %f\n", total_loss);

    float grad_out = 1.0f;
    jsd_backward(d_dx, d_dx_out, grad_out, total_ele);

    cudaFree(d_in);
    cudaFree(d_tgt);
    cudaFree(d_loss);
    cudaFree(d_dx);
    cudaFree(d_dx_out);
    cudaFree(d_labels);

    return 0;
}


