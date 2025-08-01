#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>

enum Reduction{
    NONE = 0,
    SUM = 1,
    MEAN = 2,
    BATCH_MEAN = 3
};

void CUDA_CHECK(cudaError_t call){
    if (call != cudaSuccess){
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(call));
        exit(EXIT_FAILURE);
    }
}

__global__ void tv_dist_kernel(
    const float* p,
    int p_stride,
    const float* q,
    int q_stride,
    float* loss,
    int loss_stride,
    float* grads,
    int grad_stride,
    const int* labels,
    int ignore_idx,
    int n_cols,
    Reduction reduction,
    bool hasLabel
){
    int row = blockIdx.x;
    const float* p_row = p + row * p_stride;
    const float* q_row = q + row * q_stride;
    float* loss_row = loss + row * loss_stride;
    float* grads_row = grads + row * grad_stride;

    if(hasLabel){
        int label = labels[row];
        if (label == ignore_idx){
            for(int col = threadIdx.x; col < n_cols; col += blockDim.x){
                grads_row[col] = 0.0f;
                if (reduction == NONE){
                    loss_row[col] = 0.0f;
                }
            }
            return;
        }
    }

    float threads_sum = 0.0f;
    for(int col = threadIdx.x; col < n_cols; col += blockDim.x){
        float p_val = p_row[col];
        float q_val = q_row[col];
        float tv_loss = 0.5f * fabsf(p_val - q_val);
        float grad = (p_val > q_val) ? 0.5f : -0.5f;
        grads_row[col] = grad;

        if(reduction == NONE){
            loss_row[col] = tv_loss;
        } else{
            threads_sum += tv_loss;
        }
    }

    if (reduction != NONE){
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        sdata[tid] = threads_sum;
        __syncthreads();

        for(int s = blockDim.x / 2; s > 0; s >>= 1){
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        } if (tid == 0){
            loss_row[0] = sdata[0];
        }
    }
}


int main() {
    const int rows = 4;
    const int cols = 8;
    const int size = rows * cols;

    float* p = new float[size];
    float* q = new float[size];
    int* labels = new int[rows];
    float* loss = new float[size];
    float* grads = new float[size];

    for (int i = 0; i < size; ++i) {
        p[i] = static_cast<float>(rand() % 100) / 100.0f;
        q[i] = static_cast<float>(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < rows; ++i) {
        labels[i] = rand() % 4; 
    }

    float *dp, *dq, *dloss, *dgrads;
    int* dlabels;

    CUDA_CHECK(cudaMalloc(&dp, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dq, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dloss, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dgrads, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dlabels, rows * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dp, p, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, q, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dlabels, labels, rows * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shared_mem = threads * sizeof(float);

    tv_dist_kernel<<<rows, threads, shared_mem>>>(
        dp, cols, dq, cols, dloss, cols, dgrads, cols,
        dlabels, -1, cols, SUM, true
    );

    CUDA_CHECK(cudaMemcpy(loss, dloss, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grads, dgrads, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < rows; ++i) {
        printf("Row %d loss: %.4f\n", i, loss[i * cols]);
    }

    delete[] p;
    delete[] q;
    delete[] labels;
    delete[] loss;
    delete[] grads;

    cudaFree(dp);
    cudaFree(dq);
    cudaFree(dloss);
    cudaFree(dgrads);
    cudaFree(dlabels);

    return 0;
}
