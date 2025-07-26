#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

enum Reduction {NONE = 0, SUM = 1, MEAN = 2, BATCHMEAN = 3};

__global__ void kldiv_forward_kernel_none(const float* __restrict__ y_pred, const float* __restrict__ y_true,
                                           float* __restrict__ loss, int V, float eps, bool log_target) {
    
    int b = blockIdx.x;
    int i = threadIdx.x;
    int offset = b * V + i;
    if (i < V) {
        float pred = y_pred[offset];
        float target = y_true[offset];
        float val = 0.0f;
        if(!log_target){
            val = target * (logf(fmaxf(target, eps)) - pred);
        } else {
            val = expf(target) * (target - pred);
        }
        loss[offset] = val;
    }
}

__global__ void kldiv_forward_kernel_reduce(const float* __restrict__ y_pred, const float* __restrict__ y_true,
                                           float* __restrict__ loss, int V, float eps, bool log_target, Reduction reduction) {
    
    int b = blockIdx.x;
    extern __shared__ float sdata[];
    float sum = 0.0f;

    for(int i = threadIdx.x; i < V; i += blockDim.x){
        int offset = b * V + i;
        float pred = y_pred[offset];
        float target = y_true[offset];
        float val = 0.0f;
        if(!log_target){
            val = target * (logf(fmaxf(target, eps)) - pred);
        } else {
            val = expf(target) * (target - pred);
        }
        sum += val;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        loss[b] = sdata[0];
    }
}

__global__ void kldiv_backward_kernel(const float* __restrict__ y_true, float* __restrict__ grad, int V, bool log_target){
    int b = blockIdx.x;
    for(int i = threadIdx.x; i < V; i += blockDim.x){
        int offset = b * V + i;
        float target = y_true[offset];
        float res = (!log_target) ? -target : expf(target);
        grad[offset] = res;
    }
}

__global__ void scale_kernel(float* data, int N, float factor){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i] *= factor;
}

void kldiv_forward(const float* y_pred, const float* y_true, float* loss, int B, int V, float eps, bool log_target, Reduction reduction){
    if (reduction == NONE){
        int threads = V;
        int grids = B;
        kldiv_forward_kernel_none<<<grids, threads>>>(y_pred, y_true, loss, V, eps, log_target);
    } else {
        int threads = 256;
        int grids = B;
        size_t shared_size = threads * sizeof(float);
        kldiv_forward_kernel_reduce<<<grids, threads, shared_size>>>(y_pred, y_true, loss, V, eps, log_target, reduction);
    }
    cudaDeviceSynchronize();
}

void kldiv_backward(const float* y_true, float* grad, int B, int V, bool log_target, float grad_output = 1.0f){
    int threads = 256;
    int grid = B;
    kldiv_backward_kernel<<<grid, threads>>>(y_true, grad, V, log_target);

    if(grad_output != 1.0f){
        int total = B * V;
        int blocksize = 256;
        int numBlocks = (total + blocksize - 1) / blocksize;
        scale_kernel<<<numBlocks, blocksize>>>(grad, total, grad_output);
        cudaDeviceSynchronize();
    }
}

int main(){
    const int B = 10;
    const int V = 1024;
    size_t data_size = B * V * sizeof(float);
    size_t loss_size = (B * ((BATCHMEAN == NONE) ? V : 1) * sizeof(float));

    float* y_pred = new float[B * V];
    float* y_true = new float[B * V];
    float* loss = new float[B];

    for(int i = 0; i < B * V; i++){
        y_pred[i] = 0.5f;
        y_true[i] = 0.3f;
    }

    float* d_y_pred, *d_y_true, *d_loss, *d_grad;
    cudaMalloc(&d_y_pred, data_size);
    cudaMalloc(&d_y_true, data_size);

    cudaMalloc(&d_loss, B * sizeof(float));
    cudaMalloc(&d_grad, data_size);

    cudaMemcpy(d_y_pred, y_pred, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, y_true, data_size, cudaMemcpyHostToDevice);

    Reduction reduction = BATCHMEAN;
    kldiv_forward(d_y_pred, d_y_true, d_loss, B, V, 1e-10f, false, reduction);

    cudaMemcpy(loss, d_loss, B * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Forward loss:\n");
    for(int b = 0; b < B; b++) printf("Batch %d: %f\n", b, loss[b]);

    kldiv_backward(d_y_true, d_grad, B, V, false, 1.0f);

    float* grad = new float[B * V];
    cudaMemcpy(grad, d_grad, data_size, cudaMemcpyDeviceToHost);
    printf("Grad values:\n");
    for(int i = 0; i < 10; i++) printf("%f ", grad[i]);
    printf("\n");



}