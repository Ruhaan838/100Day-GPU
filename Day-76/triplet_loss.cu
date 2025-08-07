#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__device__ float dist(const float* anchor, const float* val, int idx){
    return (anchor[idx] - val[idx]) * (anchor[idx] * val[idx]);
}

__global__ void triplet_loss_kernel(const float* anchor, const float* positive, const float* nagative, float* loss, float alpha, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    float ap_dist = dist(anchor, positive, idx);
    float an_dist = dist(anchor, nagative, idx);

    float t_ls = fmaxf(0.0f, ap_dist - an_dist + alpha);
    atomicAdd(loss, t_ls);
}

void triplet_forward(const float* anchor, const float* positive, float* negative, float* loss, float alpha, int dim){
    float *d_anchor, *d_positive, *d_negative, *d_loss;

    size_t size = dim * sizeof(float);
    cudaMalloc(&d_anchor, size);
    cudaMalloc(&d_positive, size);
    cudaMalloc(&d_negative, size);
    cudaMalloc(&d_loss, size);

    cudaMemcpy(d_anchor, anchor, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, size, cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    triplet_loss_kernel<<<blocks, threads>>>(d_anchor, d_positive, d_negative, d_loss, alpha, dim);

    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);
}

int main(){
    const int dim = 128;
    float anchor[dim], positive[dim], negative[dim], loss, alpha = 0.2f;

    for(int i = 0; i < dim; i++){
        anchor[i] = static_cast<float>(rand()) / RAND_MAX;
        positive[i] = static_cast<float>(rand()) / RAND_MAX;
        negative[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    triplet_forward(anchor, positive, negative, &loss, alpha, dim);

    printf("Triplet Loss: %f", loss);
    return 0;
}