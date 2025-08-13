#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

__global__ void consin_similarity_kernel(const float* preds, const float* targets, float* out, size_t n, size_t d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_tgt = 0.0f;
    size_t offset = idx * d;

    for(size_t j = 0; j < d; j++){
        float p = preds[offset + j];
        float t = targets[offset + j];
        dot += p * t;
        norm_pred += p * p;
        norm_tgt += t * t;
    }

    norm_pred = sqrtf(norm_pred);
    norm_tgt = sqrtf(norm_tgt);

    const float eps = 1e-8f;
    float denom = fmaxf(eps, norm_pred) * fmaxf(eps, norm_tgt);
    float cosin_sim = dot / denom;

    out[idx] = 1.0f - cosin_sim;
}

int main() {
    size_t n = 4, d = 5;
    size_t size = n * d * sizeof(float);
    size_t out_size = n * sizeof(float);

    float *preds = (float*)malloc(size);
    float *targets = (float*)malloc(size);
    float *out = (float*)malloc(out_size);

    srand(time(NULL));
    for (size_t i = 0; i < n * d; i++) {
        preds[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
        targets[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_preds, *d_targets, *d_out;
    cudaMalloc(&d_preds, size);
    cudaMalloc(&d_targets, size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_preds, preds, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    consin_similarity_kernel<<<blocks, threads>>>(d_preds, d_targets, d_out, n, d);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++) {
        printf("%d Loss: %f\n", i, out[i]);
    }

    cudaFree(d_preds);
    cudaFree(d_targets);
    cudaFree(d_out);
    free(preds);
    free(targets);
    free(out);
    return 0;
}