#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void ddpm_update(float* x, float *eps, float *out, float alpha, float beta, float alpha_bar, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        float inv_sqrt_alpha = 1.0f / sqrtf(alpha);
        float scale_eps = beta / sqrtf(1.0f - alpha_bar);
        out[idx] = inv_sqrt_alpha * (x[idx] - scale_eps * eps[idx]);
    }
}

int main(){
    int n = 1024 * 1024 * 3;
    float alpha = 0.9f, beta = 0.1f, alpha_bar = 0.5f;

    float *x = (float*)malloc(n * sizeof(float));
    float *eps = (float*)malloc(n * sizeof(float));
    float *out = (float*)malloc(n * sizeof(float));

    for(int i = 0; i < n; i++){
        x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        eps[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    float *dx, *deps, *dout;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&deps, n * sizeof(float));
    cudaMalloc(&dout, n * sizeof(float));

    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deps, eps, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;

    cudaDeviceSynchronize();
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < 10000; i++){
        ddpm_update<<<grid_size, block_size>>>(dx, deps, out, alpha, beta, alpha_bar, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Cuda Kernel Time: %f ms\n", ms / 1000.0);

    cudaMemcpy(out, dout, n * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(dx);
    cudaFree(deps);
    cudaFree(dout);

    
    return 0;
}