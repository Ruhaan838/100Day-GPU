#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define ETA 0.5f

#define EUCLIDEAN 0 // x = x_old - n * grad (general use case in ML)
#define NEGATIVE_ENTROPY 1 // x = x_old * exp(-n * grad) (probablity vector optimization, like softmax policies in RL)
#define LOG_BARRIER 2 // x = x_old / (1 + n * grad) (minimizing a function over x > 0)

__global__ void mirro_desent(float *x, float *grad, float eta, int mirror_method, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float new_x = x[idx];

    switch(mirror_method){
        case EUCLIDEAN:
            new_x = x[idx] - eta * grad[idx];
            break;

        case NEGATIVE_ENTROPY:
            new_x = x[idx] * expf(-eta * grad[idx]);
            break;
        
        case LOG_BARRIER:
            new_x = x[idx] / (1.0f + eta * grad[idx]);
            break;
        
        default:
            new_x = x[idx];
    }

    x[idx] = new_x;
}

void checkCuda(cudaError_t result, const char *work_msg){
    if(result != cudaSuccess){
        fprintf(stderr, "CUDA Error: %s (%s)\n", work_msg, cudaGetErrorString(result));
        exit(-1);
    }
}

int main() {
    float *x, *grad, *dx, *dgrad;
    int mirror_method = LOG_BARRIER;

    x = (float*)malloc(N * sizeof(float));
    grad = (float*)malloc(N * sizeof(float));

    checkCuda(cudaMalloc(&dx, N * sizeof(float)), "Allocation dx");
    checkCuda(cudaMalloc(&dgrad, N * sizeof(float)), "Allocation dgrad");

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        grad[i] = 0.5f * i;
    }

    checkCuda(cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy from x to dx");
    checkCuda(cudaMemcpy(dgrad, grad, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy from grad to dgrad");

    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1)/block_size.x);

    mirro_desent<<<grid_size, block_size>>>(dx, dgrad, ETA, mirror_method, N);
    checkCuda(cudaDeviceSynchronize(), "Kernel execution");

    checkCuda(cudaMemcpy(x, dx, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy from dx to x");

    for (int i = 0; i < 10; i++)
        printf("x[%d] = %f\n", i, x[i]);

    free(x);
    free(grad);

    cudaFree(dx);
    cudaFree(dgrad);


    return 0;
}