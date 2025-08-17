#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void mseKernel(const float* preds, const float* tgt, size_t num_ele, float* sum){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_ele) return;
    float diff = preds[idx] - tgt[idx];
    float sq_diff = diff * diff;

    atomicAdd(sum, sq_diff);
}

int main(){
    const size_t num_ele = 10;
    float h_preds[num_ele] = {1, 1, 0, 1, 0, 1, 0, 1, 1, 0};
    float h_tgt[num_ele] = {1, 1, 1, 1, 0, 0, 0, 1, 0, 1};

    

    float *d_preds, *d_tgt, *d_sum;
    cudaMalloc(&d_preds, num_ele * sizeof(float));
    cudaMalloc(&d_tgt,   num_ele * sizeof(float));
    cudaMalloc(&d_sum,   sizeof(float));

    cudaMemcpy(d_preds, h_preds, num_ele * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt,   h_tgt,   num_ele * sizeof(float), cudaMemcpyHostToDevice);

    float h_sum = 0.0f;
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (num_ele + threadsPerBlock - 1) / threadsPerBlock;
    mseKernel<<<blocks, threadsPerBlock>>>(d_preds, d_tgt, num_ele, d_sum);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    float mse = h_sum / static_cast<float>(num_ele);
    printf("MSE = %f\n", mse);

    cudaFree(d_preds);
    cudaFree(d_tgt);
    cudaFree(d_sum);

    return 0;
}
