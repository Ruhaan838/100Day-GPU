#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int BLOCK_SIZE = 256;

__global__ void linear_kernel(float* input, float* weights, float* bias, float* output, int in_features, int out_features){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= out_features) return;
    float sum = bias[idx];
    for(int i = 0; i < in_features; i++){
        sum += input[i] * weights[i * out_features + idx];
    }
    output[idx] = sum;
}

__global__ void softmax_cross_entropy_kernel(float* logits, int* label, float* loss, int num_classes){
    extern __shared__ float exp_sums[];
    int idx = threadIdx.x;

    float max_val = -INFINITY;
    for(int i = 0; i < num_classes; i++){
        max_val = fmaxf(max_val, logits[i]);
    }
    float sum = 0.0f;
    for(int i = 0; i < num_classes; i++){
        exp_sums[i] = expf(logits[i] - max_val);
        sum += exp_sums[i];
    }
    float log_prob = logf(exp_sums[label[0]] / sum);
    *loss = -log_prob;
}

void run_fused_op(float* inputs, float* weights, float* bias, int* labels, int in_featues, int out_featues){
    float *d_input, *d_weights, *d_bias, *d_output, *d_loss;
    int *d_label;

    size_t in_size = in_featues * sizeof(float);
    size_t w_size = in_featues * out_featues * sizeof(float);
    size_t out_size = out_featues * sizeof(float);

    cudaMalloc((void**)&d_input, in_size);
    cudaMalloc((void**)&d_weights, w_size);
    cudaMalloc((void**)&d_bias, out_size);
    cudaMalloc((void**)&d_output, out_size);
    cudaMalloc((void**)&d_label, sizeof(int));
    cudaMalloc((void**)&d_loss, sizeof(float));

    cudaMemcpy(d_input, inputs, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, d_bias, out_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, labels, sizeof(int), cudaMemcpyHostToDevice);

    int lin_blocks = (out_featues + BLOCK_SIZE - 1) / BLOCK_SIZE;
    linear_kernel<<<lin_blocks, BLOCK_SIZE>>>(d_input, d_weights, d_bias, d_output, in_featues, out_featues);
    softmax_cross_entropy_kernel<<<1, out_featues, out_size>>>(d_output, d_label, d_loss, out_featues);

    float loss;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Cross Entropy Loss: %f", loss);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_label);
    cudaFree(d_loss);
}

int main() {
    int in_features = 3;
    int out_features = 2;
    float input[3] = {3.4, 5.3, 8.7};
    float weights[6] = {0.2, 0.6, -0.1, 0.0, 0.5, 1.0};
    float bias[2] = {0.1, -0.2};
    int label = 1;

    run_fused_op(input, weights, bias, &label, in_features, out_features);
    return 0;
}