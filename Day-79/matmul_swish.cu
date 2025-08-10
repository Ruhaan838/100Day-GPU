#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__global__ void compute_kernel(const float* input_mat, const float* weight_mat, const float* bias, float scaling_factor,
                                float* output, size_t batch_size, size_t in_features, size_t out_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y; // FIXED bug: was threadIdx.x

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input_mat[row * in_features + k] * weight_mat[col * in_features + k];
        }
        sum += bias[col];
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sigmoid;
        output[row * out_features + col] = scaling_factor * swish;
    }
}

int main() {
    const size_t batch_size = 2;
    const size_t in_features = 10;
    const size_t out_features = 4;
    const float scaling_factor = 1.5f;

    size_t input_size = batch_size * in_features;
    size_t weight_size = out_features * in_features;
    size_t bias_size = out_features;
    size_t output_size = batch_size * out_features;

    float *h_input  = new float[input_size];
    float *h_weight = new float[weight_size];
    float *h_bias   = new float[bias_size];
    float *h_output = new float[output_size];

    auto randf = [](){ return static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f; }; 
    for (size_t i = 0; i < input_size;  i++) h_input[i]  = randf();
    for (size_t i = 0; i < weight_size; i++) h_weight[i] = randf();
    for (size_t i = 0; i < bias_size;   i++) h_bias[i]   = randf();

    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input,  input_size * sizeof(float));
    cudaMalloc(&d_weight, weight_size * sizeof(float));
    cudaMalloc(&d_bias,   bias_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input,  h_input,  input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,   h_bias,   bias_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x,
                 (out_features + blockDim.y - 1) / blockDim.y);
    compute_kernel<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, scaling_factor,
                                          d_output, batch_size, in_features, out_features);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            std::cout << h_output[i * out_features + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}
