#include <cuda_runtime.h>
#include <float.h>
#include <stddef.h>
#include <stdio.h>

__global__ void max_pooling_2d(const float* input, float* output, int input_width, int input_height, int pool_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int output_width = (input_width - pool_size) / stride + 1;
    int output_height = (input_height - pool_size) / stride + 1;

    if (out_x < output_width && out_y < output_height) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int in_x = out_x * stride + j;
                int in_y = out_y * stride + i;
                max_val = fmaxf(max_val, input[in_y * input_width + in_x]);
            }
        }
        output[out_y * output_width + out_x] = max_val;
    }
}

int main(){
    const int input_width = 8;
    const int input_height = 8;
    const int pool_size = 2;
    const int stride = 2;

    float input[input_height][input_width];

    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            input[i][j] = i * input_width + j + 1;
        }
    }

    int out_width = (input_width - pool_size) / stride + 1;
    int out_height = (input_height - pool_size) / stride + 1;

    float output[out_height][out_width];

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_output, sizeof(output));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((input_width + blockDim.x - 1) / blockDim.x, (input_height + blockDim.y - 1) / blockDim.y);
    max_pooling_2d<<<gridDim, blockDim>>>(d_input, d_output, input_width, input_height, pool_size, stride);

    cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Output:\n");
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            printf("%f ", output[i][j]);
        }
        printf("\n");
    }

    return 0;
}