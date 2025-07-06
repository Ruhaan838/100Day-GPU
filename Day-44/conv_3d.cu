#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols,
                              int output_depth, int output_rows, int output_cols) {
    // Correct thread indices
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_col >= output_cols || out_row >= output_rows || out_depth >= output_depth) return;

    float sum = 0.0f;

    // Compute convolution sum
    for (int kd = 0; kd < kernel_depth; kd++) {
        for (int kr = 0; kr < kernel_rows; kr++) {
            for (int kc = 0; kc < kernel_cols; kc++) {
                int in_depth = out_depth + kd;
                int in_row = out_row + kr;
                int in_col = out_col + kc;

                int input_idx = in_depth * (input_rows * input_cols)
                              + in_row * input_cols
                              + in_col;

                int kernel_idx = kd * (kernel_rows * kernel_cols)
                               + kr * kernel_cols
                               + kc;

                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }

    int output_idx = out_depth * (output_rows * output_cols)
                   + out_row * output_cols
                   + out_col;

    output[output_idx] = sum;
}

int main() {

    int input_depth = 3;
    int input_rows = 64;
    int input_cols = 64;

    int kernel_depth = 1;
    int kernel_rows = 3;
    int kernel_cols = 3;

    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Allocate host memory
    size_t in_size = input_depth * input_rows * input_cols * sizeof(float);
    float* input = (float*)malloc(in_size);

    size_t kernel_size = kernel_depth * kernel_rows * kernel_cols * sizeof(float);
    float* kernel = (float*)malloc(kernel_size);

    size_t output_size = output_depth * output_rows * output_cols * sizeof(float);
    float* output = (float*)malloc(output_size);

    for (int i = 0; i < input_depth; i++) {
        for (int j = 0; j < input_rows; j++) {
            for (int k = 0; k < input_cols; k++) {
                input[i * (input_rows * input_cols) + j * input_cols + k] = 1.0f;
            }
        }
    }

    for (int i = 0; i < kernel_depth; i++) {
        for (int j = 0; j < kernel_rows; j++) {
            for (int k = 0; k < kernel_cols; k++) {
                kernel[i * (kernel_rows * kernel_cols) + j * kernel_cols + k] = 2.0f;
            }
        }
    }

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, in_size);
    cudaMalloc((void**)&d_kernel, kernel_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, input, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 block_dim(16, 8, 4);
    dim3 grid_dim((output_cols + block_dim.x - 1) / block_dim.x,
                  (output_rows + block_dim.y - 1) / block_dim.y,
                  (output_depth + block_dim.z - 1) / block_dim.z);

    conv3d_kernel<<<grid_dim, block_dim>>>(d_input, d_kernel, d_output,
                                           input_depth, input_rows, input_cols,
                                           kernel_depth, kernel_rows, kernel_cols,
                                           output_depth, output_rows, output_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    free(input);
    free(kernel);
    free(output);

    cudaDeviceReset(); 
    return 0;
}
