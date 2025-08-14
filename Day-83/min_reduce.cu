#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>

__global__ void minReduceKernel(const float* input, float* output, size_t* shape, size_t ndim, size_t dim, 
    size_t out_left_dim, size_t out_right_dim, size_t dim_size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = out_left_dim * out_right_dim;

    if (idx >= out_size) return;

    size_t out_left_idx = idx / out_right_dim;
    size_t out_right_idx = idx % out_right_dim;

    float min_val = INFINITY;
    for(int i = 0; i < dim_size; i++){
        size_t input_idx = (out_left_idx * dim_size + i) * out_right_dim + out_right_idx;
        min_val = min(min_val, input[input_idx]);
    }
    output[idx] = min_val;
}

int main() {
    const size_t dim0 = 2;
    const size_t dim1 = 3;
    const size_t dim2 = 4;
    const size_t ndim = 3;
    size_t shape[ndim] = {dim0, dim1, dim2};

    const size_t input_size = dim0 * dim1 * dim2;
    const size_t output_size = dim0 * dim2;

    float* input = new float[input_size];
    float* output = new float[output_size];

    srand(time(0));
    for (size_t i = 0; i < input_size; i++)
        input[i] = static_cast<float>(rand() % 100);

    float* d_input;
    float* d_output;
    size_t* d_shape;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_shape, ndim * sizeof(size_t));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    size_t out_left_dim = dim0;
    size_t dim_size = dim1;
    size_t out_right_dim = dim2;

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    minReduceKernel<<<blocks, threads>>>(d_input, d_output, d_shape, ndim, 1, out_left_dim, out_right_dim, dim_size);
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input size: %d\n", static_cast<int>(input_size));
    printf("Output size: %d\n", static_cast<int>(output_size));
    for (size_t i = 0; i < output_size; i++)
        printf("%f ", output[i]);
    printf("\n");

    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_shape);

    return 0;
}
