#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

using namespace std;

template <typename T>
__global__ void compute_dl_dw(T* dldy, T* input_unrolled, T* dldw, int output_height, int output_wight, int num_filter, int filter_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < filter_size && col < num_filter){
        T sum = 0;
        for (int i = 0; i < output_height * output_wight; i++)
            sum += input_unrolled[i * filter_size + row] * dldy[i * num_filter + col];
        dldw[row * num_filter + col] = sum;
    }

}

template <typename T>
__global__ void computer_dl_dx(T* dldy, T* weights, T* dldx_unrolled, int output_height, int output_wights, int num_filter, int filter_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height * output_wights && col < filter_size){
        T sum = 0;
        for (int i = 0; i < num_filter; i++)
            sum += dldy[row * num_filter + i] * weights[col * num_filter + i];
        dldx_unrolled[row * filter_size + col] = sum;
    }
}

template <typename T>
__global__ void maxPoolBackwardKernel(T* dldy, T* input, T* dldx, int input_weight, int input_height, int pool_size, int stride){
    int output_height = (input_height - pool_size) / stride + 1;
    int output_weidth = (input_weight - pool_size) / stride + 1;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < output_height && col < output_weidth){
        T max_val = -INFINITY;
        int max_i = -1, max_j = -1;
        for (int i = 0; i < pool_size; i++){
            for(int j = 0; j < pool_size; j++){
                int input_row = row * stride + i;
                int input_col = col * stride + j;

                if (input_row < input_height && input_col < input_weight){
                    max_val = input[input_row * input_weight + input_col];
                    max_i = input_row;
                    max_j = input_col;
                }
            }
        }

        if (max_i != -1 && max_j != -1)
            atomicAdd(&dldx[max_i * input_weight + max_j], dldy[row * output_weidth + col]);

    }
}

__global__ void unrollKernel(const float* input, float* input_unrolled,
                            const int input_channels, const int input_height, const int input_width,
                            const int kernel_size, const int output_height, const int output_widht){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ele = output_widht * output_height;

    if (idx < total_ele){
        int out_y = idx / output_widht;
        int out_x = idx % output_widht;

        for (int c = 0; c < input_channels; c++){
            for (int ky = 0; ky < kernel_size; ky++){
                for (int kx = 0; kx < kernel_size; kx++){
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;

                    int unroll_idx = idx * (input_channels * kernel_size * kernel_size) + 
                    (c * kernel_size * kernel_size + ky * kernel_size + kx);

                    int input_idx = c * (input_height * input_width) + in_y * input_width + in_x;

                    input_unrolled[unroll_idx] = input[input_idx];
                }
            }
        }
    }
}


void unrollInput(int input_chennels, int input_height, int input_width,
                int kernel_size, float* input, float* input_unrolled){
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    int total_output_ele = output_height * output_width;

    dim3 threadsPerBlock(256);
    dim3 numBlocks(total_output_ele + threadsPerBlock.x - 1 / threadsPerBlock.x);
    
    unrollKernel<<<numBlocks, threadsPerBlock>>>(
        input,
        input_unrolled,
        input_chennels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("CUDA Error in unroll: %s\n", cudaGetErrorString(error));
    
    cudaDeviceSynchronize();
}

void ConvBackward(int batch_size, int num_filters, int input_channels,
                  int input_height, int input_width, int kernel_size, float* dldy,
                  float* input, float* weights, float* dldx, float* dldw){
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;

    float* input_unrolled;
    float* dldx_unrolled;
    size_t size = output_height * output_width * filter_size * sizeof(float);
    cudaMalloc(&input_unrolled, size);
    cudaMalloc(&dldx_unrolled, size);

    for (int n = 0; n < batch_size; n++){
        unrollInput(input_channels, input_height, input_width, kernel_size, input + n * input_channels * input_height * input_height, input_unrolled);

        dim3 blockSize(16, 16);
        dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x, (output_height + blockSize.y - 1) / blockSize.y);

        computer_dl_dx<<<gridSize, blockSize>>>(dldy, input_unrolled, dldw, output_height, output_width, num_filters, filter_size);
        compute_dl_dw<<<gridSize, blockSize>>>(dldy, weights, dldx_unrolled, output_height, output_width, num_filters, filter_size);

        cudaDeviceSynchronize();
    }

    cudaFree(input_unrolled);
    cudaFree(dldx_unrolled);
}

__global__ void maxPoolingKernel(float* input, float* output, int input_height, int input_width, int pool_size, int stride){
    int output_height = (input_height - pool_size) / stride + 1;
    int output_widht = (input_width - pool_size) / stride + 1;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.y;

    if (row < output_height && col < output_widht){
        float max_value = -INFINITY;
        for (int i = 0; i < pool_size; i++){
            for (int j = 0; j < pool_size; j++){
                int input_row = row * stride + i;
                int input_col = col * stride + j;
                max_value = fmaxf(max_value, input[input_row * input_width + input_col]);
            }
        }
        output[row * output_widht + col] = max_value;
    }
}

__global__ void MatMulKernel(float* input_unrolled, float* weights, float* output,
                             int output_height, int output_width, int num_filters, int filter_size){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_ele = output_height * output_width;

    if (idx < total_out_ele * num_filters){
        int output_idx = idx / num_filters;
        int filter_idx = idx % num_filters;

        float sum = 0.0f;

        for (int i = 0; i < filter_size; i++)
            sum += input_unrolled[output_idx * filter_size + i] * weights[i * num_filters + filter_idx];
        
        output[idx] = sum;
    }
}

__global__ void convKernel(const float* input_unrolled, const float* weights, float* output,
                           const int output_size, const int num_filters, const int filter_size){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size * num_filters){
        int output_idx = idx / num_filters;
        int filter_idx = idx % num_filters;

        float sum = 0.0f;
        for (int i = 0; i < filter_size; i++)
            sum += input_unrolled[output_idx * filter_size + i] * weights[i * num_filters + filter_idx];
        
        output[idx] = sum;
    }

}

void ConvForward(float* input, float* weights, float* output, 
                int batch_size, int num_filters, int input_channels,
                int input_height, int input_width, int kernel_size){
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int output_size = output_height * output_width;
    int filter_size = input_channels * kernel_size * kernel_size;

    float* input_unrolled;
    size_t unrolled_size = output_size * filter_size * sizeof(float);
    cudaMalloc(&input_unrolled, unrolled_size);

    int unroll_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int conv_blocks = (output_size * num_filters + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int n = 0; n < batch_size; n++){
        float* input_n = input + n * input_channels * input_height * input_width;
        float* output_n = output + n * num_filters * output_height * output_width;

        unrollKernel<<<unroll_blocks, BLOCK_SIZE>>>(
            input_n,
            input_unrolled,
            input_channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width
        );

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            printf("Unroll kernel error: %s\n", cudaGetErrorString(error));

        convKernel<<<conv_blocks, BLOCK_SIZE>>>(
            input_unrolled,
            weights,
            output_n,
            output_size,
            num_filters,
            filter_size
        );

        error = cudaGetLastError();
        if (error != cudaSuccess)
            printf("Conv Kernel Error: %s\n", cudaGetErrorString(error));
        
        cudaDeviceSynchronize();
    }

    cudaFree(input_unrolled);
}

int main() {
    const int batch_size = 1;
    const int input_channels = 1;
    const int input_width = 4;
    const int input_height = 4;
    const int kernel_size = 3;
    const int num_filter = 2;

    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;

    float input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float weight[] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
        0, 1, -1,
        0, 1, -1,
        0, 1, -1
    };

    float *d_input, *d_weight, *d_output;
    size_t input_size = batch_size * input_channels * input_height * input_width * sizeof(float);
    size_t weight_size = num_filter * input_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = batch_size * num_filter * output_height * output_width * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weight, weight_size);
    cudaMalloc(&d_output, output_size);
    cudaMemset(d_output, 0, output_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_size, cudaMemcpyHostToDevice);

    ConvForward(d_input, d_weight, d_output, batch_size, num_filter, input_channels, input_height, input_width, kernel_size);

    float* output = new float[output_size / sizeof(float)];
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cout << "Forward Output:\n";
    for (int f = 0; f < num_filter; f++){
        cout << "Filter " << f << ":\n";
        for (int i = 0; i < output_height; i++){
            for (int j = 0; j < output_width; j++){
                cout << output[f * output_height * output_width + i + output_width + j] << " ";
            }
            cout << '\n';
        }
        cout << '\n';
    }

    delete[] output;
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}