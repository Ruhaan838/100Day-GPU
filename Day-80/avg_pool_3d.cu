#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

__global__ void avg_pool_3d(const float* input, float* output, int H, int W, int D,
                            int kernel_size, int stride, int padding,
                            int H_out, int W_out, int D_out) {
    int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_j = blockIdx.y * blockDim.y + threadIdx.y;
    int out_k = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_i >= H_out || out_j >= W_out || out_k >= D_out) return;

    int in_start_i = out_i * stride - padding;
    int in_start_j = out_j * stride - padding;
    int in_start_k = out_k * stride - padding;

    float sum = 0.0f;
    const int kernel_vol = kernel_size * kernel_size * kernel_size;

    #pragma unroll
    for (int m = 0; m < kernel_size; m++) {
        int crr_i = in_start_i + m;
        #pragma unroll
        for (int n = 0; n < kernel_size; n++) {
            int crr_j = in_start_j + n;
            #pragma unroll
            for (int o = 0; o < kernel_size; o++) {
                int crr_k = in_start_k + o;
                if (crr_i >= 0 && crr_i < H &&
                    crr_j >= 0 && crr_j < W &&
                    crr_k >= 0 && crr_k < D) {
                    int idx = crr_i * (W * D) + crr_j * D + crr_k;
                    sum += input[idx];
                }
            }
        }
    }
    int out_idx = out_i * (W_out * D_out) + out_j * D_out + out_k;
    output[out_idx] = sum / static_cast<float>(kernel_vol);
}

int main() {
    
    int H = 8, W = 8, D = 8;  
    int kernel_size = 2;
    int stride = 2;
    int padding = 0;

    
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    int D_out = (D + 2 * padding - kernel_size) / stride + 1;

    size_t input_size = H * W * D * sizeof(float);
    size_t output_size = H_out * W_out * D_out * sizeof(float);

    
    float* h_input = (float*)malloc(input_size);
    float* h_output = (float*)malloc(output_size);

    
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < H * W * D; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    
    dim3 block(4, 4, 4);
    dim3 grid((H_out + block.x - 1) / block.x,
              (W_out + block.y - 1) / block.y,
              (D_out + block.z - 1) / block.z);

    
    avg_pool_3d<<<grid, block>>>(d_input, d_output, H, W, D,
                                 kernel_size, stride, padding,
                                 H_out, W_out, D_out);

    cudaDeviceSynchronize();

    
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    
    cout << "3D Avg Pool Output:\n";
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            for (int k = 0; k < D_out; k++) {
                int idx = i * (W_out * D_out) + j * D_out + k;
                cout << h_output[idx] << " ";
            }
            cout << "\n";
        }
        cout << "----\n";
    }

    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
