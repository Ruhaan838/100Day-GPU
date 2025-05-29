#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

__global__ void trans_mat(const float* in, float* out, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height){
        int in_idx = y * width + x;
        int out_idx = x * height + y;
        out[out_idx] = in[in_idx];
    }

}

void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); //NEW: cudaError_t and CudaGetLastError
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    int w = WIDTH;
    int h = HEIGHT;

    size_t size = w * h * sizeof(float);
    float* in = (float*)malloc(size);
    float* out = (float*)malloc(size);

    for(int i = 0; i < w * h; i++)
        in[i] = static_cast<float>(i);
    
    float* d_in;
    float* d_out;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    CudaError("Failed to copy input data to device");

    dim3 blockSize(32, 32);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    trans_mat<<<gridSize, blockSize>>>(d_in, d_out, w, h);
    cudaDeviceSynchronize();
    CudaError("Kernel Execution Failed");

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    CudaError("Failed to copy Output Data");

    bool success = true;
    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            if(in[i*h + j] != out[j*w+ i]){
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Mat Transpose successfuly Done!!" : "Mat Transpose Have a Problem") << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    free(in);
    free(out);

    return 0;
    
}