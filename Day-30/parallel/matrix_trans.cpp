#include <hip/hip_runtime.h>
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

void HipError(const char* msg){
    hipError_t error = hipGetLastError();
    if (error != hipSuccess){
        std::cerr << msg << " - Getting HIP Error: " << hipGetErrorString(error) << std::endl;
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

    hipMalloc(&d_in, size);
    hipMalloc(&d_out, size);

    hipMemcpy(d_in, in, size, hipMemcpyHostToDevice);
    HipError("Failed to copy input data to device");

    dim3 blockSize(32, 32);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                  (h + blockSize.y - 1) / blockSize.y);

    hipLaunchKernelGGL(trans_mat, gridSize, blockSize, 0, 0, d_in, d_out, w, h);
    hipDeviceSynchronize();
    HipError("Kernel Execution Failed");

    hipMemcpy(out, d_out, size, hipMemcpyDeviceToHost);
    HipError("Failed to copy Output Data");

    bool success = true;
    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            if(in[i*h + j] != out[j*w + i]){
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Mat Transpose successfully Done!!" : "Mat Transpose has a Problem") << std::endl;

    hipFree(d_in);
    hipFree(d_out);
    free(in);
    free(out);

    return 0;
}
