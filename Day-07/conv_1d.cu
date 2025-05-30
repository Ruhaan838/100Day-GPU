#include <iostream>
#include <cuda_runtime.h>

#define Wigths_size 5

__constant__ float wigths[Wigths_size];

__global__ void conv_1d(const float *in, float* out, int n){
    int threadidx = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadidx;

    if (i < n){
        float result = 0.0f;
        for (int k = -1 * Wigths_size / 2; k < Wigths_size / 2 + 1; k++){
            // printf("%.i", k);
            if (i + k >= 0 && i + k < n)
                result += in[i+k] * wigths[k+Wigths_size / 2];
            
        }
        out[i] = result;
    }
}

void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void print_data(float *vec, int n){
    for(int i = 0; i < n; i++)
        printf("%.2f ", vec[i]);
}

int main(){
    int n = 10;
    float in[n], out[n];
    float w[Wigths_size];
    for(int i = 0; i < Wigths_size; i++)
        w[i] = i;
    
    for(int i = 0; i < n; i++)
        in[i] = i;

    float *din, *dout;
    int size = n * sizeof(float);
    cudaMalloc(&din, size);
    cudaMalloc(&dout, size);

    cudaMemcpy(din, in, size, cudaMemcpyHostToDevice);
    CudaError("Failed to copy the input data");

    cudaMemcpyToSymbol(wigths, w, Wigths_size * sizeof(float));
    CudaError("Failed to copy the wigths data");

    dim3 dimBlock(32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    conv_1d<<<dimGrid, dimBlock>>>(din, dout, n);
    CudaError("Failed to execute the kernel");

    cudaDeviceSynchronize();
    cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);
    CudaError("Failed to copy the output data");

    cudaFree(din);
    cudaFree(dout);

    printf("In:\n");
    print_data(in, n);
    printf("\n");
    
    printf("Wigths:\n");
    print_data(w, Wigths_size);
    printf("\n");


    printf("Out:\n");
    print_data(out, n);
    printf("\n");

    
}
