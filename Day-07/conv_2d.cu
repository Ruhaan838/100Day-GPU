#include <iostream>
#include <cuda_runtime.h>

#define Wigths_size 5
#define shared_size (32 + Wigths_size - 1)

__constant__ float wigths[Wigths_size][Wigths_size];


__global__ void conv_2d(const float *in, float *out, int n){
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int x = blockDim.x * blockIdx.x + thread_x;
    int y = blockDim.y * blockIdx.y + thread_y;

    __shared__ float shared_mem[shared_size][shared_size];

    if((x < n) && (y < n))
        shared_mem[thread_x + Wigths_size / 2][thread_y + Wigths_size / 2] = in[x * n + y];
    
    if (thread_x < Wigths_size / 2){
        int left_idx = blockIdx.x * blockDim.x - (Wigths_size / 2) + thread_x;
        if (left_idx >= 0 && y < n)
            shared_mem[thread_x][thread_y + Wigths_size / 2] = in[left_idx * n + y];

        else
            shared_mem[thread_x][thread_y + Wigths_size / 2] = 0.0f;

    }

    if (thread_x < Wigths_size / 2){
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + thread_x;
        if (right_idx < n && y < n)
            shared_mem[thread_x + blockDim.x + Wigths_size / 2][thread_y + Wigths_size / 2] = in[right_idx * n + y];
        
        else
            shared_mem[thread_x + blockDim.x + Wigths_size / 2][thread_y + Wigths_size / 2] = 0.0f;
    }

    if (thread_y < Wigths_size / 2){
        int top_y = y - (Wigths_size / 2) + thread_y;
        if(top_y >= 0 && x < n)
            shared_mem[thread_x + Wigths_size / 2][thread_y] = in[x * n + top_y];
        else
            shared_mem[thread_x + Wigths_size / 2][thread_y] = 0.0f;
    }

    if (thread_y < Wigths_size / 2){
        int bottom_y = y + (Wigths_size / 2) + thread_y;
        if(bottom_y < n && x < n)
            shared_mem[thread_x + Wigths_size / 2][thread_y + blockDim.y + Wigths_size / 2] = in[x * n + bottom_y];
        else
            shared_mem[thread_x + Wigths_size / 2][thread_y + blockDim.y + Wigths_size / 2] = 0.0f;
    }

    __syncthreads();

    if ((x < n) && (y < n)){
        float result = 0.0f;
        for (int k = 0; k < Wigths_size; k++){
            for (int i = 0; i < Wigths_size; i++){
                result += shared_mem[thread_x + k][thread_y + i] * wigths[k][i];
            }
        }
        out[x*n + y] = result;
    }
}

void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void print_data(float *mat, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%.2f ", mat[i*n + j]);
        }
        printf("\n");
    }
}

int main(){
    int n = 10;
    int size = n * n * sizeof(float);
    float *in = (float*)malloc(size);
    float *out = (float*)malloc(size);
    
    float w[Wigths_size][Wigths_size];

    for(int i = 0; i < Wigths_size; i++){
        for(int j = 0; j < Wigths_size; j++){
            w[i][j] = 5;
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            in[i*n + j] = 3;
        }
    }

    float *din, *dout;
    cudaMalloc(&din, size);
    cudaMalloc(&dout, size);

    cudaMemcpy(din, in, size , cudaMemcpyHostToDevice);
    CudaError("Failed to copy the input data");

    cudaMemcpyToSymbol(wigths, w, Wigths_size * Wigths_size * sizeof(float));
    CudaError("Failed to copy the mask data");

    dim3 dimBlock(32, 32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);
    conv_2d<<<dimGrid, dimBlock>>>(din, dout, n);

    CudaError("Failed to execute the kernel");

    cudaDeviceSynchronize();
    cudaMemcpy(out, dout, size, cudaMemcpyDeviceToHost);
    CudaError("Failed to copy output data");

    printf("Results:\n");
    print_data(out, n);

    cudaFree(din);
    cudaFree(dout);
    free(in);
    free(out);

    return 0;

}