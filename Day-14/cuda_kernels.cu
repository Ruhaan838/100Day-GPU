#include "cuda_kernels.cuh"
#include "cuda_helper.cuh"

__global__ void addBiasKernel(float *output, const float* bias, int batch_size, int output_features){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / output_features;
    int features = idx % output_features;
    if (batch < batch_size && features < output_features)
        output[batch * output_features + features] += bias[features];
    
}

void Linear(cublasHandle_t cublas_handle, const float* input_data, 
            const float* weights_data, const float* bias_data,
            float* output_data, int batch_size, int input_features,
            int output_features){

            
            const float alpha = 1.0f;
            const float beta = 0.0f;

            checkCublasStatus(cublasSgemm(
                cublas_handle,
                CUBLAS_OP_T,           // weights is row‐major (in_features × out_features)
                CUBLAS_OP_T,           // input is row‐major (batch_size × in_features)
                output_features,       // m
                batch_size,            // n
                input_features,        // k
                &alpha,
                weights_data,          // pointer to device weights
                input_features,        // lda (leading dim of row‐major “weights” when transposed)
                input_data,            // pointer to device input
                input_features,           // ldb (leading dim of row‐major “input” when transposed)
                &beta,
                output_data,           // pointer to device output
                output_features           // ldc (leading dim of column‐major output)
                ));

            int total_ele = batch_size * output_features;
            int block_size = 256;
            int num_blocks = (total_ele + block_size - 1) / block_size;

            addBiasKernel<<<num_blocks, block_size>>>(output_data, bias_data, batch_size, output_features);
            checkCudaStatus(cudaGetLastError());
            checkCudaStatus(cudaDeviceSynchronize());
        }