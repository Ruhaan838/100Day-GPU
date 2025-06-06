#include "cuda_helper.cuh"
#include "cuda_kernels.cuh"


int main(){
    const int batch_size = 2;
    const int in_features = 128;
    const int out_features = 64;

    const size_t in_size = batch_size * in_features;
    const size_t weight_size = in_features * out_features;
    const size_t bias_size = out_features;
    const size_t output_size = batch_size * out_features;

    float* input = new float[in_size];
    float* weights = new float[weight_size];
    float* bias = new float[bias_size];
    float* output = new float[output_size];


    init_random_mat(input, in_size, -1.0f, 1.0f);
    init_random_mat(weights, weight_size, -1.0f, 1.0f);
    init_random_mat(bias, bias_size, -1.0f, 1.0f);

    float *d_input, *d_weights, *d_bias, *d_output;
    allocate_Device_Mem(&d_input, in_size * sizeof(float));
    allocate_Device_Mem(&d_weights, weight_size * sizeof(float));
    allocate_Device_Mem(&d_bias, bias_size * sizeof(float));
    allocate_Device_Mem(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    Linear(cublas_handle, d_input, d_weights, d_bias, d_output, batch_size, in_features, out_features);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    printMat("Input:", input, batch_size, in_features);
    printMat("Weights:", weights, in_features, out_features);
    printMat("Bias:", bias, 1, out_features);
    printMat("Output:", output, batch_size, out_features);

    cublasDestroy(cublas_handle);
    free_Device_Mem(d_input);
    free_Device_Mem(d_weights);
    free_Device_Mem(d_bias);
    free_Device_Mem(d_output);

    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output;

    return 0;

}