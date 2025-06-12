#include <cudnn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(func) { \
    cudnnStatus_t status = (func); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        cerr << "cuDNN Error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(status) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

const int input_size = 1000;
const int hidden_size = 512;
const int output_size = 10;
const int batch_size = 64;
const int epochs = 10;

void init_weights(float* weights, int size){
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
    curandGenerateUniform(generator, weights, size);
    curandDestroyGenerator(generator);
}

int main(){
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // tensor descriptors
    cudnnTensorDescriptor_t input_desc, hidden1_desc, hidden2_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden1_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden2_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(hidden1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(hidden2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_size, 1, 1));

    // filter descriptors
    cudnnFilterDescriptor_t fc1_w, fc2_w, fc3_w;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc1_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc1_w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, hidden_size, input_size, 1, 1));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc2_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc2_w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, hidden_size, hidden_size, 1, 1));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc3_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc3_w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_size, hidden_size, 1, 1));

    // activation descriptor
    cudnnActivationDescriptor_t relu_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // allocate data
    float *d_input, *d_label;
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_label, batch_size * output_size * sizeof(float)));

    float *dw1, *db1, *dw2, *db2, *dw3, *db3;
    CHECK_CUDA(cudaMalloc(&dw1, hidden_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db1, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw2, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db2, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw3, output_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db3, output_size * sizeof(float)));

    init_weights(dw1, hidden_size * input_size);
    init_weights(dw2, hidden_size * hidden_size);
    init_weights(dw3, output_size * hidden_size);
    CHECK_CUDA(cudaMemset(db1, 0, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(db2, 0, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(db3, 0, output_size * sizeof(float)));

    init_weights(d_input, batch_size * input_size);
    init_weights(d_label, batch_size * output_size);

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));

    for (int epoch = 0; epoch < epochs; epoch++){
        float alpha = 1.0f, beta = 0.0f;

        float *d_hidden_1, *d_hidden_2, *d_output;
        CHECK_CUDA(cudaMalloc(&d_hidden_1, batch_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hidden_2, batch_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));

        // FC1: input -> hidden_1
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, fc1_w, dw1,
                                            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            nullptr, 0, &beta, hidden1_desc, d_hidden_1));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, hidden1_desc, db1, &alpha, hidden1_desc, d_hidden_1));
        CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc, &alpha, hidden1_desc, d_hidden_1,
                                           &beta, hidden1_desc, d_hidden_1));

        // FC2: hidden_1 -> hidden_2
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, hidden1_desc, d_hidden_1, fc2_w, dw2,
                                            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            nullptr, 0, &beta, hidden2_desc, d_hidden_2));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, hidden2_desc, db2, &alpha, hidden2_desc, d_hidden_2));
        CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc, &alpha, hidden2_desc, d_hidden_2,
                                           &beta, hidden2_desc, d_hidden_2));

        // FC3: hidden_2 -> output
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, hidden2_desc, d_hidden_2, fc3_w, dw3,
                                            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            nullptr, 0, &beta, output_desc, d_output));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, output_desc, db3, &alpha, output_desc, d_output));

        cudaFree(d_hidden_1);
        cudaFree(d_hidden_2);

        if (epoch == epochs - 1){
            float* h_output = new float[batch_size * output_size];
            CHECK_CUDA(cudaMemcpy(h_output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
            cout << "Final Results sample " << h_output[0] << '\n';
            delete[] h_output;
        }
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_label));

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(relu_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc1_w));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc2_w));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc3_w));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(hidden1_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(hidden2_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
