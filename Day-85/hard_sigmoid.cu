#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace std;

__global__ void hard_sigmoid_kernel(const float* input, float* output, size_t total_ele){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_ele) return;

    float x = input[idx];
    if (x <= -3.0f)
        output[idx] = 0.0f;
    else if (x >= 3.0f)
        output[idx] = 1.0f;
    else
        output[idx] = (x + 3.0f) / 6.0f;
}

int main(){
    const size_t N = 10;
    vector<float> h_input = {-5.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    vector<float> h_output(N, 0.0f);

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hard_sigmoid_kernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}