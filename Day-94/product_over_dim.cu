#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

__global__ void prod_reduce_kernel(const float* input, float* output, size_t M, size_t S_d, size_t N){
    size_t out_idx = blockIdx.x;

    size_t m = out_idx / N;
    size_t n = out_idx - m * N;

    const float* base = input + (m * S_d) * N + n;

    double prod = 1.0;
    for(size_t k = threadIdx.x; k < S_d; k+=blockDim.x){
        prod *= static_cast<double>(base[k*N]);
    }
    constexpr unsigned FULL_MASK = 0xffffffffu;
    for(int offset = warpSize/2; offset > 0; offset >>=1){
        prod *= __shfl_down_sync(FULL_MASK, prod, offset);
    }

    __shared__ double warp_preod[1024/32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_preod[wid] = prod;
    __syncthreads();

    if (wid == 0){
        int temp_idx = (blockDim.x + 32 - 1) / 32;
        double block_prod = (lane < (temp_idx)) ? warp_preod[lane] : 1.0;
        for(int offset = temp_idx; offset > 0; offset >>= 1){
            block_prod *= __shfl_down_sync(FULL_MASK, block_prod, offset);
        }
        if (lane == 0){
            output[out_idx] = static_cast<float>(block_prod);
        }
    }
}

int main() {
    size_t M = 2;    
    size_t S_d = 4;  
    size_t N = 3;    

    size_t in_size = M * S_d * N;
    size_t out_size = M * N;

    vector<float> h_input(in_size);
    vector<float> h_output(out_size, 0.0f);

    iota(h_input.begin(), h_input.end(), 1.0f);

    float *d_input, *d_output;
    cudaMalloc(&d_input, in_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;           
    int blocks = out_size;       
    prod_reduce_kernel<<<blocks, threads>>>(d_input, d_output, M, S_d, N);

    cudaMemcpy(h_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output:\n";
    for (size_t i = 0; i < out_size; i++) {
        cout << h_output[i] << " ";
    }
    cout << "\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}