#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;
const int MAX_THREADS = 128;

void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void groupNormForwardKernel(const float* x, float *y, const float *gamma, const float *beta, 
    int N, int C, int H, int W, int G, float eps){
    
    int n = blockIdx.x; //batch_size
    int g = blockIdx.y; //groups

    int group_channels = C / G;
    int spatial_size = H * W;
    int group_size = group_channels * spatial_size;
    int start_c = g * group_channels;

    int tid = threadIdx.x; // this will go thorugh the all groups

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for(int i = tid; i < group_size; i += blockDim.x) {
        int c = start_c + (i / spatial_size);
        int h = (i % spatial_size) / W;
        int w = (i % spatial_size) % W;

        float val = x[n * C * H * W + c * H * W + h * W + w];
        local_sum += val;
        local_sq_sum += val * val;
    }

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = &shared[blockDim.x];

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sq_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / group_size;
    float var = (s_sum_sq[0] / group_size) - (mean * mean);
    var = rsqrtf(var + eps);
    __syncthreads();

    for(int i = tid; i < group_size; i += blockDim.x){
        int c_offset = i / spatial_size;
        int s = i % spatial_size;
        int c = start_c + c_offset;
        int idx = n * C * spatial_size + c * spatial_size + s;
        float val = x[idx];
        float norm = (val - mean) * var;
        y[idx] = norm * gamma[c] + beta[c];
    }
}

int main(){
    const int N = 2;
    const int C = 8;
    const int H = 2;
    const int W = 2;
    const int G = 4;

    const float eps = 1e-5f;
    int size = N * C * H * W;

    float x[size], y[size], gamma[C], beta[C];

    for(int i = 0; i < size; ++i) {
        x[i] = static_cast<float>(i) / size;
    }
    for(int i = 0; i < C; ++i) {
        gamma[i] = 1.0f; 
        beta[i] = 0.0f; 
    }

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaCheck(cudaMalloc((void**)&d_x, size * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_y, size * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_gamma, C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_beta, C * sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_gamma, gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_beta, beta, C * sizeof(float), cudaMemcpyHostToDevice));

    int g_c = C / G;
    int spetial_size = H * W;
    int g_size = g_c * spetial_size;
    int threads_per_block = (g_size < MAX_THREADS) ? g_size : MAX_THREADS;

    size_t shared_mem_size = threads_per_block * 2 * sizeof(float);;
    dim3 grid(N, G);

    groupNormForwardKernel<<<grid, threads_per_block, shared_mem_size>>>(d_x, d_y, d_gamma, d_beta, N, C, H, W, G, eps);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Output of Group Normalization:\n";
    for(int n = 0; n < N; ++n) {
        for(int c = 0; c < C; ++c) {
            for(int h = 0; h < H; ++h) {
                for(int w = 0; w < W; ++w) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    cout << y[idx] << " ";
                }
                cout << '\n';
            }
            cout << '\n';
        }
    }

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));
    cudaCheck(cudaFree(d_gamma));
    cudaCheck(cudaFree(d_beta));
    return 0;
}