#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define NUM_CLUSTER 2
#define N 1024
#define THREADS_PER_BLOCK 256
#define PI 3.1415

using namespace std;

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void EstepKernel(float* data, int n, float* mu, float* sigma, float* pival, float* reposi){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        float x = data[idx];
        float probs[NUM_CLUSTER];
        float sum = 0.0f;

        for (int k = 0; k < NUM_CLUSTER; k++){
            float diff = x - mu[k];
            float expoent = -0.5f * (diff * diff) / (sigma[k] * sigma[k]);
            float gauss = (1.0f / (sqrtf(2.0f * PI) * sigma[k])) * expf(expoent);
            probs[k] = pival[k] * gauss;
            sum += probs[k];
        }

        for (int k = 0; k < NUM_CLUSTER; k++)
            reposi[idx * NUM_CLUSTER + k] = probs[k] / sum;

    }
}

__global__ void MstepKernel(float* data, int n, float* reposi, float* sum_gamma, float* sum_x, float* sum_x2){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        float x = data[idx];
        for (int k = 0; k < NUM_CLUSTER; k++){
            float gamma = reposi[idx * NUM_CLUSTER + k];
            atomicAdd(&sum_gamma[k], gamma);
            atomicAdd(&sum_x[k], gamma * x);
            atomicAdd(&sum_x2[k], gamma * x * x);
        }
    }
}

int main(){
    srand(static_cast<unsigned>(time(NULL)));
    float data[N];
    for (int i = 0; i < N; i++){
        if (i < N / 2)
            data[i] = 2.0f + static_cast<float>(rand()) / RAND_MAX;
        else
            data[i] = 8.0f + static_cast<float>(rand()) / RAND_MAX;
    }

    float mu[NUM_CLUSTER] = {1.0f, 9.0f};
    float sigma[NUM_CLUSTER] = {1.0f, 1.0f};
    float pival[NUM_CLUSTER] = {0.5f, 0.5f};

    float *d_data, *d_mu, *d_sigma, *d_pival;
    float *d_reponsi, *d_sum_gamma, *d_sum_x, *d_sum_x2;

    size_t temp_size = NUM_CLUSTER * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mu, temp_size));
    CHECK_CUDA(cudaMalloc(&d_sigma, temp_size));
    CHECK_CUDA(cudaMalloc(&d_pival, temp_size));
    CHECK_CUDA(cudaMalloc(&d_reponsi, N * temp_size));
    CHECK_CUDA(cudaMalloc(&d_sum_gamma, temp_size));
    CHECK_CUDA(cudaMalloc(&d_sum_x, temp_size));
    CHECK_CUDA(cudaMalloc(&d_sum_x2, temp_size));

    CHECK_CUDA(cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mu, mu, temp_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigma, sigma, temp_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pival, pival, temp_size, cudaMemcpyHostToDevice));

    dim3 gird_size((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    float sum_gamma[NUM_CLUSTER];
    float sum_x[NUM_CLUSTER];
    float sum_x2[NUM_CLUSTER];

    int Max_Iter = 100;
    for (int iter = 0; iter < Max_Iter; iter++){
        EstepKernel<<<gird_size, THREADS_PER_BLOCK>>>(d_data, N, d_mu, d_sigma, d_pival, d_reponsi);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemset(d_sum_gamma, 0, temp_size));
        CHECK_CUDA(cudaMemset(d_sum_x, 0, temp_size));
        CHECK_CUDA(cudaMemset(d_sum_x2, 0, temp_size));

        MstepKernel<<<gird_size, THREADS_PER_BLOCK>>>(d_data, N, d_reponsi, d_sum_gamma, d_sum_x, d_sum_x2);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(sum_gamma, d_sum_gamma, temp_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(sum_x, d_sum_x, temp_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(sum_x2, d_sum_x2, temp_size, cudaMemcpyDeviceToHost));

        float thrasold = 1e-6f;
        for (int k = 0; k < NUM_CLUSTER; k++){
            if (sum_gamma[k] > thrasold){
                mu[k] = sum_x[k] / sum_gamma[k];

                float var = sum_x2[k] / sum_gamma[k] - mu[k] * mu[k];
                sigma[k] = sqrtf(max(var, thrasold));

                pival[k] = sum_gamma[k] / N;
            }
        }

        CHECK_CUDA(cudaMemcpy(d_mu, mu, temp_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sigma, sigma, temp_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_pival, pival, temp_size, cudaMemcpyHostToDevice));

        if (iter % 10 == 0){
            cout << "Iter" << iter << ":\n";
            for (int k = 0; k < NUM_CLUSTER; k++){
                cout << "cluster " << k << ": " << "mu = " << mu[k] << ", " << "sigma = " << sigma[k] << ", " 
                << "pival = " << pival[k] << '\n';
            }
            cout << '\n';
        }
    }

    cudaFree(d_data);
    cudaFree(d_mu);
    cudaFree(d_sum_gamma);
    cudaFree(d_sigma);
    cudaFree(d_pival);
    cudaFree(d_reponsi);
    cudaFree(d_sum_x2);
    cudaFree(d_sum_x2);
}