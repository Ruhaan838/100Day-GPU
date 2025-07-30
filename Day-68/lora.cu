#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace std;

void CUDA_CHECK(cudaError_t error){
    if(error != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << " : " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
}

__global__ void lora_kernel(const float* x, const float *W, const float *A, const float *B, float* y, 
    int M, int N, int K, int R){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float acc = 0.0f;

        for(int k = 0; k < K; k++){
            float sum_ab = 0.0f;
            for(int r = 0; r < R; r++){
                sum_ab += A[k * R + r] * B[k * N + col];
            }

            float w_eff = W[k * N + col] + sum_ab;
            acc += x[row * K + k] * w_eff;
        }
        y[row * N + col] = acc;
    }
}

void cpu_lora(const float* x, const float *W, const float* A, const float* B, float* y, 
    int M, int N, int K, int R){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float acc = 0.0f;
            for(int k = 0; k < K; k++){
                float sum_ab = 0.0f;
                for(int r = 0; r < R; r++){
                    sum_ab += A[k * R + r] * B[r * N + j];
                }
                float w_eff = W[k * N + j] + sum_ab;
                acc += x[i * K + k] * w_eff;
            }
            y[i * N + j] = acc;
        }
    }
}

void print_mat(const float* mat, int M, int N, int max_rows = 5, int max_cols = 5){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            cout << mat[i * N + j] << "\t";
        }
        cout << '\n';
    }
}

int main(){
    int M = 128, K = 256, N = 64, R = 32;

    size_t size_x = M * K * sizeof(float);
    size_t size_W = K * N * sizeof(float);
    size_t size_A = K * R * sizeof(float);
    size_t size_B = R * N * sizeof(float);
    size_t size_y = M * N * sizeof(float);

    float *x = (float*)malloc(size_x);
    float *W = (float*)malloc(size_W);
    float *A = (float*)malloc(size_A);
    float *B = (float*)malloc(size_B);
    float *y = (float*)malloc(size_y);
    float *y_cpu = (float*)malloc(size_y);

    for(int i = 0; i < M * K; i++) x[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < N * K; i++) W[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < R * K; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < R * N; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *dx, *dw, *da, *db, *dy;
    CUDA_CHECK(cudaMalloc((void**)&dx, size_x));
    CUDA_CHECK(cudaMalloc((void**)&dw, size_W));
    CUDA_CHECK(cudaMalloc((void**)&da, size_A));
    CUDA_CHECK(cudaMalloc((void**)&db, size_B));
    CUDA_CHECK(cudaMalloc((void**)&dy, size_y));

    CUDA_CHECK(cudaMemcpy(dx, x, size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, W, size_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(da, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, B, size_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    CUDA_CHECK(cudaEventRecord(start, 0));

    lora_kernel<<<blocks, threads>>>(dx, dw, da, db, dy, M, N, K, R);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    CUDA_CHECK(cudaMemcpy(y, dy, size_y, cudaMemcpyDeviceToHost));

    auto cpu_start = chrono::high_resolution_clock::now();
    cpu_lora(x, W, A, B, y_cpu, M, N, K, R);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpuTime = cpu_end - cpu_start;

    float max_diff = 0.0f;
    for(int i = 0; i < M * N; i++){
        float diff = fabsf(y_cpu[i] - y[i]);
        if (diff > max_diff)
            max_diff = diff;
    }

    cout << "Max Diffrence on GPU and CPU: " << max_diff << '\n';
    cout << "GPU Time: " << gpuTime << " ms" << '\n';
    cout << "CPU Time: " << cpuTime.count() << " ms" << '\n';


    cout << "GPU result";
    print_mat(y, M, N, M, N);


    cout << "CPU result";
    print_mat(y_cpu, M, N, M, N);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(x); free(W); free(A); free(B); free(y); free(y_cpu);
    cudaFree(dx); cudaFree(dw); cudaFree(da); cudaFree(db); cudaFree(dy);

    return 0;
}