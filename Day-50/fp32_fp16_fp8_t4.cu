#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>

const int TILE_SIZE = 16;
const float FP8_MIN = -127.0f;
const float FP8_MAX = 127.0f;

__device__ uint8_t quantize_fp8(float x){
    x = fmaxf(FP8_MIN, fminf(FP8_MAX, x)); //torch.clamp
    return (uint8_t)((x - FP8_MIN) * 255.0f / (FP8_MAX - FP8_MIN));
}

__device__ float dequantize_fp8(uint8_t x){
    return FP8_MIN + (x * (FP8_MAX - FP8_MIN) / 255.0f);
}

__global__ void matmul_fp32(const float *A, const float* B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_fp16(const __half *A, const __half *B, __half *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N){
        __half sum = 0.0;
        for(int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_fp8(const float *A, const float *B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < N && row < N){
        uint8_t sum = 0.0f;
        for(int k = 0; k < N; k++){
            uint8_t qa = quantize_fp8(A[row * N + k]);
            uint8_t qb = quantize_fp8(B[k * N + col]);
            sum += qa * qb;
        }
        C[row * N + col] = dequantize_fp8(sum);
    }
}

template <typename T>
float benchmark(void (*kernel)(const T*, const T*, T*, int),
                const T *da, const T*db, T *dc, int N){
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(((N + TILE_SIZE - 1) / TILE_SIZE), 
    ((N + TILE_SIZE - 1) / TILE_SIZE));

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(da, db, dc, N);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return ms;
}


int main(){

    int N = 1024;

    float *ha, *hb, *hc;
    float *da, *db, *dc;

    size_t size = N * N * sizeof(float);
    ha = (float*)malloc(size);
    hb = (float*)malloc(size);
    hc = (float*)malloc(size);

    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    for(int i = 0; i < N * N; i ++){
        ha[i] = rand() % 100 / 100.f;
        hb[i] = rand() % 100 / 100.f;
    }

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    printf("\n Matmul Benchmarl (N = %d) \n", N);

    float fp32_time = benchmark<float>(matmul_fp32, da, db, dc, N);
    printf("FP32 Time: %f\n", fp32_time);

    __half *ha_f16, *hb_f16;
    __half *da_f16, *db_f16, *dc_f16;
    size_t size_f16 = N * N * sizeof(__half);

    ha_f16 = (__half*)malloc(size_f16);
    hb_f16 = (__half*)malloc(size_f16);

    for (int i = 0; i < N * N; i++) {
        ha_f16[i] = __float2half(ha[i]);
        hb_f16[i] = __float2half(hb[i]);
    }

    cudaMalloc(&da_f16, size_f16);
    cudaMalloc(&db_f16, size_f16);
    cudaMalloc(&dc_f16, size_f16);

    cudaMemcpy(da_f16, ha_f16, size_f16, cudaMemcpyHostToDevice);
    cudaMemcpy(db_f16, hb_f16, size_f16, cudaMemcpyHostToDevice);

    float fp16_time = benchmark<__half>(matmul_fp16, da_f16, db_f16, dc_f16, N);
    printf("FP16 Time: %f\n", fp16_time);

    float fp8_time = benchmark<float>(matmul_fp8, da, db, dc, N);
    printf("FP8 Time: %f\n", fp8_time);

}