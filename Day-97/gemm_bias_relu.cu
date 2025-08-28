#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

constexpr int TILE_M = 32;
constexpr int TILE_B = 32;
constexpr int TILE_K = 16;

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

static_assert(TILE_M % (THREADS_X * 2) == 0, "TILE_M must be divisible by THREADS_X * 2");
static_assert(TILE_B % (THREADS_Y * 2) == 0, "TILE_B must be divisible by THREADS_Y * 2");

#define REG_TILE_B 2
#define REG_TILE_M 2

__launch_bounds__ (THREADS_X * THREADS_Y, 4)
__global__ void gemm_bias_relu(const float* A,
    const float* W, const float* b, float* C, size_t B, size_t N, size_t M){
        int block_row = blockIdx.x;
        int block_col = blockIdx.y;

        float acc[REG_TILE_M][REG_TILE_B] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int local_m0 = 2 * tx;
        int local_m1 = 2 * ty;

        __shared__ float sA[TILE_B][TILE_K];
        __shared__ float sW[TILE_K][TILE_M];

        for(int k0 = 0; k0 < (int)N; k0 += TILE_K){
            int flatId = ty * THREADS_X + tx;
            #pragma unroll
            for(int i = 0; i < 2; ++i){
                int idx = flatId + i * THREADS_X * THREADS_Y;

                int a_row = idx / TILE_K;
                int a_col = idx % TILE_K;
                int global_row = block_row * TILE_B + a_row;
                int global_col = k0 + a_col;
                float a_val = 0.0f;
                if (global_row < (int)B && global_col < (int)N){
                    a_val = A[global_row * N + global_col];
                }
                sA[a_row][a_col] = a_val;

                int w_id = idx;
                int w_row = w_id / TILE_M;
                int w_col = w_id % TILE_M;
                int global_m = block_col * TILE_M + w_col;
                int global_k = k0 + w_row;
                float w_val = 0.0f;
                if (global_k < (int)N && global_m < (int)M){
                    w_val = W[global_k * M + global_m];
                }
                sW[w_row][w_col] = w_val;
            }
            __syncthreads();

            #pragma unroll
            for(int kk=0; kk < TILE_K; ++kk){
                float a_reg[REG_TILE_B];
                a_reg[0] = sA[local_m1 + 0][kk];
                a_reg[1] = sA[local_m1 + 1][kk];

                float w_reg[REG_TILE_M];
                w_reg[0] = sW[kk][local_m0 + 0];
                w_reg[1] = sW[kk][local_m0 + 1];

                #pragma unroll
                for(int i = 0; i < REG_TILE_B; ++i){
                    for(int j = 0; j < REG_TILE_M; ++j){
                        acc[i][j] += a_reg[i] * w_reg[j];
                    }
                }
            }
            __syncthreads();
        }
        int base_row = block_row * TILE_B;
        int base_col = block_col * TILE_M;

        for (int i = 0; i < REG_TILE_B; ++i) {
            for (int j = 0; j < REG_TILE_M; ++j) {
                int global_row = base_row + i;
                int global_col = base_col + j;
                if (global_row < (int)B && global_col < (int)M) {
                    C[global_row * M + global_col] = acc[i][j];
                }
            }
        }
}

int main() {
    const int B = 4; 
    const int N = 4; 
    const int M = 4; 

    vector<float> h_A(B * N);
    vector<float> h_W(N * M);
    vector<float> h_b(M, 0.0f);
    vector<float> h_C(B * M, 0.0f);

    for (int i = 0; i < B * N; i++) h_A[i] = i + 1;
    for (int i = 0; i < N * M; i++) h_W[i] = (i % M) + 1;

    float *d_A, *d_W, *d_b, *d_C;
    cudaMalloc(&d_A, B * N * sizeof(float));
    cudaMalloc(&d_W, N * M * sizeof(float));
    cudaMalloc(&d_b, M * sizeof(float));
    cudaMalloc(&d_C, B * M * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), B * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((B + TILE_B - 1) / TILE_B, (M + TILE_M - 1) / TILE_M);

    gemm_bias_relu<<<blocks, threads>>>(d_A, d_W, d_b, d_C, B, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, B * M * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output C (" << B << "x" << M << "):\n";
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < M; j++) {
            cout << h_C[i * M + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_C);

    return 0;
}