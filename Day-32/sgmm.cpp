#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>

using namespace std;

const int BLOCK_SIZE = 256;

__global__ void kernel3_reg(float* a, float* b, float* c, int N, float alpha, float beta) {
    constexpr int BN = 128, BM = 128, BK = 8;
    constexpr int TN = 4, TM = 4;
    constexpr int nbWaves = BLOCK_SIZE / 32;
    constexpr int WN = 64;
    constexpr int WM = BN * BM / nbWaves / WN;
    constexpr int nbWavesX = BN / WN;
    constexpr int nbWavesY = BM / WM;

    int waveId = threadIdx.x / 32;
    int waveX = waveId % nbWavesX;
    int waveY = waveId / nbWavesX;
    int laneId = threadIdx.x % 32;

    constexpr int out_wave_x = 8;
    constexpr int out_wave_y = 4;
    int laneX = laneId % out_wave_x;
    int laneY = laneId / out_wave_x;

    constexpr int inner_wave_N = WN / (out_wave_x * TN);
    constexpr int inner_wave_M = WM / (out_wave_y * TM);
    constexpr int sub_WN = WN / inner_wave_N;
    constexpr int sub_WM = WM / inner_wave_M;

    int thread_idx_a_x = threadIdx.x % BK;
    int thread_idx_a_y = threadIdx.x / BK;
    int thread_idx_b_x = threadIdx.x % BN;
    int thread_idx_b_y = threadIdx.x / BN;

    constexpr int stride_b = BLOCK_SIZE / BN;
    constexpr int stride_a = BLOCK_SIZE / BK;
    constexpr int nb_read_b = BN * BK / BLOCK_SIZE;
    constexpr int nb_read_a = BM * BK / BLOCK_SIZE;

    float A_col[inner_wave_M * TM];
    float B_row[inner_wave_N * TN];
    __shared__ float A_shared[BK][BM];
    __shared__ float B_shared[BK][BN];

    float c_register[TM * inner_wave_M * TN * inner_wave_N] = {0};

    for (int kId = 0; kId < N; kId += BK) {
        for (int i = 0; i < nb_read_b; ++i) {
            int x = BN * blockIdx.x + thread_idx_b_x;
            int y = thread_idx_b_y + i * stride_b + kId;
            B_shared[y % BK][x % BN] = b[N * y + x];
        }

        for (int i = 0; i < nb_read_a; ++i) {
            int x = thread_idx_a_x + kId;
            int y = BM * blockIdx.y + thread_idx_a_y + i * stride_a;
            A_shared[x % BK][y % BM] = a[N * y + x];
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int iterN = 0; iterN < inner_wave_N; ++iterN) {
                for (int i = 0; i < TN; ++i) {
                    int index = waveX * WN + iterN * sub_WN + TN * laneX + i;
                    B_row[iterN * TN + i] = B_shared[k][index];
                }
            }

            for (int iterM = 0; iterM < inner_wave_M; ++iterM) {
                for (int i = 0; i < TM; ++i) {
                    int index = waveY * WM + iterM * sub_WM + TM * laneY + i;
                    A_col[iterM * TM + i] = A_shared[k][index];
                }
            }

            for (int m = 0; m < inner_wave_M; ++m) {
                for (int n = 0; n < inner_wave_N; ++n) {
                    for (int yt = 0; yt < TM; ++yt) {
                        for (int xt = 0; xt < TN; ++xt) {
                            int x = n * TN + xt;
                            int y = m * TM + yt;
                            c_register[y * TN * inner_wave_N + x] += A_col[y] * B_row[x];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < inner_wave_M; ++m) {
        for (int n = 0; n < inner_wave_N; ++n) {
            int xOut = blockIdx.x * BN + waveX * WN + n * sub_WN + TN * laneX;
            int yOut = blockIdx.y * BM + waveY * WM + m * sub_WM + TM * laneY;

            for (int yt = 0; yt < TM; ++yt) {
                for (int xt = 0; xt < TN; ++xt) {
                    int index = N * (yOut + yt) + xOut + xt;
                    c[index] = beta * c[index] + alpha * c_register[TN * inner_wave_N * (m * TM + yt) + n * TN + xt];
                }
            }
        }
    }
}

class SGEMM {
public:
    virtual void init() = 0;
    virtual void run(float* da, float* db, float* dc, float alpha, float beta, int N) = 0;
    virtual void finalize() = 0;
    virtual ~SGEMM() = default;
};

class Kernel3Reg : public SGEMM {
public:
    void init() override;
    void run(float* da, float* db, float* dc, float alpha, float beta, int N) override;
    void finalize() override;
};

void Kernel3Reg::init() {}

void Kernel3Reg::finalize() {}

void Kernel3Reg::run(float* da, float* db, float* dc, float alpha, float beta, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(N / 128, N / 128);
    hipLaunchKernelGGL(kernel3_reg, blocksPerGrid, threadsPerBlock, 0, 0, da, db, dc, N, alpha, beta);
}

void init_mat(vector<float>& mat, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < size * size; ++i) {
        mat[i] = dis(gen);
    }
}

double calculate_gflops(int N, double time_s) {
    double op = 2.0 * pow(N, 3);
    return (op / time_s) / 1e9;
}

int main() {
    vector<int> mat_size = {1024, 2048, 4096};
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int N : mat_size) {
        vector<float> a(N * N);
        vector<float> b(N * N);
        vector<float> c(N * N, 0.0f);
        init_mat(a, N);
        init_mat(b, N);

        float *da, *db, *dc;
        size_t size = N * N * sizeof(float);

        (void)hipMalloc(&da, size);
        (void)hipMalloc(&db, size);
        (void)hipMalloc(&dc, size);
        (void)hipMemcpy(da, a.data(), size, hipMemcpyHostToDevice);
        (void)hipMemcpy(db, b.data(), size, hipMemcpyHostToDevice);
        (void)hipMemcpy(dc, c.data(), size, hipMemcpyHostToDevice);

        Kernel3Reg kernel;
        kernel.init();

        auto start = chrono::high_resolution_clock::now();
        kernel.run(da, db, dc, alpha, beta, N);
        hipDeviceSynchronize();  // Make sure kernel is finished
        auto end = chrono::high_resolution_clock::now();

        double time_sec = chrono::duration<double>(end - start).count();
        double gflops = calculate_gflops(N, time_sec);

        (void)hipMemcpy(c.data(), dc, size, hipMemcpyDeviceToHost);
        cout << "N: " << N << ", Time: " << time_sec * 1000.0 << " ms, GFLOPS: " << gflops << endl;

        kernel.finalize();

        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dc);
    }

    return 0;
}
