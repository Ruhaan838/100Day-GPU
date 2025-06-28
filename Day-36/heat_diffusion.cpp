#include <hip/hip_runtime.h>
#include <iostream>

const int N = 100;            
const int T = 1000;           
const float DX = 0.1f;        
const float DT = 0.01f;       
const float ALPHA = 0.1f;     
const int BLOCK_SIZE = 16;    
const int TILE_SIZE = (BLOCK_SIZE - 2); 

void HIP_ERROR(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

__global__ void heat_diffusion_kernel(const float* __restrict__ u,
                                      float* __restrict__ u_new,
                                      int n) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int gx = blockIdx.x * TILE_SIZE + threadIdx.x - 1;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (gx >= 0 && gx < n && gy >= 0 && gy < n) {
        tile[ty][tx] = u[gy * n + gx];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    if (tx > 0 && tx < BLOCK_SIZE - 1 && ty > 0 && ty < BLOCK_SIZE - 1) {
        if (gx >= 1 && gx < n - 1 && gy >= 1 && gy < n - 1) {
            float d2u_dx2 = (tile[ty][tx + 1] - 2.0f * tile[ty][tx] + tile[ty][tx - 1]) / (DX * DX);
            float d2u_dy2 = (tile[ty + 1][tx] - 2.0f * tile[ty][tx] + tile[ty - 1][tx]) / (DX * DX);
            u_new[gy * n + gx] = u[gy * n + gx] + ALPHA * DT * (d2u_dx2 + d2u_dy2);
        }
    }
}

int main() {
    float *u = new float[N * N];
    float *u_new = new float[N * N];
    float *du, *du_new;

    // Initialize grid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float num = (i == 0 || i == N - 1 || j == 0 || j == N - 1) ? 100.0f : 0.0f;
            u[i * N + j] = num;
            u_new[i * N + j] = num;
        }
    }

    HIP_ERROR(hipMalloc(&du, N * N * sizeof(float)));
    HIP_ERROR(hipMalloc(&du_new, N * N * sizeof(float)));

    HIP_ERROR(hipMemcpy(du, u, N * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_ERROR(hipMemcpy(du_new, u_new, N * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    for (int t = 0; t < T; ++t) {
        heat_diffusion_kernel<<<grid_size, block_size>>>(du, du_new, N);
        HIP_ERROR(hipGetLastError());
        std::swap(du, du_new);
    }

    HIP_ERROR(hipMemcpy(u_new, du, N * N * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << u_new[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    HIP_ERROR(hipFree(du));
    HIP_ERROR(hipFree(du_new));
    delete[] u;
    delete[] u_new;

    return 0;
}
