#include <iostream>
#include <cuda_runtime.h>

const int N = 500;
const int BLOCK_SIZE = 16;
const float TOLERANCE = 1e-5;

using namespace std;

__global__ void jacobi_kernel(double *u_new, double *u_old, double *f, int n, double h2){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i < n - 1 && j < n - 1){
        int idx = j * n + i;
        u_new[idx] = 0.25 * (u_old[idx - 1] + u_old[idx + 1] + u_old[idx - n] + u_old[idx + n] - h2 * f[idx]);
    }
}

void poisson(double *u, double *f, int n){
    double *du_old, *du_new, *df;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&du_old, size);
    cudaMalloc(&du_new, size);
    cudaMalloc(&df, size);

    cudaMemcpy(du_old, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(du_new, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(df, f, size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + threads.x - 1) / threads.x, 
                (n + threads.y - 1) / threads.y);

    double h2 = 1.0 / (n * n);
    int maxIter = 10000;

    for(int iter = 0; iter < maxIter; iter++){
        jacobi_kernel<<<blocks, threads>>>(du_new, du_old, df, n, h2);
        cudaMemcpy(du_old, du_new, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(u, du_new, size, cudaMemcpyDeviceToHost);
    cudaFree(du_old);
    cudaFree(du_new);
    cudaFree(df);
}

int main(){
    double *u = new double[N * N]();
    double *f = new double[N * N]();

    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            f[j * N + i] = sin(i * M_PI / N) * sin(j * M_PI / N);
        }
    }

    poisson(u, f, N);

    for(int j = 0; j < N; j += N / 16){
        for(int i = 0; i < N; i += N / 16){
            cout << u[j * N + i] << " ";
        }
        cout << "\n";
    }

    delete[] u;
    delete[] f;
    return 0;
}