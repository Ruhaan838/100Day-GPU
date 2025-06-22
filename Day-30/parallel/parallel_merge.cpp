#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <iostream>

__device__ __host__ inline int CUDA_MAX(int a, int b) { return (a > b) ? a : b; }
__device__ __host__ inline int CUDA_MIN(int a, int b) { return (a < b) ? a : b; }

__device__ void simple_merge_sort(const int* A, const int* B, int k, const int N, const int M, int* i_out, int* j_out){
    int low = CUDA_MAX(0, k - M);
    int high = CUDA_MIN(k, N);

    while (low <= high){
        int i = (low + high) / 2;
        int j = k - i;

        if (j < 0){
            high = i - 1;
            continue;
        } 
        if (j > M){
            low = i + 1;
            continue;
        }

        if (i > 0 && j < M && A[i - 1] > B[j])
            high = i - 1;
        else if (j > 0 && i < N && B[j - 1] > A[i])
            low = i + 1;
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

__global__ void parallel_merge(const int* A, const int* B, int* C, const int N, const int M){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N + M){
        int i, j;
        simple_merge_sort(A, B, idx, N, M, &i, &j);

        if (j >= M || (i < N && A[i] <= B[j]))
            C[idx] = A[i];
        else
            C[idx] = B[j];
    }
}

template<typename T>
void print_data(T* data, int N){
    for (int i = 0; i < N; i++)
        std::cout << data[i] << " ";
    std::cout << '\n';
}

int main(){
    const int N = 5;
    const int M = 5;
    int A[N], B[M], C[N + M];

    for (int i = 0; i < N; i++)
        A[i] = 2 * i;           // 0, 2, 4, 6, 8
    for (int i = 0; i < M; i++)
        B[i] = 2 * i + 1;       // 1, 3, 5, 7, 9

    std::cout << "Array A: ";
    print_data<int>(A, N);
    std::cout << "Array B: ";
    print_data<int>(B, M);

    int *da, *db, *dc;
    hipMalloc(&da, N * sizeof(int));
    hipMalloc(&db, M * sizeof(int));
    hipMalloc(&dc, (N + M) * sizeof(int));

    hipMemcpy(da, A, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(db, B, M * sizeof(int), hipMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_size((N + M + block_dim.x - 1) / block_dim.x);

    hipLaunchKernelGGL(parallel_merge, grid_size, block_dim, 0, 0, da, db, dc, N, M);

    hipMemcpy(C, dc, (N + M) * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(da);
    hipFree(db);
    hipFree(dc);

    std::cout << "Merged Array: ";
    print_data<int>(C, (N + M));

    return 0;
}
