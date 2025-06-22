#include <iostream>
#include <hip/hip_runtime.h>

__global__ void MatrixAdd(const float* a, const float* b, float* c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row idx
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col idx

    if (i >= N || j >= N) return;

    c[i * N + j] = a[i * N + j] + b[i * N + j]; // simple add
}

// lol!!, I am sooo lazy that's why this function here !!! XD
void print_mat(const float* mat, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%.2f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main(){
    const int N = 10;
    float *A, *B, *C;

    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    float *da, *db, *dc;

    hipMalloc(&da, N * N * sizeof(float));
    hipMalloc(&db, N * N * sizeof(float));
    hipMalloc(&dc, N * N * sizeof(float));

    hipMemcpy(da, A, N * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(db, B, N * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 dimBlock(32, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    hipLaunchKernelGGL(MatrixAdd, dimGrid, dimBlock, 0, 0, da, db, dc, N);
    hipDeviceSynchronize();

    hipMemcpy(C, dc, N * N * sizeof(float), hipMemcpyDeviceToHost);

    printf("C:\n");
    print_mat(C, N);

    printf("A:\n");
    print_mat(A, N);

    printf("B:\n");
    print_mat(B, N);

    hipFree(da);
    hipFree(db);
    hipFree(dc);

    free(A);
    free(B);
    free(C);

    return 0;
}
