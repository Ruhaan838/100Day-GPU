#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vectorMatMult(const float* a, const float* b, float* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sum = 0.0f;
    for(int j = 0; j < N; j++)
        sum += a[idx * N + j] * b[j];
    c[idx] = sum;
}

void print_data(const float* mat, int N, int M){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            printf("%.2f ", mat[i * M + j]);
        }
        printf("\n");
    }
}

void print_data(const float* mat, int N){
    for (int i = 0; i < N; i++)
        printf("%.2f ", mat[i]);
    printf("\n");
}

int main(){
    const int N = 10;
    float *a, *b, *c;

    size_t matrix_size = N * N * sizeof(float);
    size_t vector_size = N * sizeof(float);

    a = (float*)malloc(matrix_size);
    b = (float*)malloc(vector_size);
    c = (float*)malloc(vector_size);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            a[i * N + j] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    float *da, *db, *dc;
    hipMalloc(&da, matrix_size);
    hipMalloc(&db, vector_size);
    hipMalloc(&dc, vector_size);

    hipMemcpy(da, a, matrix_size, hipMemcpyHostToDevice);
    hipMemcpy(db, b, vector_size, hipMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    hipLaunchKernelGGL(vectorMatMult, grid_size, block_size, 0, 0, da, db, dc, N);
    hipDeviceSynchronize();

    hipMemcpy(c, dc, vector_size, hipMemcpyDeviceToHost);

    printf("A:\n");
    print_data(a, N, N);
    printf("B:\n");
    print_data(b, N);
    printf("C (Result A x B):\n");
    print_data(c, N);

    hipFree(da);
    hipFree(db);
    hipFree(dc);
    free(a);
    free(b);
    free(c);

    return 0;
}
