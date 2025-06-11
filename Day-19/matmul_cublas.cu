#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

void init_data(float* data, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            data[i * y + j] = i + j;  // row-major: data[row * cols + col]
        }
    }
}

void print_mat(float* data, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            printf("%f ", data[i * y + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed!\n");
        return -1;
    }

    int M = 2, N = 3, K = 4;  
    float *a, *b, *c;
    a = (float*)malloc(M * K * sizeof(float));  
    b = (float*)malloc(K * N * sizeof(float));  
    c = (float*)malloc(M * N * sizeof(float));  

    init_data(a, M, K);
    init_data(b, K, N);

    float *da, *db, *dc;
    cudaMalloc(&da, M * K * sizeof(float));
    cudaMalloc(&db, K * N * sizeof(float));
    cudaMalloc(&dc, M * N * sizeof(float));

    cudaMemcpy(da, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    // C = alpha * op(A) * op(B) + beta * C
    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  
        M, N, K,
        &alpha,
        da, K,  
        db, N,  
        &beta,
        dc, M   
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SGEMM failed!\n");
        return -1;
    }

    cudaMemcpy(c, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Mat A:\n");
    print_mat(a, M, K);

    printf("Mat B:\n");
    print_mat(b, K, N);

    printf("Mat C:\n");
    print_mat(c, M, N);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    cublasDestroy(handle);

    return 0;
}
