#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK_ROCBLAS_ERROR(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS Error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    const int M = 2;
    const int N = 3;
    const int K = 4;

    float h_A[M * K] = {1, 2, 3, 4,
                        5, 6, 7, 8};

    float h_B[K * N] = {1, 4, 7,
                        2, 5, 8,
                        3, 6, 9,
                        4, 7, 10};

    float h_C[M * N] = {0};  

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, M * N * sizeof(float));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_none,
                                      N,        // M (cols of C)
                                      M,        // N (rows of C)
                                      K,        // K
                                      &alpha,
                                      d_B, N,   // B, ldb
                                      d_A, K,   // A, lda
                                      &beta,
                                      d_C, N)); // C, ldc

    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Matrix A (" << M << "x" << K << "):\n";
    print_matrix(h_A, M, K);
    std::cout << "\nMatrix B (" << K << "x" << N << "):\n";
    print_matrix(h_B, K, N);
    std::cout << "\nMatrix C = A x B (" << M << "x" << N << "):\n";
    print_matrix(h_C, M, N);

    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
