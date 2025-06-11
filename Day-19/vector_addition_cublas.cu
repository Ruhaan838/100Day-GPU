#include <iostream>
#include <cublas_v2.h>

using namespace std;

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    for(int i = 0; i < N; i++){
        A[i] = i; B[i] = i;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t vec_bit  = N * sizeof(float);
    //alocate cuda memeory
    float *da, *db;
    cudaMalloc(&da, vec_bit);
    cudaMalloc(&db, vec_bit);

    cudaMemcpy(da, A, vec_bit, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B, vec_bit, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;

    //perfroming addition
    // 
    cublasSaxpy(handle, N, &alpha, da, 2, db, 2);

    cudaMemcpy(C, db, vec_bit, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
        cout << C[i] << " ";
    
    cout << '\n';

    cudaFree(da);
    cudaFree(db);
    cublasDestroy(handle);

    return 0;
}