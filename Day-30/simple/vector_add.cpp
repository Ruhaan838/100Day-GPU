#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(){
    const int N = 10;
    float A[N], B[N], C[N];

    for(int i = 0; i < N; i++){
        A[i] = i;
        B[i] = i * 2;
    }

    float *da, *db, *dc;
    size_t size = N * sizeof(float);
    hipMalloc(&da, size);
    hipMalloc(&db, size);
    hipMalloc(&dc, size);

    hipMemcpy(da, A, size, hipMemcpyHostToDevice);
    hipMemcpy(db, B, size, hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    hipLunchKernelGGL(vectorAdd, dim3(grid_size), dim3(block_size), 0, 0, da, db, dc, N);
    hipMemcpy(C, dc, size, hipMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
        cout << A[i] << " + " << B[i] << " = " << C[i] << '\n';

    hipFree(da);
    hipFree(db);
    hipFree(dc);

    return 0;

}