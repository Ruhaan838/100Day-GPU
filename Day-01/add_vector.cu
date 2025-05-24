#include <iostream>

__global__ void add_vector(const float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int main(){
    const int N = 10;
    float a[N], b[N], c[N];

    float *da, *db, *dc;
    cudaMalloc(&da, N*sizeof(float));
    cudaMalloc(&db, N*sizeof(float));
    cudaMalloc(&dc, N*sizeof(float));

    cudaMemcpy(da, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N*sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = ceil(N/blocksize);

    add_vector<<<gridsize, blocksize>>>(da, db, dc, N);

    cudaMemcpy(c, dc, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}