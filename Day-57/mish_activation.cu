#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void mish_kernel(float* x, float* y, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        float val = x[idx];
        float soft = logf(1 + expf(val));
        y[idx] = val * tanf(soft);
    }
}

void mish_cuda(float* dx, float* dy, int size){
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mish_kernel<<<blocks, threads>>>(dx, dy, size);
    cudaDeviceSynchronize();
}

int main(){
    const int N = 10;
    size_t size = N * sizeof(float);
    float* x = new float[N];
    float* y = new float[N];

    for(int i = 0; i < N; i++){
        x[i] = rand() % 10 + 1;
    }

    float *dx, *dy;
    cudaMalloc(&dx, size);
    cudaMalloc(&dy, size);

    cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
    mish_cuda(dx, dy, N);
    cudaMemcpy(y, dy, size, cudaMemcpyDeviceToHost);

    cout << "output:";
    for(int i = 0; i < N; i++)
        cout << x[i]<< " " << y[i] << "\n";

    cudaFree(dx);
    cudaFree(dy);

    return 0;

}