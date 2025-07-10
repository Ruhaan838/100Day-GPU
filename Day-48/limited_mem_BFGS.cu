#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void dotCudaKernel(const double* a, const double* b, double* result, int n){
    __shared__ double cache[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_idx = threadIdx.x;
    double temp = 0.0;
    while (idx < n){
        temp += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }
    cache[cache_idx] = temp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (cache_idx < stride)
            cache[cache_idx] += cache[cache_idx + stride];
        __syncthreads();
    }
    if (cache_idx == 0)
        atomicAdd(result, cache[0]);
}

__global__ void addCudaKernel(double* a, const double* b, double scaler, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] += scaler * b[idx];
}

__global__ void subCudaKernel(double* a, const double* b, double scaler, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] -= scaler * b[idx];
}

__global__ void scaleCudaKernel(double* a, double scaler, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] *= scaler;
}

struct CudaVector{
    double* d_data;
    int n;

    CudaVector(int n_) : n(n_){
        cudaMalloc(&d_data, n * sizeof(double));
    }

    ~CudaVector(){
        cudaFree(d_data);
    }

    void copyFromHost(double* h_data) {cudaMemcpy(d_data, h_data, n * sizeof(double), cudaMemcpyHostToDevice);}
    void copyToHost(double* h_data) {cudaMemcpy(h_data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost);}

    double dot(const CudaVector& other) const {
        double result = 0.0;
        double* d_result;
        cudaMalloc(&d_result, sizeof(double));
        cudaMemset(d_result, 0, sizeof(double));

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        dotCudaKernel<<<grid_size, block_size>>>(d_data, other.d_data, d_result, n);
        cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    }

    void operator*=(double scaler){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        scaleCudaKernel<<<grid_size, block_size>>>(d_data, scaler, n);
    }

    void operator+=(const CudaVector& other){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        addCudaKernel<<<grid_size, block_size>>>(d_data, other.d_data, 1.0, n);
    }

    void addScaled(const CudaVector& other, double scalar){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        addCudaKernel<<<grid_size, block_size>>>(d_data, other.d_data, scalar, n);
    }

    void subScaled(const CudaVector& other, double scalar){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        subCudaKernel<<<grid_size, block_size>>>(d_data, other.d_data, scalar, n);
    }

    void copyFrom(const CudaVector& other){
        cudaMemcpy(d_data, other.d_data, n * sizeof(double), cudaMemcpyDeviceToHost);
    }
};

int main(){

    const int n = 1024;

    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_grad = (double*)malloc(n * sizeof(double));
    double* h_s = (double*)malloc(n * sizeof(double));
    double* h_y = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++){
        h_x[i] = (double)rand() / RAND_MAX;
        h_grad[i] = (double)rand() / RAND_MAX;
        h_s[i] = (double)rand() / RAND_MAX;
        h_y[i] = (double)rand() / RAND_MAX;
    }

    CudaVector d_x(n), d_grad(n), d_s(n), d_y(n), d_q(n), d_r(n);

    d_x.copyFromHost(h_x);
    d_grad.copyFromHost(h_grad);
    d_s.copyFromHost(h_s);
    d_y.copyFromHost(h_y);

    d_q.copyFrom(d_grad);

    double rho = 1.0 / d_s.dot(d_y);
    double alpha = rho * d_s.dot(d_q);

    d_q.subScaled(d_y, alpha);
    double H0 = d_s.dot(d_y) / d_y.dot(d_y);

    d_r.copyFrom(d_q);
    d_r *= H0;
    double beta = rho * d_y.dot(d_r);

    d_r.addScaled(d_s, alpha - beta);
    d_r *= -1.0;

    double step = 0.1;
    d_x.addScaled(d_r, step);
    d_x.copyToHost(h_x);

    printf("Updated parameters:\n");
    for(int i = 0; i < 10; i++)
        printf("x[%d] = %f\n", i, h_x[i]);

    free(h_x); free(h_grad); free(h_s); free(h_y);

    return 0;
}