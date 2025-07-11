#include <cuda_runtime.h>
#include <iostream>

const int cache_size = 256;

using namespace std;

void cudaCheck(cudaError_t err){
    if (err != cudaSuccess){
        cerr << "CUDA error in " << __FILE__ << "at " << __LINE__ << ": " << cudaGetErrorString(err) << '\n';
        exit(EXIT_FAILURE);
    }
}

// C = A * x
__global__ void matvecMul(const float *mat, const float *vec, float *out, int n){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0;
    for (int j = 0; j < n; j++){
        sum += mat[row * n + j] * vec[j];
    }
    out[row] = sum;
}

__global__ void vectorAdd(float *y, const float *x, const float alpha, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        y[idx] += alpha * x[idx];
    }
}

__global__ void vectorSub(float* y, const float *x, const float alpha, int n){
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        y[idx] -= alpha * x[idx];
    }
}

__global__ void vectorScale(float* y, const float alpha, int n){
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        y[idx] *= alpha;
    }
}

__global__ void dotProduct(const float* a, const float *b, float* ans, int n){
    __shared__ float cache[cache_size];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    if (i < n) {
        temp = a[i] * b[i];
    }
    cache[tid] = temp;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(ans, cache[0]);
    }
}



struct GpuVector{
    float* data;
    int n;
    size_t size;

    GpuVector(int size_): n(size_), size(size_ * sizeof(float)){
        cudaCheck(cudaMalloc(&data, size));
    }

    ~GpuVector(){
        cudaFree(data);
    }

    void setHostData(const float *h_data){
        cudaCheck(cudaMemcpy(data, h_data, size, cudaMemcpyHostToDevice));
    }

    void getHostData(float *h_data) const {
        cudaCheck(cudaMemcpy(h_data, data, size, cudaMemcpyDeviceToHost));
    }

    void zero(){
        cudaCheck(cudaMemset(data, 0, size));
    }

    void operator+=(const GpuVector &x){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        vectorAdd<<<grid_size, block_size>>>(data, x.data, 1.0f, n);
    }

    void axpy(float alpha, const GpuVector &x){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        vectorAdd<<<grid_size, block_size>>>(data, x.data, alpha, n);
    }

    void axpy_sub(float alpha, const GpuVector &x){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        vectorSub<<<grid_size, block_size>>>(data, x.data, alpha, n);
    }

    void scale(float beta){
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        vectorScale<<<grid_size, block_size>>>(data, beta, n);
    }

    float dot(const GpuVector &x) const {
        float *d_dot;
        cudaCheck(cudaMalloc(&d_dot, sizeof(float)));
        cudaCheck(cudaMemset(d_dot, 0, sizeof(float)));
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        dotProduct<<<grid_size, block_size>>>(data, x.data, d_dot, n);
        cudaCheck(cudaDeviceSynchronize());
        float result = 0.0f;
        cudaCheck(cudaMemcpy(&result, d_dot, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_dot);
        return result;
    }
};


struct GpuMatrix{
    float *data;
    int n;
    size_t mat_size;

    GpuMatrix(int size_) : n(size_), mat_size(size_ * size_ * sizeof(float)){
        cudaCheck(cudaMalloc(&data, mat_size));
    }

    ~GpuMatrix(){
        cudaFree(data);
    }

    void setHostData(const float *h_data){
        cudaCheck(cudaMemcpy(data, h_data, mat_size, cudaMemcpyHostToDevice));
    }

    void multiply(const GpuVector &in, GpuVector &out) const {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        matvecMul<<<grid_size, block_size>>>(data, in.data, out.data, n);
    }
};

int main(){
    const int n = 4;

    float h_A[n * n] = {
        4, 1, 0, 0,
        1, 3, 1, 0,
        0, 1, 2, 1,
        0, 0, 1, 1
    };
    float h_b[n] = {15, 10, 10, 10};
    float h_x[n] = {0};

    GpuMatrix A(n);
    A.setHostData(h_A);

    GpuVector b(n), x(n), r(n), p(n), Ap(n);

    b.setHostData(h_b);
    x.zero();

    cudaCheck(cudaMemcpy(r.data, b.data, n * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(p.data, r.data, n * sizeof(float), cudaMemcpyDeviceToDevice));

    float rdotr = r.dot(r);
    float new_rdotr = 0.0f;

    int max_iter = 1000;
    float tol = 1e-6f;
    int k = 0;

    while (sqrtf(rdotr) > tol && k < max_iter){
        A.multiply(p, Ap);
        float pAp = p.dot(Ap);
        float alpha = rdotr / pAp;

        x.axpy(alpha, p);
        r.axpy_sub(alpha, Ap);

        new_rdotr = r.dot(r);
        if(sqrtf(new_rdotr) < tol) break;

        float beta = new_rdotr / rdotr;
        p.scale(beta);
        p += r;

        rdotr = new_rdotr;
        k++;
    }

    x.getHostData(h_x);
    cout << "Conjugate Gradient converged in " << k << " iterations. \n";
    cout << "Solution x:\n";
    for(int i = 0; i < n; i++) cout << h_x[i] << "\n";

}