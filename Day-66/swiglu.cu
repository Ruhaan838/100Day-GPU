#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

__device__ inline float sigmoid(float x) {return 1.0f / (1.0f + expf(-x));}
__device__ inline float silu(float x) {return x * sigmoid(x);}

__global__ void swiglu_forward_kernel(const float* a, const float* b, float* c, int stride, int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= n_cols) return;
    int offset = row * stride + col;
    float a_val = a[offset];
    float b_val = b[offset];
    c[offset] = silu(a_val) * b_val;
}

__global__ void swiglu_backward_kernel(const float *dc, const float *a, const float *b, 
                                       float *d_a, float *d_b, int stride, int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= n_cols) return;
    int offset = row * stride + col;
    float dc_val = dc[offset];
    float a_val = a[offset];
    float b_val = b[offset];

    float sig_a = sigmoid(a_val);
    float silu_a = a_val * sig_a;
    
    float db = dc_val * silu_a;
    
    float da = dc_val * b_val * sig_a * (1.0f + a_val * (1.0f - sig_a));

    d_a[offset] = da;
    d_b[offset] = db;
}

void do_swiglu(float *a, float *b, float *c, float *da, float *db, int stride, int n_rows, int n_cols) {
    const size_t num_ele = n_rows * n_cols;
    const size_t size = num_ele * sizeof(float);
    float *d_a, *d_b, *d_c, *d_da, *d_db, *d_dc;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_da, size);
    cudaMalloc(&d_db, size);
    cudaMalloc(&d_dc, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    swiglu_forward_kernel<<<n_rows, n_cols>>>(d_a, d_b, d_c, stride, n_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    float ones = 1.0f;
    cudaMemcpy(d_dc, &ones, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_dc, 0, size); 

    float *temp_ones = new float[num_ele];
    for(int i = 0; i < num_ele; i++) temp_ones[i] = 1.0f;
    cudaMemcpy(d_dc, temp_ones, size, cudaMemcpyHostToDevice);
    delete[] temp_ones;

    swiglu_backward_kernel<<<n_rows, n_cols>>>(d_dc, d_a, d_b, d_da, d_db, stride, n_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(da, d_da, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(db, d_db, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_da);
    cudaFree(d_db);
    cudaFree(d_dc);
}

int main() {
    const int n_rows = 4;
    const int n_cols = 8;
    const int stride = n_cols;
    const int num_ele = n_rows * n_cols;

    float a[num_ele];
    float b[num_ele];
    float c[num_ele];
    float da[num_ele];
    float db[num_ele];

    for(int i = 0; i < num_ele; i++) {
        a[i] = (float)(i % 10) / 10.0f;  
        b[i] = (float)((i + 5) % 10) / 10.0f;  
        c[i] = 0.0f;
        da[i] = 0.0f;
        db[i] = 0.0f;
    }

    do_swiglu(a, b, c, da, db, stride, n_rows, n_cols);

    cout << "c\n";
    for(int i = 0; i < num_ele; i++) {
        cout << c[i] << " ";
    }
    cout << "\n";

    cout << "da \n";
    for(int i = 0; i < num_ele; i++) {
        cout <<  da[i] << " ";
    }
    cout << "\n";

    cout << "db \n";
    for(int i = 0; i < num_ele; i++) {
        cout <<  db[i] << " ";
    }
    cout << "\n";

    return 0;
}