#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>

__device__ __constant__ float sqrt_2_over_pi = 0.7978845608f;
__device__ __constant__ float eps = 0.044715f;
 

__global__ void geglu_forward_kernel(const float* a, const float* b, float* c, int n){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= n) return;
    int idx = row * n + col;
    float a_val = a[idx];
    float b_val = b[idx];
    
    float a_3 = a_val * a_val * a_val;

    float tanh_val = sqrt_2_over_pi * (a_val + eps * a_3);
    float tanh_result = tanhf(tanh_val);
    float geglu = 0.5f * a_val * (1.0f + tanh_result);
    c[idx] = geglu * b_val;
}

__global__ void geglu_backward_kernel(const float* dc, const float* a, const float* b, float* da, float* db, int n){
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(col >= n) return;
    int idx = row * n + col;
    float dc_val = dc[idx];
    float a_val = a[idx];
    float b_val = b[idx];

    float a_3 = a_val * a_val * a_val;
    float tanh_arg = sqrt_2_over_pi * (a_val + eps * a_3);
    float tanh_result = tanhf(tanh_arg);  
    float geglu = 0.5f * a_val * (1.0f + tanh_result);
    db[idx] = dc_val * geglu;
    float term1 = 0.5f * (1.0f + tanh_result);
    float tanh_2 = tanh_result * tanh_result;
    float term2 = 0.5f * a_val * (1.0f - tanh_2) * 
                (sqrt_2_over_pi * (1.0f + 3.0f * eps * a_val * a_val));
    da[idx] = dc_val * b_val * (term1 + term2);
}

void do_geglu(const float* a, const float* b, float* c, float* dr_a, float* dr_b, int n_rows, int n_cols){

    size_t num_ele = n_rows * n_cols;
    size_t size = num_ele * sizeof(float);
    float *da, *db, *dc, *driv_c, *driv_a, *driv_b;

    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);
    cudaMalloc(&driv_c, size);
    cudaMalloc(&driv_a, size);
    cudaMalloc(&driv_b, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
    
    float *ones = (float*)malloc(size);
    for(int i = 0; i < num_ele; i++) {
        ones[i] = 1.0f;
    }
    cudaMemcpy(driv_c, ones, size, cudaMemcpyHostToDevice);
    free(ones);

    // Add thread block size checking
    int block_size = (n_cols > 1024) ? 1024 : n_cols;
    int grid_size = n_rows;
    
    if (n_cols > 1024) {
        grid_size = (n_rows * n_cols + 1023) / 1024;
        block_size = 1024;
    }

    geglu_forward_kernel<<<grid_size, block_size>>>(da, db, dc, n_cols);
    cudaDeviceSynchronize(); 
    
    geglu_backward_kernel<<<grid_size, block_size>>>(driv_c, da, db, driv_a, driv_b, n_cols);
    cudaDeviceSynchronize(); 

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dr_a, driv_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dr_b, driv_b, size, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(driv_a);
    cudaFree(driv_b);
    cudaFree(driv_c);
}

void print_mat(const float *data, int row, int col, const char* name){
    printf("%s:\n", name);  // Fixed spacing
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            printf("%.4f ", data[i * col + j]);  
        }
        printf("\n");
    }
    printf("\n");  
}

int main(){
    const int n_rows = 3;
    const int n_cols = 3;
    const int n_ele = n_rows * n_cols;
    float a[n_ele], b[n_ele], c[n_ele];
    float dr_a[n_ele], dr_b[n_ele];

    srand(42);
    
    for (int i = 0; i < n_ele; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;  
        b[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    
    do_geglu(a, b, c, dr_a, dr_b, n_rows, n_cols);

    print_mat(a, n_rows, n_cols, "Input A");
    print_mat(b, n_rows, n_cols, "Input B");
    print_mat(c, n_rows, n_cols, "Output C (GEGLU result)");
    print_mat(dr_a, n_rows, n_cols, "Grad A");
    print_mat(dr_b, n_rows, n_cols, "Grad B");
    
    return 0;
}