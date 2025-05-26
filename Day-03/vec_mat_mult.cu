#include <iostream>

__global__ void vec_mat_mult(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        float sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += a[i * n + j] * b[j];
        c[i] = sum;
    }

}

void print_mat(const float* mat, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%.2f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

void print_vec(const float* vec, int n){
    for (int i = 0; i < n; i++)
        printf("%.2f ", vec[i]);
    printf("\n");
}

int main(){

    const int n = 10;
    float *a, *b, *c;

    a = (float *)malloc(n * n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));
    c = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++)
            a[i * n + j] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    float *da, *db, *dc;

    cudaMalloc(&da, n*n*sizeof(float));
    cudaMalloc(&db, n*sizeof(float));
    cudaMalloc(&dc, n*sizeof(float));

    cudaMemcpy(da, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int gridsize = (n + block_size - 1) / block_size; //ceil

    vec_mat_mult<<<gridsize, block_size>>>(da, db, dc, n);

    cudaDeviceSynchronize();
    cudaMemcpy(c, dc, n*sizeof(float), cudaMemcpyDeviceToHost);

    printf("A(mat):\n");
    print_mat(a, n);

    printf("B(vec):\n");
    print_vec(b, n);

    printf("C(vec):\n");
    print_vec(c, n);

}