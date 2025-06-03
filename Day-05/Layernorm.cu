#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// y = (x - E[x] / sqrt(Var[x] + eps]) * y + ß 
__global__ void LayerNorm(const float *input, float *output, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows){
        extern __shared__ float row_shared[];
        float* row_data = row_shared;

        for (int col = threadIdx.y; col < cols; col += blockDim.y)
            row_data[col] = input[row * cols + col];
            printf("%f", row_data[col]);

        __syncthreads();

        float mean = 0.0f;
        for (int col = 0; col < cols; col++)
            mean += row_data[col];
        mean /= cols;

        float var = 0.0f;
        for (int col = 0; col < cols; col++)
            var += (row_data[col] - mean) * (row_data[col] - mean);
        var /= cols;
        float eps = 1e-7;
        float std = sqrtf(var + eps);

        for (int col = threadIdx.y; col < cols; col += blockDim.y)
            output[row * cols + col] = (row_data[col] - mean) / std;

    }

}

void print_mat(const float* mat, int rows, int cols){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}


int main(){
    const int rows = 10, cols = 10;
    float *in, *out;

    in = (float*)malloc(rows * cols * sizeof(float));
    out = (float*)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            in[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *da, *db;
    cudaMalloc(&da, rows * cols * sizeof(float));
    cudaMalloc(&db, rows * cols * sizeof(float));

    cudaMemcpy(da, in, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (rows + blocksize - 1) / blocksize;
    size_t shared_mem_size = cols * sizeof(float);
    LayerNorm<<<gridsize, blocksize, shared_mem_size>>>(da, db, rows, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(out, db, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("In(mat)\n");
    print_mat(in, rows, cols);

    printf("Out(mat)\n");
    print_mat(out, rows, cols);

    cudaFree(da);
    cudaFree(db);
    free(in);
    free(out);


}